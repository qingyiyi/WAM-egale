"""Real-tensor, numerically checked software timing; no hardware counters."""
from pathlib import Path
from datetime import datetime
import json
import os
import statistics
import time
import traceback
import gc

import torch
import measure as m

P = Path(__file__).resolve().parent
out = P / "results/detail"
out.mkdir(parents=True, exist_ok=False)
job = json.loads((P / "JOB.json").read_text())
deadline = datetime.fromisoformat(job["absolute_cleanup_deadline"]).timestamp() - 300
torch.set_num_threads(4)
torch.set_num_interop_threads(1)
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
result = {"status": "RUNNING", "calls": 0, "rows": [], "registered_job": job,
          "scope": "instrumentation only; no production performance decision",
          "fields": json.loads((P / "COST-FIELDS.json").read_text())}

def save():
    (out / "RESULT.json").write_text(json.dumps(result, indent=2))

def tick():
    result["calls"] += 1
    if time.time() > deadline or result["calls"] > min(300, job["calls_cap"]-job["calls_before_phase"]):
        raise TimeoutError("cost detail wall/call bound")

def binary(name):
    return next((P / "build" / name).glob("*.so"))

try:
    if m.resource()["processes"]:
        raise RuntimeError("visible GPU context exists")
    libs = {
        name: m.load_extension(binary(name), "wam_sm120_cluster2_" + name + "_cuda")
        for name in ["c2", "g2", "c2_detail", "g2_detail"]
    }
    block = m.load_block()
    bind = m.WholeLayerBlock(block)
    result["modules"] = {k: v.__file__ for k, v in libs.items()}
    result["before"] = m.resource()
    with torch.inference_mode():
        for length in ["short", "middle", "long"]:
            f = m.BlockFixture(
                m.FIXTURES / f"input-{length}-B14-action-layer0-first.pt",
                m.device_tree(m.decode(torch.load(
                    m.FIXTURES / f"input-{length}-B14-action-layer0-first.pt",
                    map_location="cpu", mmap=True, weights_only=False,
                ))),
            )
            a, kw = f.arguments()
            cache = f.install_cache(block)
            initial = m.visible_snapshot(cache)
            inputs = {k: v.clone() for k, v in m.tensors((a, kw))}
            rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
            with m.capture(block) as ref:
                ref["output"] = block(*a, **kw)
                tick()
            bind.prepare_context(a[1], precompute=True)
            fullkv = [v.index_select(0, bind.context_rows) for v in bind.context_kv]
            assert all(torch.equal(ref[k], v) for k, v in zip(
                ["cross_k_linear", "cross_k", "cross_v"], fullkv))
            cl = [cache[k] for k in ["k", "v", "id", "mask", "is_pred"]]
            common = (bind.weights, *a, cl, kw["candidate_cache_rows"], 0, bind.eps)

            def own(name, debug=False):
                tick()
                return libs[name].forward(
                    *common, debug, bind.context_rows, bind.context_roots, bind.context_kv)

            row = {"case": length, "precision": {}, "resources": {}, "seven_segments": {},
                   "timing_samples": [], "cost_samples": {}}
            result["rows"].append(row)
            golden = torch.load(P.parent / "T1-r1/data" / ("numeric-" + length + ".pt"),
                                map_location="cpu", weights_only=False)
            for name in libs:
                y = own(name, True)
                assert not (set(ref) - set(y)), (name, set(ref)-set(y))
                row["precision"][name] = {
                    k: m.compare(ref[k].reshape(-1), y[k].reshape(-1)) for k in ref}
                assert all(x["bitwise_pass"] for x in row["precision"][name].values())
                row["resources"][name] = y["resources"].tolist()
                assert y["ffn_splitk_commit"].tolist() == [7] * 96
                if name.endswith("_detail"):
                    trace = y["ffn_numeric"].cpu()
                    row["seven_segments"][name] = [
                        {"s": s, "partial": m.compare(golden[s, 0], trace[s, 0]),
                         "running": m.compare(golden[s, 1], trace[s, 1])}
                        for s in range(7)]
                    assert all(x["partial"]["bitwise_pass"] and x["running"]["bitwise_pass"]
                               for x in row["seven_segments"][name])
                    del trace
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    custom = own(name)["output"]
                stream.synchronize()
                assert m.compare(ref["output"], custom)["bitwise_pass"]
            del golden
            row["pre_timing_state"] = {
                "valid_KV": m.state_equal(initial, m.visible_snapshot(cache)),
                "inputs_unchanged": all(m.byte_equal(inputs[k], v) for k, v in m.tensors((a, kw))),
                "RNG_unchanged": m.byte_equal(rng[0], torch.get_rng_state()) and m.byte_equal(rng[1], torch.cuda.get_rng_state()),
            }
            assert row["pre_timing_state"]["valid_KV"]["pass_all"]
            assert row["pre_timing_state"]["inputs_unchanged"] and row["pre_timing_state"]["RNG_unchanged"]
            save()
            for name in libs:
                for _ in range(3):
                    own(name)
            for repeat in range(6):
                order = list(libs) if repeat % 2 == 0 else list(libs)[::-1]
                for name in order:
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    begin.record()
                    for _ in range(2):
                        y = own(name)
                    end.record()
                    end.synchronize()
                    assert m.compare(ref["output"], y["output"])["bitwise_pass"]
                    row["timing_samples"].append({
                        "rep": repeat, "arm": name, "event_ms": begin.elapsed_time(end)/2})
            row["median_ms"] = {
                name: statistics.median(x["event_ms"] for x in row["timing_samples"] if x["arm"]==name)
                for name in libs}
            for name in ["c2_detail", "g2_detail"]:
                row["cost_samples"][name] = []
                for _ in range(3):
                    y = own(name)  # debug=False: no extra F1 intermediate materialization
                    costs = y["hybrid_cost"].cpu()
                    assert tuple(costs.shape) == (336, 2, 16)
                    assert (costs[:, :, 13] > costs[:, :, 0]).all()
                    for j in range(336):
                        assert costs[j, :, 15].tolist() == [j, j]
                    row["cost_samples"][name].append({
                        "cost": costs.tolist(), "stage_ticks": y["stage_ticks"].cpu().tolist()})
            row["post_state"] = {
                "valid_KV": m.state_equal(initial, m.visible_snapshot(cache)),
                "inputs_unchanged": all(m.byte_equal(inputs[k], v) for k,v in m.tensors((a,kw))),
                "RNG_unchanged": m.byte_equal(rng[0], torch.get_rng_state()) and m.byte_equal(rng[1], torch.cuda.get_rng_state()),
            }
            assert row["post_state"]["valid_KV"]["pass_all"]
            assert row["post_state"]["inputs_unchanged"] and row["post_state"]["RNG_unchanged"]
            row["after"] = m.resource()
            save()
            del ref, f, a, kw, cache, cl, initial, inputs, fullkv, y, costs
            block.attn1.attn_caches = {}
            bind.context_kv = []
            gc.collect()
            torch.cuda.empty_cache()
    result["status"] = "COMPLETE_NUMERIC_SOFTWARE_COST"
except BaseException as exc:
    result.update(status="FAILED_OR_PARTIAL", error=repr(exc), traceback=traceback.format_exc())
finally:
    save()
    print(result["status"], result.get("error", ""), flush=True)
if result["status"] != "COMPLETE_NUMERIC_SOFTWARE_COST":
    raise SystemExit(1)
