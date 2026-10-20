"""Current B14 stage-clock attribution; never used as ordinary timing."""
from pathlib import Path
import importlib.util
import json
import os
import time
import traceback

import torch
from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture, decode, visible_snapshot
from tools.metrics import compare, state_equal
from tools.replay import device_tree
from wam_ops.optimized.native import NativeBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock

B = Path(__file__).resolve().parent
os.sched_setaffinity(0, {4, 5})
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
t0 = time.time()
result = dict(status="RUNNING", full_block_calls=0, primitive_calls=0, rows=[],
              profiling_only=True, production_admitted=False,
              timing_boundary="stage including boundary wait vs standalone Torch Linear",
              hardware_counters="UNKNOWN", context_tile_rows=128)


def save():
    p = B / "RESULT.json"
    p.with_suffix(".tmp").write_text(json.dumps(result, indent=2) + "\n")
    p.with_suffix(".tmp").replace(p)


try:
    path = next(B.glob("wam_stage_profile_cuda*.so"))
    spec = importlib.util.spec_from_file_location("wam_stage_profile_cuda", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    block = load_block("cuda")
    native, binding = NativeBlock(block), WholeLayerBlock(block)
    with torch.inference_mode():
        for length in ("short", "middle", "long"):
            p = FIXTURES / f"input-{length}-B14-action-layer0-first.pt"
            f = BlockFixture(p, device_tree(decode(torch.load(
                p, map_location="cpu", mmap=True, weights_only=False))))
            args, kw = f.arguments()
            f.install_cache(block)
            ref = native(*args, **kw, debug=True)
            result["full_block_calls"] += 1
            expected_cache = visible_snapshot(block.attn1.attn_caches[kw["cache_name"]])
            row = dict(case=length, fixture=str(p), profiles=[], primitives=[])
            for repeat in range(3):
                f.install_cache(block)
                cache = block.attn1.attn_caches[kw["cache_name"]]
                values = [cache[k] for k in ("k", "v", "id", "mask", "is_pred")]
                value = mod.forward(binding.weights, *args, values,
                                    kw.get("candidate_cache_rows"), kw.get("update_cache", 0),
                                    binding.eps, False, True, False)
                result["full_block_calls"] += 1
                torch.cuda.synchronize()
                matched = {}
                for key in ref.keys() & value.keys():
                    if key not in ("tile_stats", "cache_info", "resources",
                                   "logical_slots", "attention_meta", "ffn_schedule"):
                        matched[key] = compare(ref[key].reshape(-1), value[key].reshape(-1))
                cache_match = state_equal(expected_cache, visible_snapshot(
                    block.attn1.attn_caches[kw["cache_name"]]))
                row["profiles"].append(dict(repeat=repeat, compared=matched,
                    cache_equal=cache_match, resources=value["resources"].tolist(),
                    stage_ticks_ns=value["stage_ticks"].cpu().tolist()))
                if not all(x["bitwise_pass"] for x in matched.values()) or not cache_match["pass_all"]:
                    raise RuntimeError("Instrumented current source changed finite contract")
            components = [
                ("Q_proj", value["modulated1"], block.attn1.to_q, "self_q_linear"),
                ("K_proj", value["modulated1"], block.attn1.to_k, "self_k_linear"),
                ("V_proj", value["modulated1"], block.attn1.to_v, "self_v"),
                ("self_output_proj", value["self_attention"], block.attn1.to_out[0], "self_out"),
                ("cross_Q_proj", value["norm2"], block.attn2.to_q, "cross_q_linear"),
                ("cross_K_proj", args[1], block.attn2.to_k, "cross_k_linear"),
                ("cross_V_proj", args[1], block.attn2.to_v, "cross_v"),
                ("cross_output_proj", value["cross_attention"], block.attn2.to_out[0], "cross_out"),
                ("FFN1", value["modulated3"], block.ffn.net[0].proj, "ffn1"),
                ("FFN2", value["gelu"], block.ffn.net[2], "ffn2")]
            for label, x, layer, key in components:
                x = x.reshape(-1, x.shape[-1])
                y = layer(x)
                result["primitive_calls"] += 1
                rec = dict(label=label, M=x.shape[0], N=layer.weight.shape[0],
                           K=x.shape[1], same_input=compare(y, value[key].reshape_as(y)),
                           stream_ms=[])
                for repeat in range(3):
                    a, z = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    y = layer(x)
                    result["primitive_calls"] += 1
                    z.record()
                    z.synchronize()
                    rec["stream_ms"].append(a.elapsed_time(z))
                row["primitives"].append(rec)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                for label, x, layer, _ in components:
                    with torch.profiler.record_function("component::" + label):
                        layer(x.reshape(-1, x.shape[-1]))
                        result["primitive_calls"] += 1
                torch.cuda.synchronize()
            prof.export_chrome_trace(str(B / f"{length}-components.trace.json"))
            row["primitive_kernel_events"] = [
                dict(name=e.name, us=e.time_range.elapsed_us())
                for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            row["primitive_scopes"] = [
                dict(name=e.name, device_us=e.device_time_total)
                for e in prof.events() if e.name.startswith("component::")]
            result["rows"].append(row)
            save()
            del ref, value, args, kw, f, components, x, y
    assert result["full_block_calls"] == 12 and result["primitive_calls"] == 150
    result["status"] = "COMPLETE_CURRENT_B14_PROFILE"
except BaseException as error:
    result.update(status="FAILED_OR_PARTIAL", error=repr(error), traceback=traceback.format_exc())
finally:
    result["lifecycle_seconds"] = time.time() - t0
    save()
