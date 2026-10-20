"""Pinned original RoboTwin Target, complete text bank and observed teacher rows.

No speculative route is installed here. This is the first six FIT episodes.
Canonical model, scheduler, CFG, latent/RNG allocation and action mapping stay
in the pinned source. Observation hooks do not replace their outputs.
"""
from pathlib import Path
import copy
import json
import os
import random
import sys
import time
import traceback
import types

import numpy as np
import torch

R = Path("/root/cxz/robotwin")
L = R / "source/lingbot-va-pinned"
J = Path(sys.argv[1])
J.mkdir(parents=True, exist_ok=True)
os.sched_setaffinity(0, {20, 21})
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
sys.path.insert(0, str(L))
sys.path.insert(0, str(L / "wan_va"))
import wan_va_server as original

state = {"status": "LOADING", "start": time.time(), "teacher_attempted": 0,
         "teacher_returned": 0, "policy_chunks": 0, "episodes_started": 0,
         "evaluation_seed": 0, "text_bank_hits": 0, "text_bank_misses": 0}


def save():
    p = J / "STATUS.json"
    p.with_suffix(".tmp").write_text(json.dumps(state, indent=2) + "\n")
    p.with_suffix(".tmp").replace(p)


def cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu(v) for v in value]
    return value


def event(row):
    with (J / "requests.jsonl").open("a") as stream:
        stream.write(json.dumps(row) + "\n")


try:
    original.init_logger()
    config = copy.deepcopy(original.VA_CONFIGS["robotwin"])
    config.wan22_pretrained_model_name_or_path = str(R / "models/target")
    config.save_root = str(J / "original-debug-directories")
    config.rank = config.local_rank = 0
    config.world_size = 1
    # Pinned default: GPU VAE and GPU T5 for building the exact 512-token bank.
    # Only after bank construction is T5 moved off GPU; no online CPU encoding.
    assert config.enable_offload is False
    original.init_distributed(1, 0, 0)
    server = original.VA_Server(config)
    manifest = json.loads((R / "runs/prefilter-4/STATUS.json").read_text())
    if manifest["status"] != "COMPLETE_FIXED_SCENES_NO_POLICY_QUALITY_RESULT":
        raise RuntimeError("Expert manifest is not complete")
    prompts = sorted({row["instruction"] for task in manifest["tasks"] for row in task["valid"]} | {""})
    bank = {}
    encode = server._get_t5_prompt_embeds
    state["status"] = "BUILDING_FULL_512_TEXT_BANK"
    state["bank_prompt_count"] = len(prompts)
    save()
    with torch.no_grad():
        for prompt in prompts:
            before = torch.cuda.get_rng_state()
            value = encode(prompt=prompt, max_sequence_length=512, device=server.device, dtype=server.dtype)
            if value.shape != (1, 512, 4096) or value.dtype != torch.bfloat16:
                raise RuntimeError("Original full text shape/dtype differs")
            if not torch.equal(before, torch.cuda.get_rng_state()):
                raise RuntimeError("Text construction consumed CUDA RNG")
            bank[prompt] = value.detach().cpu().clone()
            if not torch.equal(bank[prompt].to(server.device), value):
                raise RuntimeError("Text bank round-trip differs")
            state["bank_completed"] = len(bank)
            save()
    torch.save({"schema": "robotwin-original-full512-text-bank/v1",
                "source": "7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb",
                "target_revision": "8c9dea8abbc5c91cc9e18bc3264b8915083bbe70",
                "dtype": "bfloat16", "sequence_length": 512, "embeddings": bank}, J / "text-bank.pt")
    del value
    server.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    def lookup(_self, prompt=None, num_videos_per_prompt=1, max_sequence_length=512,
               device=None, dtype=None):
        if isinstance(prompt, list) and len(prompt) == 1:
            prompt = prompt[0]
        if num_videos_per_prompt != 1 or max_sequence_length != 512 or prompt not in bank:
            state["text_bank_misses"] += 1
            save()
            raise RuntimeError("Unregistered text request; no silent bank fallback")
        if dtype is not None and dtype != torch.bfloat16:
            raise RuntimeError("Text-bank dtype differs")
        state["text_bank_hits"] += 1
        return bank[prompt].to(device=device or _self.device, dtype=torch.bfloat16).clone()

    server._get_t5_prompt_embeds = types.MethodType(lookup, server)
    # The original save_async is debug-only. Keep bounded, explicit collection
    # below instead of copying every camera and latent into debug directories.
    original.save_async = lambda *args, **kwargs: None
    current = {"case": None, "route": None, "chunk": 0, "rows": [], "pending": None,
               "episode_forwards": 0, "scheduler_updates": 0, "previous": None}
    channel_ids = list(config.used_action_channel_ids)
    canonical_forward = server.transformer.forward
    canonical_step = server.action_scheduler.step
    canonical_infer = server.infer

    def forward(*args, **kwargs):
        if current["episode_forwards"] >= 8000:
            raise RuntimeError("Pre-registered episode teacher forward cap")
        current["episode_forwards"] += 1
        state["teacher_attempted"] += 1
        action = bool(kwargs.get("action_mode", False))
        update = int(kwargs.get("update_cache", 0))
        key = ("action" if action else "video") + f"_update{update}"
        counts = current.setdefault("counts", {})
        counts[key + "_attempted"] = counts.get(key + "_attempted", 0) + 1
        collected_input = None
        if current["route"] == "policy" and action and update == 0:
            d = args[0]
            collected_input = {
                "state_full": d["noisy_latents"][:1].detach().clone(),
                "timesteps": d["timesteps"].detach().clone(),
                "grid_id": d["grid_id"].detach().clone()}
        out = canonical_forward(*args, **kwargs)
        state["teacher_returned"] += 1
        counts[key + "_returned"] = counts.get(key + "_returned", 0) + 1
        if collected_input is not None:
            collected_input["teacher_raw_CFG2"] = out.detach().clone()
            current["pending"] = collected_input
        if state["teacher_attempted"] % 25 == 0:
            save()
        return out

    def step(direction, timestep, sample, *args, **kwargs):
        pending = current["pending"]
        if pending is None:
            raise RuntimeError("Scheduler update lacks matching observed teacher query")
        index = current["scheduler_updates"]
        sigmas = server.action_scheduler.sigmas
        actual_index = int(torch.argmin((server.action_scheduler.timesteps - timestep.cpu()).abs()))
        if actual_index != index:
            raise RuntimeError("Observed scheduler timestep order differs")
        next_sigma = sigmas[index + 1] if index + 1 < len(sigmas) else sigmas.new_tensor(0)
        pending.update(
            sigma=sigmas[index].detach().clone(),
            next_sigma=next_sigma.detach().clone(),
            timestep=timestep.detach().clone(),
            teacher_direction_full=direction.detach().clone(),
            sample_before=sample.detach().clone(),
            previous_target_velocity=(torch.zeros_like(direction) if current["previous"] is None
                                      else current["previous"]).clone(),
            effective_channel_ids=channel_ids)
        raw = pending["teacher_raw_CFG2"]
        raw = raw.permute(0, 2, 1).reshape(raw.shape[0], 30, 2, 16, 1)
        expected = (raw[1:] + config.action_guidance_scale * (raw[:1] - raw[1:])
                    if config.action_guidance_scale > 1 else raw[:1])
        if not torch.equal(expected, direction):
            raise RuntimeError("Observed teacher CFG label differs from actual scheduler input")
        result = canonical_step(direction, timestep, sample, *args, **kwargs)
        pending["sample_after"] = result.detach().clone()
        current["rows"].append(pending)
        current["previous"] = direction.detach().clone()
        current["pending"] = None
        current["scheduler_updates"] += 1
        return result

    server.transformer.forward = forward
    server.action_scheduler.step = step

    def infer(_self, obs):
        case = obs.get("_audit_case")
        if obs.get("reset"):
            if not case:
                raise RuntimeError("Reset missing registered case identity")
            current.update(case=case, chunk=0, episode_forwards=0)
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            state["episodes_started"] += 1
            e = J / case
            e.mkdir(exist_ok=False)
            torch.save({"CPU": torch.get_rng_state(), "CUDA": torch.cuda.get_rng_state_all(),
                        "python": random.getstate(), "numpy": np.random.get_state()},
                       e / "RNG-before-reset.pt")
        route = "reset" if obs.get("reset") else ("cache" if obs.get("compute_kv_cache") else "policy")
        current.update(route=route, counts={}, rows=[], pending=None,
                       scheduler_updates=0, previous=None)
        start = time.time()
        result = canonical_infer(obs)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        if route == "policy":
            if current["scheduler_updates"] != 50 or len(current["rows"]) != 50:
                raise RuntimeError("Original full action50 coverage differs")
            torch.save(cpu({"rows": current["rows"], "decoded_action": result["action"],
                            "case": current["case"], "chunk": current["chunk"],
                            "source_kind": "actual_target_visited_query",
                            "teacher_label": "exact direction passed to original scheduler",
                            "full_conditions_for_teacher_requery": False}),
                       J / current["case"] / f"chunk-{current['chunk']:04d}.pt")
            current["chunk"] += 1
            state["policy_chunks"] += 1
        event({"case": current["case"], "route": route, "chunk_count": current["chunk"],
               "counts": current["counts"], "valid_updates": current["scheduler_updates"],
               "elapsed_seconds": elapsed, "time_kind": "collection_instrumented_not_formal_performance",
               "teacher_episode_attempted": current["episode_forwards"],
               "text_bank_hits": state["text_bank_hits"]})
        current["rows"] = []
        state.update(status="SERVING", active_case=current["case"], active_route=route,
                     active_chunk=current["chunk"], last_request_end=time.time())
        save()
        return result

    server.infer = types.MethodType(infer, server)
    state.update(status="READY", ready=time.time(), bank_bitwise_roundtrip=True,
                 original_full_episode_cached_equivalence="NOT_YET_MEASURED",
                 target_forwards_during_setup=0,
                 cuda_allocated=torch.cuda.memory_allocated())
    save()
    original.run_async_server_mode(server, 0, "127.0.0.1", 34956)
except BaseException as error:
    state.update(status="FAILED", error=repr(error), traceback=traceback.format_exc(), end=time.time())
    save()
    raise
