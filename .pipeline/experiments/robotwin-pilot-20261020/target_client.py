"""One complete original policy episode on a frozen expert-valid FIT scene."""
from pathlib import Path
import ast
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
S = R / "source/RoboTwin-pinned"
case = json.loads(Path(sys.argv[1]).read_text())
J = Path(sys.argv[2])
J.mkdir(exist_ok=False)
os.sched_setaffinity(0, {16, 17})
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
state = {"status": "INITIALIZING", "start": time.time(), "case": case,
         "policy_chunks": 0, "environment_commands": 0, "requests": []}


def save():
    p = J / "STATUS.json"
    p.with_suffix(".tmp").write_text(json.dumps(state, indent=2) + "\n")
    p.with_suffix(".tmp").replace(p)


def equal_tree(a, b):
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b)
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_tree(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return bool(a == b)


try:
    torch.cuda.init()
    from device_binding import install
    state["render_device"] = install(J / "RENDER-DEVICE.json")
    for p in (L, S):
        sys.path.insert(0, str(p))
    os.chdir(S)
    source = L / "evaluation/robotwin/eval_polict_client_openpi.py"
    tree = ast.parse(source.read_text())
    # Keep the original complete action loop and its command/cache semantics.
    # Replace only expert selection and instructions with the frozen manifest.
    class FrozenCase(ast.NodeTransformer):
        def visit_Assign(self, node):
            names = [x.id for x in node.targets if isinstance(x, ast.Name)]
            if "robowin_root" in names:
                node.value = ast.Call(func=ast.Name(id="Path", ctx=ast.Load()),
                                      args=[ast.Constant(str(S))], keywords=[])
            if "expert_check" in names:
                node.value = ast.Constant(False)
            if "now_seed" in names:
                node.value = ast.Constant(case["seed"])
            if "episode_info_list" in names:
                node.value = ast.List(elts=[], ctx=ast.Load())
            if "results" in names and isinstance(node.value, ast.Call):
                node.value = ast.List(elts=[], ctx=ast.Load())
            if "instruction" in names:
                node.value = ast.Constant(case["instruction"])
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(FrozenCase().visit(tree))
    module = types.ModuleType("_robotwin_target_client")
    module.__file__ = str(source)
    sys.modules[module.__name__] = module
    exec(compile(tree, str(source), "exec"), module.__dict__)
    frozen = torch.load(case["initial"], map_location="cpu", weights_only=False)
    original_class = module.class_decorator

    def create(task):
        env = original_class(task)
        set_instruction = env.set_instruction
        take_action = env.take_action

        def instruction(*args, **kwargs):
            result = set_instruction(*args, **kwargs)
            from prefilter_physical import physical
            rng = frozen["RNG"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_CPU"])
            torch.cuda.set_rng_state_all(rng["torch_CUDA"])
            obs = env.get_obs()
            checks = {"physical_exact": equal_tree(physical(env), frozen["physical"]),
                      "initial_observation_exact": equal_tree(obs, frozen["initial_observation"])}
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch_CPU"])
            torch.cuda.set_rng_state_all(rng["torch_CUDA"])
            torch.save({"physical": physical(env), "observation": obs, "RNG": rng},
                       J / "INITIAL-ACTUAL.pt")
            state["initial_checks"] = checks
            state["task_step_limit"] = int(env.step_lim)
            save()
            if not all(checks.values()):
                raise RuntimeError("Frozen initial scene/observation mismatch")
            return result

        def command(action, *args, **kwargs):
            result = take_action(action, *args, **kwargs)
            state["environment_commands"] += 1
            with (J / "commands.jsonl").open("a") as output:
                output.write(json.dumps({"index": state["environment_commands"],
                                         "command": np.asarray(action).tolist(),
                                         "environment_take_action_cnt": int(env.take_action_cnt)}) + "\n")
            return result

        env.set_instruction = instruction
        env.take_action = command
        return env

    module.class_decorator = create
    original_policy = module.eval_policy
    def policy(*args, **kwargs):
        kwargs.update(test_num=1, save_visualization=False)
        args[2]["eval_video_log"] = False
        args[2].pop("eval_video_save_dir", None)
        return original_policy(*args, **kwargs)
    module.eval_policy = policy
    module.save_comparison_video = lambda *args, **kwargs: None
    original_client = module.WebsocketClientPolicy

    class Client(original_client):
        def infer(self, request):
            route = "reset" if request.get("reset") else ("cache" if request.get("compute_kv_cache") else "policy")
            if route == "reset":
                request["_audit_case"] = case["task"] + "-valid00"
            start = time.time()
            state.update(status="RUNNING_COMPLETE_EPISODE", active_request=route)
            save()
            result = super().infer(request)
            if route == "policy":
                state["policy_chunks"] += 1
            state["requests"].append({"route": route, "elapsed": time.time() - start})
            save()
            return result
    module.WebsocketClientPolicy = Client
    module.main(dict(task_name=case["task"], task_config="demo_clean",
                     ckpt_setting="M40-Target-first6", save_root=str(J / "official-output"),
                     policy_name="LingBotVA-original", seed=0, port=34956,
                     video_guidance_scale=5, action_guidance_scale=1, test_num=1))
    metrics = list((J / "official-output").glob("stseed-*/metrics/*/res.json"))
    if len(metrics) != 1:
        raise RuntimeError("Missing original complete-episode metrics")
    state["original_metrics"] = json.loads(metrics[0].read_text())
    state["success"] = state["original_metrics"]["succ_num"] == 1
    state["failure_penalty"] = (state["environment_commands"] if state["success"]
                                else state["task_step_limit"])
    state["status"] = "COMPLETE_SUCCESS" if state["success"] else "COMPLETE_ALGORITHMIC_FAILURE"
except BaseException as error:
    state.update(status="PARTIAL_OR_INVALID", error=repr(error), traceback=traceback.format_exc())
finally:
    state.update(end=time.time(), lifecycle_seconds=time.time() - state["start"])
    save()
