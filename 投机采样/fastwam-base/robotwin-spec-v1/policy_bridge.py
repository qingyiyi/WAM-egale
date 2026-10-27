"""Use the pinned RoboTwin wrapper's exact input preparation and postprocessing.

This extracts only the statements before its inference call. It does not
replace the installed wrapper or change the canonical model method.
"""
import ast
import inspect
import textwrap
import types
from pathlib import Path

import numpy as np
import torch


def install_policy_bridge(policy, output):
    original = type(policy)._infer_action_chunk
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    boundary = [i for i, node in enumerate(function.body)
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == 'infer_t0' for t in node.targets)]
    assert len(boundary) == 1, 'pinned RoboTwin wrapper changed'
    prefix = function.body[:boundary[0]]
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr == 'infer_action' for node in prefix for n in ast.walk(node))
    function.name = 'prepare_robotwin_kwargs'
    function.returns = None
    function.body = prefix + ast.parse('return infer_kwargs').body
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    output = Path(output)
    output.write_text(ast.unparse(module) + '\n')
    namespace = dict(original.__globals__)
    exec(compile(module, str(output), 'exec'), namespace)
    policy.prepare_robotwin_kwargs = types.MethodType(namespace[function.name], policy)


def prepare(policy, observation, instruction):
    kwargs = policy.prepare_robotwin_kwargs(observation, instruction)
    assert kwargs['action_horizon'] == 32 and kwargs['num_inference_steps'] == 10
    assert kwargs['sigma_shift'] == 5.0 and kwargs['seed'] == 42
    assert kwargs['text_cfg_scale'] == 1.0 and kwargs['rand_device'] == 'cpu'
    assert kwargs['input_image'].shape == (1, 3, 384, 320)
    assert kwargs['proprio'].shape == (1, 14)
    return kwargs


def postprocess(policy, result):
    # The official RoboTwin wrapper applies exactly one zscore inverse and no
    # LIBERO gripper sign/binary conversion. Preserve the continuous qpos values.
    value = policy._denormalize_action(result['action'])[0]
    assert value.shape == (32, 14) and np.isfinite(value).all()
    return value
