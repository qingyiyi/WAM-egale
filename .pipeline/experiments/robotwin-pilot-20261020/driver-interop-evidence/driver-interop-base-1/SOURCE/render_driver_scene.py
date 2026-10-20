"""Run one fixed, complete official expert scene with its original renderer."""
from pathlib import Path
import ast, datetime, importlib, json, os, sys, time, traceback, types

R = Path('/root/cxz/robotwin')
J = Path(sys.argv[1])
J.mkdir(exist_ok=False)
os.sched_setaffinity(0, {16, 17})
started = time.time()
state = dict(status='RUNNING', start=started, phase='IMPORT', task='place_object_basket',
             seed=10000, policy_episodes=0, teacher_forwards=0, expert_complete=False)

def libraries():
    return sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                   if any(x in line for x in ('libcuda.', 'libOpenImageDenoise', 'libvulkan',
                                              'libGLX_nvidia', 'libnvidia-'))})

def save(phase=None):
    if phase is not None:
        state['phase'] = phase
    state['loaded_libraries'] = libraries()
    (J / 'STATUS.json').write_text(json.dumps(state, indent=2) + '\n')
    print(json.dumps({k: state.get(k) for k in ('status', 'phase', 'expert_complete')}), flush=True)

try:
    import numpy as np
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.cuda.init()
    state.update(torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                 cuda_name=torch.cuda.get_device_name(0))
    save('CUDA_READY')
    from device_binding import install
    state['render_device'] = install(J / 'RENDER-DEVICE.json')
    S = R / 'source/RoboTwin-pinned'
    L = R / 'source/lingbot-va-pinned'
    for root in (L, S):
        sys.path.insert(0, str(root))
    os.chdir(S)
    client = L / 'evaluation/robotwin/eval_polict_client_openpi.py'
    tree = ast.parse(client.read_text())
    class Root(ast.NodeTransformer):
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'robowin_root' for t in node.targets):
                node.value = ast.Call(func=ast.Name(id='Path', ctx=ast.Load()),
                                      args=[ast.Constant(str(S))], keywords=[])
            return self.generic_visit(node)
    module = types.ModuleType('_robotwin_driver_scene')
    module.__file__ = str(client)
    sys.modules[module.__name__] = module
    exec(compile(ast.fix_missing_locations(Root().visit(tree)), str(client), 'exec'), module.__dict__)
    save('OFFICIAL_CLIENT_READY')

    def expert(task_name, env, args, model, st_seed, **kwargs):
        if task_name != state['task'] or st_seed != state['seed']:
            raise RuntimeError(f'fixed case mismatch: {task_name}/{st_seed}')
        args = dict(args, eval_mode=True, render_freq=0, eval_video_log=False)
        save('SETUP_ORIGINAL_SCENE')
        env.setup_demo(now_ep_num=0, seed=st_seed, is_test=True, **args)
        save('PLAY_COMPLETE_EXPERT')
        info = env.play_once()
        state.update(expert_complete=True, plan_success=bool(env.plan_success),
                     expert_success=bool(env.check_success()))
        save('EXPERT_TERMINAL')
        env.close_env()
        save('RESET_SAME_SCENE_FOR_OBSERVATION')
        env.setup_demo(now_ep_num=0, seed=st_seed, is_test=True, **args)
        observation = env.get_obs()
        torch.save(observation, J / 'OBSERVATION.pt')
        def describe(value):
            if isinstance(value, dict):
                return {str(k): describe(v) for k, v in value.items()}
            if isinstance(value, (np.ndarray, torch.Tensor)):
                return dict(shape=list(value.shape), dtype=str(value.dtype))
            return type(value).__name__
        state['observation'] = describe(observation)
        save('ORIGINAL_OBSERVATION_SAVED')
        env.close_env(clear_cache=True)
        return st_seed, int(state['expert_success'])

    module.eval_policy = expert
    module.WebsocketClientPolicy = lambda **kwargs: None
    module.main(dict(task_name=state['task'], task_config='demo_clean',
                     ckpt_setting='M40-driver-interop-repair', save_root=str(J / 'official-output'),
                     policy_name='LingBotVA-original', seed=0, port=34956,
                     video_guidance_scale=5, action_guidance_scale=1, test_num=1))
    state['status'] = 'COMPLETE_ORIGINAL_EXPERT_AND_OBSERVATION'
except BaseException as exc:
    state.update(status='FAILED_OR_PARTIAL_SCENE', error=repr(exc), traceback=traceback.format_exc())
finally:
    state.update(end=time.time(), lifecycle_seconds=time.time() - started)
    save()
