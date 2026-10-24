"""Original complete LIBERO client plus benchmark identity and initial-state audit."""
from pathlib import Path
import importlib.util,json,os,sys
import numpy as np

P=Path('/root/autodl-tmp/cxz/spec-speed-20261025')
suite,task,arm,out=sys.argv[1:5];task=int(task);out=Path(out)
script=P/'bundle/project/投机采样/scripts/p0_libero_multichunk_client.py'
spec=importlib.util.spec_from_file_location('original_complete_libero_client',script)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
case=f'{suite}-task{task}-init49'
transport=original._install_transport
def install(worktree):
    cls,benchmark=transport(worktree)
    import robosuite,mujoco
    out.mkdir(parents=True,exist_ok=True)
    (out/'SIMULATOR-IDENTITY.json').write_text(json.dumps(dict(robosuite_actual_version=getattr(robosuite,'__version__','UNKNOWN'),robosuite_source=robosuite.__file__,mujoco_version=mujoco.__version__,source_scope='Actual imported source copied from existing BAAI client; metadata version alone is not source identity'),indent=2)+'\n')
    class Tagged(cls):
        def infer(self,request):
            if request.get('reset'):
                request=dict(request,_benchmark=dict(case=case,arm=arm))
            return super().infer(request)
    return Tagged,benchmark
original._install_transport=install
init=original._init_env
def initial(env,state):
    result=init(env,state)
    target=getattr(env,'env',env)
    physical=np.asarray(target.sim.get_state().flatten()).copy()
    payload=dict(physical=physical,**result[0])
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'INITIAL-ACTUAL.npz',**payload)
    ref=P/'results'/suite/'initial-references'/(case+'.npz');ref.parent.mkdir(parents=True,exist_ok=True)
    if ref.exists():
        with np.load(ref) as old:
            checks={k:k in old and old[k].dtype==v.dtype and old[k].shape==v.shape and np.array_equal(old[k],v) for k,v in payload.items()}
        scope='Same5090 first arm of same case; not old A100 scene certification'
    else:
        np.savez_compressed(ref,**payload);checks={k:True for k in payload};scope='First5090 arm establishes reference'
    (out/'INITIAL-PAIR-AUDIT.json').write_text(json.dumps(dict(checks=checks,exact=all(checks.values()),scope=scope),indent=2)+'\n')
    if not all(checks.values()):raise RuntimeError('New5090 paired initial scene differs before reset')
    return result
original._init_env=initial
benchmark='libero_10' if suite=='long' else 'libero_'+suite
plan=json.loads((P/'bundle/manifests'/f'{suite}-original-plan.json').read_text())
checkpoint=plan.get('suite_info',{}).get('checkpoint_id',plan.get('target_checkpoint_id'))
os.environ.update(LINGBOT_SEED=str(plan['evaluation_seed']),LINGBOT_VA_LIBERO_SUITE=benchmark,LINGBOT_VA_LIBERO_CHECKPOINT=str(P/'assets'/suite),LINGBOT_VA_LIBERO_CHECKPOINT_ID=checkpoint,LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=str(P/'registry.json'))
import torch,random
random.seed(plan['evaluation_seed']);np.random.seed(plan['evaluation_seed']);torch.manual_seed(plan['evaluation_seed'])
sys.argv=[str(script),'--port','19601','--benchmarks',benchmark,'--task-indices',str(task),'--episodes-per-task','1','--episode-start','49','--condition',arm,'--out-dir',str(out),'--worktree',str(P/'bundle/project/投机采样/wam-p0-worktree'),'--trace-mode','off','--latency-benchmark','--no-resume','--fail-fast']
raise SystemExit(original.main())
