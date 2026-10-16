"""Actual Torch import/predictor/640-row objective smoke, CPU11 only."""
import json
import os
from pathlib import Path
import sys
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from paths import ROOT,BASE,OWN
from candidate_predictor import CandidatePredictor
from prepare import CANDIDATE,CANDIDATE_SHA,CURRENT

def main():
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or os.sched_getaffinity(0)!={8}:raise ValueError('CPU8 only')
    # Real imports used by the model entry, with no construction/load of target.
    import fixed_worker,server_entry,runtime,measured_online,buffers,state_check,scalar,online
    import production_fixture
    from prefix_kernel import load_native,NATIVE_SHA
    native=load_native()
    generic=Path(sys.modules['adapter'].__file__).resolve()
    if generic!=BASE/'adapter.py':raise ValueError('generic adapter import collision')
    old=torch.load(CURRENT,map_location='cpu',weights_only=False)
    rng=torch.get_rng_state().clone();predictor=CandidatePredictor(CANDIDATE,CANDIDATE_SHA)
    if not torch.equal(rng,torch.get_rng_state()):raise ValueError('predictor construction changed CPU RNG')
    for row in old['train_rows'][:16]:
        state=row['state'].bfloat16();prev=row['previous'].bfloat16();before=(state.clone(),prev.clone());rng=torch.get_rng_state().clone()
        got=predictor.predict(state,prev,row['sigma'],row['next_sigma'],row['timestep'])
        gold=predictor.predictor.predict(state,prev,row['sigma'],row['next_sigma'],row['timestep'])
        if (got.shape!=(2,112) or got.dtype!=torch.bfloat16 or got.device.type!='cpu'
            or got.requires_grad or not torch.equal(got,gold) or not torch.equal(state,before[0])
            or not torch.equal(prev,before[1]) or not torch.equal(rng,torch.get_rng_state())):
            raise ValueError('candidate CPU decode/input/RNG contract failed')
    # train_cpu sets interop itself: execute only definitions via AST to avoid
    # modifying process threading a second time.
    import ast
    text=(OWN/'train_cpu.py').read_text();tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('aux_data','objective')]
    root=(OWN.parent/'draft_root_retention_v1/run.py').read_text()
    fn=next(n for n in ast.parse(root).body if isinstance(n,ast.FunctionDef) and n.name=='objective')
    ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<root-objective>','exec'),ns)
    ns['current_objective']=ns['objective'];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<new-objective>','exec'),ns)
    rows=old['train_rows'];d=__import__('importlib.util',fromlist=['x'])
    from candidate_predictor import MODEL_SOURCE
    arch=sys.modules['onpolicy_frozen_architecture_models']
    def data(ix):
        rs=[rows[i] for i in ix]
        return dict(state=torch.stack([r['state'] for r in rs]),previous=torch.stack([r['previous'] for r in rs]),
            scalars=torch.tensor([[r['sigma'],r['next_sigma'],r['timestep']/1000] for r in rs]),target=torch.stack([r['target'] for r in rs]))
    ds=data(range(512));ax=data(range(128));ds.update(weight=old['weights'][:512],retain_mask=old['retain_mask'][:512],old_bf16=old['old_train'][:512])
    merged={k:torch.cat((ds[k],ax[k]),0) for k in ('state','previous','scalars')};checks={}
    for name,count in (('transformer64',53070),('mlp96',52928)):
        torch.manual_seed(20260922);m=arch.Predictor(name,old['normalization']);seen=[]
        handle=m.model.register_forward_hook(lambda module,args,out:seen.append(tuple(args[0].shape)))
        pred=m.decode_batch(merged);value,base,extra,kd=ns['objective'](pred,ds,ax['target'],m.residual_std,float(old['retain_mask'].mean()))
        gold,_,gold_kd=ns['current_objective'](pred[:512],ds,m.residual_std,'C_WEIGHT_PLUS_ROOT_RETAIN',float(old['retain_mask'].mean()))
        if seen!=[(640,227)] or not torch.equal(base,gold) or not torch.equal(kd,gold_kd):raise ValueError('current exposure or merged forward changed')
        value.backward();handle.remove()
        if sum(x.numel() for x in m.model.parameters())!=count or not bool(torch.isfinite(value)) or not all(x.grad is None or bool(torch.isfinite(x.grad).all()) for x in m.model.parameters()):raise ValueError('parameter/gradient check failed')
        checks[name]=dict(parameters=count,single_forward_shape=seen,current_C_exact=True,current_KD_exact=True,finite_gradient=True)
    if torch.cuda.is_initialized():raise ValueError('CUDA initialized')
    print(json.dumps(dict(status='PASS_REAL_TORCH_CPU_IMPORT_AND_OBJECTIVE',torch_version=torch.__version__,generic_adapter=str(generic),
        candidate_checkpoint_sha256=CANDIDATE_SHA,original_CPU_predict_exact=True,input_RNG_unchanged=True,
        model_remains_CPU=True,device_return_GPU_pending=True,models=checks,
        shared_KV_native_path=str(Path(native.__file__).resolve()),shared_KV_native_sha256=NATIVE_SHA,
        CUDA_initialized=False,model_loaded=False)))

if __name__=='__main__':main()
