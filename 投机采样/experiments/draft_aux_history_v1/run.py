"""Nine CPU historical-auxiliary runs preserving full current C/KD exposure."""
import argparse
import ast
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

if os.getenv('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU training only')
import torch
torch.set_num_threads(min(4,len(os.sched_getaffinity(0))));torch.set_num_interop_threads(1)
import hashlib
import importlib.util
from datetime import datetime
from zoneinfo import ZoneInfo
from collections import Counter
OWN=Path(__file__).resolve().parent
EXP=OWN.parent
BASEJOB=Path('/var/tmp/cxz-wam-quality-20261014/p0-draft-root-retention-v1-20261014')
HISTJOB=Path('/var/tmp/cxz-wam-quality-20261014/p0-draft-data-rebuild-v1-20261014')
HISTORY=HISTJOB/'historical_long_train_columnar_v2.pt'
HISTORY_SHA='a54cbf16dc97df8a06150cb93f5e02c0ee162b7bb88aa325c423cf90d2280654'
HISTMANIFEST_SHA='b6960595e8d7baa3c8628df43efb40b5deae7c4316091a67a763cd789490a9d8'
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def stamp():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def write(path,value,exclusive=True,durable=False):
    with Path(path).open('x' if exclusive else 'w') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush()
        if durable:os.fsync(f.fileno())
_saved_path=list(sys.path)
c=load('aux_history_root_contracts',EXP/'draft_root_retention_v1/contracts.py')
sys.path[:]=_saved_path
_src=(EXP/'draft_root_retention_v1/run.py').read_text()
_node=next(n for n in ast.parse(_src).body if isinstance(n,ast.FunctionDef) and n.name=='objective')
_ns={'torch':torch};exec(compile(ast.Module(body=[_node],type_ignores=[]),str(EXP/'draft_root_retention_v1/run.py'),'exec'),_ns)
current_objective=_ns['objective']
CONDITIONS=('TF_A_CURRENT_AUX','TF_B_HISTORICAL_AUX','MLP96_B_HISTORICAL_AUX')
SEEDS=(20260922,20260923,20260924)
CURRENT=Path('/var/tmp/cxz-wam-quality-20261014/p0-draft-root-retention-v1-20261014/dataset.pt')
CURRENT_SHA='51c2947fa9af27d3b65beafd03a4c6a4d542907b44819f82d50b3a95d24ee044'

def model_sha(m):return c.d.canonical_sha({k:c.tensor_sha(v) for k,v in m.state_dict().items()})

def aux_data(pred,target,std):
    mse=(pred-target[:,None]).square().mean(2)
    gap=((pred[:,0]-pred[:,1])/std).square().mean(1)
    return (mse.min(1).values+.1*mse.mean(1)+.01*torch.relu(.05-gap)).mean()

def objective(pred,batch,aux_target,std,mask_mean):
    if pred.shape!=(640,2,112):raise ValueError('one merged640 forward required')
    old,_,kd=current_objective(pred[:512],batch,std,'C_WEIGHT_PLUS_ROOT_RETAIN',mask_mean)
    extra=aux_data(pred[512:],aux_target,std)
    return old+.25*extra,old,extra,kd

def prepare(job):
    start=time.monotonic()
    if not os.sched_getaffinity(0).issubset({8,9,10,11}):raise ValueError('CPU8-11 only')
    def stop(*_):raise TimeoutError('CPU prepare budget exhausted')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,1800)
    baseplan=json.loads((BASEJOB/'plan.json').read_text())
    if digest(CURRENT)!=CURRENT_SHA or digest(HISTORY)!=HISTORY_SHA or digest(HISTJOB/'MERGE-READY-MANIFEST.json')!=HISTMANIFEST_SHA:
        raise ValueError('certified input identity changed')
    sources={p:h for p,h in baseplan['source_sha256'].items() if p.endswith('.py')}
    for p,h in sources.items():
        if digest(p)!=h:raise ValueError('original computation source changed: '+p)
    sources[str(OWN/'run.py')]=digest(OWN/'run.py')
    for p in (BASEJOB/'plan.json',BASEJOB/'dataset_manifest.json',BASEJOB/'CPU-OPERATOR-AUDIT.json',BASEJOB/'BASELINE.json',HISTJOB/'MERGE-READY-MANIFEST.json',HISTJOB/'MERGE-READY-RESULT.json',HISTJOB/'INDEPENDENT-VERIFICATION.json'):
        sources[str(p)]=digest(p)
    if json.loads((HISTJOB/'MERGE-READY-RESULT.json').read_text())['status']!='PASS_TRAIN_ONLY_OVERLAP_AND_MERGE_READY':raise ValueError('historical audit missing')
    old=torch.load(CURRENT,map_location='cpu',weights_only=False)
    hist=torch.load(HISTORY,map_location='cpu',weights_only=False)
    if len(old['train_rows'])!=3906 or len(old['validation_rows'])!=1890:raise ValueError('old split changed')
    if {(r['task_index'],r['episode_index']) for r in old['train_rows']}!={(t,i) for t in (0,1) for i in (0,1)}:raise ValueError('TRAIN split changed')
    if {(r['task_index'],r['episode_index']) for r in old['validation_rows']}!={(0,8),(1,8)}:raise ValueError('VAL split changed')
    if hist['schema']!='historical-long-columnar-merge-ready/v1' or hist['manifest_sha256']!=HISTMANIFEST_SHA:raise ValueError('historical schema changed')
    historical=hist['columns'];metadata=hist['metadata']
    if len(metadata)!=4802 or not hist['eligibility']['eligible_historical_supervision'] or hist['eligibility']['eligible_current_CFG14_teacher_oracle']:raise ValueError('history not eligible as historical supervision')
    for k in ('state','previous','target','scalars'):
        if historical[k].shape!=(4802,3 if k=='scalars' else 112) or historical[k].dtype!=torch.float32 or not bool(torch.isfinite(historical[k]).all()):raise ValueError('invalid historical column')
    if any(r['node'] is not None or r['is_speculative_root'] is not None or r['episode_index']!=0 or r['state_sha256']!=r['target_query_state_sha256'] for r in metadata):raise ValueError('historical provenance mismatch')
    if Counter(r['task_index'] for r in metadata)!=Counter({0:833,2:833,4:784,5:588,6:735,9:1029}):raise ValueError('historical tasks changed')
    if set(hist['views']['scratch_historical_all'].tolist())!=set(range(4802)):raise ValueError('scratch view must contain all historical TRAIN rows')
    old_data=c.d.prepared(old['train_rows'])
    fresh=c.normalize(old_data)
    normsha={k:c.tensor_sha(v) for k,v in old['normalization'].items()}
    if normsha!=baseplan['normalizer_sha256'] or {k:c.tensor_sha(v) for k,v in fresh.items()}!=normsha:raise ValueError('train-only normalizer differs')
    design_node=next(n for n in ast.parse(_src).body if isinstance(n,ast.FunctionDef) and n.name=='training_design')
    design_ns={'torch':torch,'d':c.d}
    exec(compile(ast.Module(body=[design_node],type_ignores=[]),str(EXP/'draft_root_retention_v1/run.py'),'exec'),design_ns)
    weights,mask,design=design_ns['training_design'](old['train_rows'],old_data,old['old_train'])
    if not torch.equal(weights,old['weights']) or not torch.equal(mask,old['retain_mask']) or design!=baseplan['design']:raise ValueError('train-only weight/KD design changed')
    job.mkdir(exist_ok=False,parents=True);out=job/'training';out.mkdir()
    draws={};initial={};coverage={}
    for seed in SEEDS:
        ds={'current':torch.randint(3906,(1000,512),generator=torch.Generator().manual_seed(seed)),
            'aux_current':torch.randint(3906,(1000,128),generator=torch.Generator().manual_seed(seed+200000)),
            'aux_historical':torch.randint(4802,(1000,128),generator=torch.Generator().manual_seed(seed+300000))}
        if c.tensor_sha(ds['current'])!=baseplan['sample_indices_sha256'][str(seed)]:raise ValueError('paired current sampler changed')
        draws[str(seed)]=ds;coverage[str(seed)]={k:dict(draws=v.numel(),unique_rows=int(v.unique().numel())) for k,v in ds.items()}
        for arch in ('transformer64','mlp96'):
            torch.manual_seed(seed);model=c.models.Predictor(arch,old['normalization']);initial[f'{arch}:{seed}']=model_sha(model.model)
    current={k:v[:512] for k,v in old_data.items()};current.update(weight=old['weights'][:512],retain_mask=old['retain_mask'][:512],old_bf16=old['old_train'][:512])
    checks={};fraction=float(old['retain_mask'].mean())
    for arch in ('transformer64','mlp96'):
        for name,aux in (('current',old_data),('historical',historical)):
            torch.manual_seed(SEEDS[0]);m=c.models.Predictor(arch,old['normalization'])
            merged={k:torch.cat((old_data[k][:512],aux[k][:128]),0) for k in ('state','previous','scalars')}
            pred=m.decode_batch(merged);value,base,extra,kd=objective(pred,current,aux['target'][:128],m.residual_std,fraction)
            reference=current_objective(pred[:512],current,m.residual_std,'C_WEIGHT_PLUS_ROOT_RETAIN',fraction)
            if not torch.equal(base,reference[0]) or not torch.equal(kd,reference[2]) or not torch.equal(value,reference[0]+.25*aux_data(pred[512:],aux['target'][:128],m.residual_std)):raise ValueError('640 current/KD exposure differs')
            value.backward()
            checks[arch+'_'+name+'_finite_backward']=bool(torch.isfinite(value)) and all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in m.model.parameters())
        checks[arch+'_parameter_count']=sum(v.numel() for v in m.model.parameters())
    if not all(v for k,v in checks.items() if k.endswith('finite_backward')) or checks['transformer64_parameter_count']!=53070 or checks['mlp96_parameter_count']!=52928:raise ValueError('CPU objective/parameter checks failed')
    checks.update(current_C_and_KD_exact=True,normalization_train_only_exact=True,weights_mask_exact=True,current_draws_original_exact=True,CUDA_initialized=torch.cuda.is_initialized(),torch_version=torch.__version__)
    if checks['CUDA_initialized']:raise ValueError('CUDA initialization forbidden')
    pack=dict(current=old,historical_columns=historical,historical_metadata=metadata,historical_eligibility=hist['eligibility'])
    torch.save(pack,out/'dataset.pt');torch.save(draws,out/'draws.pt')
    run_order=[dict(seed=s,condition=x,architecture='mlp96' if x.startswith('MLP96') else 'transformer64') for s in SEEDS for x in CONDITIONS]
    tensors={**{f'normalization/{k}':c.tensor_sha(v) for k,v in old['normalization'].items()},**{k:c.tensor_sha(old[k]) for k in ('weights','retain_mask','old_train','old_val','val_target_next','old_val_next')}}
    training=dict(status='PREPARED_NINE_HISTORICAL_AUX_RUNS_NO_TRAINING',at=stamp(),conditions=CONDITIONS,seeds=SEEDS,run_order=run_order,
        current_rows=3906,historical_rows=4802,validation_rows=1890,old_current_dataset_sha256=CURRENT_SHA,historical_dataset_sha256=HISTORY_SHA,
        historical_manifest_sha256=HISTMANIFEST_SHA,historical_eligibility=hist['eligibility'],historical_tasks=dict(Counter(r['task_index'] for r in metadata)),
        historical_semantics='Long historical target-only TRAIN init0; speculative node unknown; no historical root label, weight or KD; not current CFG14 oracle or newly collected candidate-induced data',
        heldout_policy='old VAL init8 used only adaptive development evaluation; old test init9-16 and historical heldouts not read; no new onpolicy episode available',
        updates=1000,current_batch=512,aux_batch=128,merged_forward_batch=640,lr=1e-4,weight_decay=.01,
        loss='current_C + .25 aux_DATA',aux_KD=False,current_weights_KD_and_normalization_unchanged=True,retain_fraction=fraction,
        design=design,current_tensor_sha256=tensors,
        init_rule='from scratch all default Torch module initializers; TF A/B same initialized bytes; architectures only share seed and default initialization strategy',
        initial_sha256=initial,draw_sha256={s:{k:c.tensor_sha(v) for k,v in ds.items()} for s,ds in draws.items()},draw_coverage=coverage,
        sampler='current seed; aux_current seed+200000; aux_historical seed+300000; paired current512 all conditions, paired auxiliary128 TF/MLP historical conditions',
        draws_sha256=digest(out/'draws.pt'),dataset_sha256=digest(out/'dataset.pt'),source_sha256=sources,
        CPU_total_limit_seconds=1800,prepare_seconds=time.monotonic()-start,CPU_affinity=[8,9,10,11],torch_threads=4,torch_interop=1,
        no_GPU=True,no_deployment=True,old_validation_is_adaptive_development=True,
        original_entry_gate=['root and each task root pass >= OLD','all and root Euler L2 <= OLD','root OLD-pass losses<=1%'],
        selection='all component median margins>=0 AND >=2/3 seeds pass AND median-root-pass seed itself passes',
        rank=['descending median root pass','ascending median root L2','ascending median all L2','declared condition order'])
    write(out/'CPU-TRAINING-CHECKS.json',checks,exclusive=True)
    write(out/'plan.json',training,exclusive=True,durable=True)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(dict(status=training['status'],plan_sha256=digest(out/'plan.json'),historical_rows=4802,prepare_seconds=training['prepare_seconds'])))

def summary(results):
    configs=[]
    for condition in CONDITIONS:
        rs=[r for r in results if r['condition']==condition]
        if len(rs)!=3:raise ValueError('all three seeds required')
        med=lambda key:statistics.median(r['euler_proxy']['groups']['root'][key] for r in rs)
        margins={k:statistics.median(r['entry_gate']['values'][k] for r in rs) for k in rs[0]['entry_gate']['values']}
        middle=sorted(rs,key=lambda r:(r['euler_proxy']['groups']['root']['pass_count'],r['euler_proxy']['groups']['root']['successor_l2_mean'],r['seed']))[1]
        eligible=all(v>=0 for v in margins.values()) and sum(r['entry_gate']['passed'] for r in rs)>=2 and middle['entry_gate']['passed']
        configs.append(dict(condition=condition,eligible=eligible,median_gate_margins=margins,
            seeds_passing=sum(r['entry_gate']['passed'] for r in rs),checkpoint=middle['checkpoint'],seed=middle['seed'],
            rank=[-med('pass_count'),med('successor_l2_mean'),statistics.median(r['euler_proxy']['groups']['all']['successor_l2_mean'] for r in rs),CONDITIONS.index(condition)]))
    passed=sorted((x for x in configs if x['eligible']),key=lambda x:x['rank'])
    return dict(configurations=configs,selected=passed[0] if passed else None,decision='ONE_CANDIDATE_NEEDS_MATCHED_GPU_QUALITY' if passed else 'KEEP_A_OFF')

def train(job,approved):
    out=job/'training';p=json.loads((out/'plan.json').read_text())
    if approved!=digest(out/'plan.json') or (out/'TRAINING-STARTED.json').exists():raise ValueError('fresh approved exact plan required')
    if not os.sched_getaffinity(0).issubset({8,9,10,11}) or torch.get_num_threads()>4:raise ValueError('CPU8-11 four threads max')
    start=time.monotonic();remaining=1800-p['prepare_seconds']
    if remaining<=0:raise TimeoutError('CPU budget exhausted')
    def stop(*_):raise TimeoutError('CPU1800sec includes prepare/train/eval')
    signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,remaining)
    write(out/'TRAINING-STARTED.json',dict(pid=os.getpid(),at=stamp(),plan_sha256=approved),exclusive=True)
    results=[];active=None
    try:
        for file,h in p['source_sha256'].items():
            if digest(file)!=h:raise ValueError('frozen source changed: '+file)
        for f,k in (('dataset.pt','dataset_sha256'),('draws.pt','draws_sha256')):
            if digest(out/f)!=p[k]:raise ValueError('training artifact changed')
        pack=torch.load(out/'dataset.pt',map_location='cpu',weights_only=False);draws=torch.load(out/'draws.pt',map_location='cpu',weights_only=False)
        cache=pack['current'];val=cache['validation_rows'];data=c.d.prepared(cache['train_rows']);new=pack['historical_columns']
        data.update(weight=cache['weights'],retain_mask=cache['retain_mask'],old_bf16=cache['old_train']);fraction=float(cache['retain_mask'].mean())
        operator=c.CPUTransition()
        for spec in p['run_order']:
            seed,condition,arch=spec['seed'],spec['condition'],spec['architecture'];folder=out/f'{condition}-seed{seed}';folder.mkdir()
            active=dict(**spec,updates_completed=0);torch.manual_seed(seed);m=c.models.Predictor(arch,cache['normalization'])
            if model_sha(m.model)!=p['initial_sha256'][f'{arch}:{seed}']:raise ValueError('initialization changed')
            m.model.train();opt=torch.optim.AdamW(m.model.parameters(),lr=1e-4,weight_decay=.01);logs=[]
            ds=draws[str(seed)];aux_kind='aux_current' if condition=='TF_A_CURRENT_AUX' else 'aux_historical';aux=data if aux_kind=='aux_current' else new
            for step,(ix,jx) in enumerate(zip(ds['current'],ds[aux_kind]),1):
                batch={k:v[ix] for k,v in data.items()};extra={k:v[jx] for k,v in aux.items()}
                merged={k:torch.cat((batch[k],extra[k]),0) for k in ('state','previous','scalars')}
                pred=m.decode_batch(merged);value,base,add,kd=objective(pred,batch,extra['target'],m.residual_std,fraction)
                if not bool(torch.isfinite(value)):raise ValueError('nonfinite loss')
                opt.zero_grad(set_to_none=True);value.backward()
                if not all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in m.model.parameters()):raise ValueError('nonfinite gradient')
                opt.step();active['updates_completed']=step
                if step==1 or step%100==0:logs.append(dict(step=step,loss=float(value.detach()),current=float(base.detach()),aux=float(add.detach()),KD=float(kd.detach())))
            m.model.eval();path=folder/'checkpoint.pt';m.save(path,dict(**spec,updates=1000,plan_sha256=approved,deployment=False))
            pred=c.predictions(m,val);reloaded=c.models.Predictor.load(path)
            if not torch.equal(pred,c.predictions(reloaded,val)):raise ValueError('reload mismatch')
            metrics=c.evaluate(pred,val,cache['old_val']);projected=operator.projected(val,pred)
            actual=c.projected_metrics(projected,cache['val_target_next'],cache['old_val_next'],val)
            torch.save(dict(predicted_directions=pred,cpu_projected_successors=projected),folder/'validation_predictions.pt')
            result=dict(**spec,checkpoint=str(path),checkpoint_sha256=digest(path),euler_proxy=metrics,cpu_projected_successor=actual,
                entry_gate=c.gate(metrics),loss_log=logs,updates=1000,all_validation_reload_exact=True)
            write(folder/'RESULT.json',result,exclusive=True);results.append(result);active=None
            print(json.dumps(dict(run=condition,seed=seed,root=metrics['groups']['root'],gate=result['entry_gate']['passed'])),flush=True)
        for file,h in p['source_sha256'].items():
            if digest(file)!=h:raise ValueError('frozen source changed during training: '+file)
        final=dict(status='PASS_CPU_NINE_HISTORICAL_AUX_RUNS',complete_runs=9,summary=summary(results),
            elapsed_including_prepare=time.monotonic()-start+p['prepare_seconds'],CUDA_initialized=torch.cuda.is_initialized(),no_deployment=True)
        write(out/'FINAL-RESULT.json',final,exclusive=True,durable=True)
    except BaseException as exc:
        write(out/'PARTIAL-RESULT.json',dict(status='INCOMPLETE_NO_SELECTION',completed=len(results),active=active,error=repr(exc)),exclusive=True,durable=True)
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('command',choices=['prepare','train']);ap.add_argument('--job',type=Path,required=True);ap.add_argument('--approved-plan-sha256');a=ap.parse_args()
    if a.command=='prepare':prepare(a.job.resolve())
    else:train(a.job.resolve(),a.approved_plan_sha256)
