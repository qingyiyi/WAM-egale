"""Nine fixed CPU runs after complete candidate-induced TRAIN collection."""
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
from paths import OWN,EXP,load
from experiment import digest,write,stamp,plan
_saved_path=list(sys.path)
c=load('onpolicy_root_contracts',EXP/'draft_root_retention_v1/contracts.py')
sys.path[:]=_saved_path
_src=(EXP/'draft_root_retention_v1/run.py').read_text()
_node=next(n for n in ast.parse(_src).body if isinstance(n,ast.FunctionDef) and n.name=='objective')
_ns={'torch':torch};exec(compile(ast.Module(body=[_node],type_ignores=[]),str(EXP/'draft_root_retention_v1/run.py'),'exec'),_ns)
current_objective=_ns['objective']
CONDITIONS=('TF_A_CURRENT_AUX','TF_B_NEW_AUX','MLP96_B_NEW_AUX')
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

def load_new(job,p):
    rows=[];sources=[];identities=set()
    for item in p['assignments'][0]:
        out=job/'campaigns'/item['arm']/'workers'/item['id']
        records=[json.loads(v) for v in (out/'client/episodes.jsonl').read_text().splitlines() if v.strip()]
        if len(records)!=1:raise ValueError('one complete episode per assignment')
        e=records[0]
        if (e['status']!='completed' or e['truncated_after_max_policy_chunks'] or
            e['task_index']!=item['task'] or e['episode_index']!=17 or not(e['success'] or e['horizon_reached'])):
            raise ValueError('full episode identity/termination invalid')
        root=out/'teacher'
        if list(root.glob('ERROR-*')) or list(root.glob('*.tmp')):raise ValueError('teacher capture errors')
        count={}
        for path in sorted(root.glob('*.pt')):
            x=torch.load(path,map_location='cpu',weights_only=False);ct=x['contract'];prov=ct['provenance']
            if (x['schema_version']!='bonus-prefix-same-query-teacher/v2' or x['extra_teacher_forwards']!=0
                or len(x['rows'])!=3 or {r['node'] for r in x['rows']}!={0,1,2}
                or prov['draft_sha256']!=p['new_checkpoint_sha256'] or prov['plan_sha256']!=digest(job/'plan.json')
                or prov['target_checkpoint_id']!=p['frozen_reference']['target_checkpoint_id']
                or ct['threshold']!=c.d.THRESHOLD or ct['selector']!='prefix_then_cost'
                or ct['mode']!='verifier_k2_b2_bonus' or not ct['bonus_enabled']
                or ct['action_updates_per_chunk']!=50 or ct['used_action_channel_ids']!=list(range(7))
                or ct['cfg_scale']!=1. or ct['cache_mode']!='original'):
                raise ValueError('same-query production teacher contract changed')
            for r in x['rows']:
                if (r['split']!='train' or r['episode_index']!=17 or r['task_index']!=item['task']
                    or r['episode_attempt_id']!=e['episode_attempt_id'] or r['source_state_dtype']!='torch.bfloat16'):
                    raise ValueError('new data split/episode identity mismatch')
                key=(r['episode_attempt_id'],r['rollout_chunk'],r['tree_step_index'],r['node'])
                if key in identities:raise ValueError('duplicate row identity')
                identities.add(key)
                for k in ('state','previous','target'):
                    if r[k].shape!=(112,) or r[k].dtype!=torch.float32 or not bool(torch.isfinite(r[k]).all()):raise ValueError('invalid compact tensor')
                if c.tensor_sha(r['state'])!=r['state_sha256'] or r['state_sha256']!=r['target_query_state_sha256']:
                    raise ValueError('actual query hash mismatch')
                rows.append(r);count[r['rollout_chunk']]=count.get(r['rollout_chunk'],0)+1
            sources.append(dict(path=str(path),sha256=digest(path)))
        markers=[json.loads(x.read_text()) for x in root.glob('CHUNK-*.json')]
        if len(markers)!=e['infer_chunks'] or set(count)!=set(range(e['infer_chunks'])):raise ValueError('missing policy chunk labels')
        for m in markers:
            if not m['complete'] or m['attempted_trees']!=m['saved_trees'] or count[m['context']['rollout_chunk']]!=3*m['saved_trees']:
                raise ValueError('incomplete all-node labels')
    rows.sort(key=lambda r:(r['task_index'],r['rollout_chunk'],r['tree_step_index'],r['node']))
    if not rows:raise ValueError('no candidate-induced TRAIN labels')
    return rows,sources

def prepare(job):
    start=time.monotonic();p=plan(job);terminal=json.loads((job/'GPU-COLLECTION.json').read_text())
    if terminal['status']!='PASS_GPU_FULL_EPISODE_COLLECTION' or terminal['completed_episodes']!=2 or not terminal['owned_roots_exited']:
        raise ValueError('closed complete collection required before preparing training')
    if digest(CURRENT)!=CURRENT_SHA:raise ValueError('current TRAIN/VAL cache changed')
    old=torch.load(CURRENT,map_location='cpu',weights_only=False)
    if len(old['train_rows'])!=3906 or len(old['validation_rows'])!=1890:raise ValueError('old split changed')
    if {r['episode_index'] for r in old['train_rows']}!={0,1} or {r['episode_index'] for r in old['validation_rows']}!={8}:raise ValueError('old episode split changed')
    new,sources=load_new(job,p);transition=c.CPUTransition();audit=transition.audit(new,sources)
    def features(rows):
        d=c.d.prepared(rows)
        return [c.tensor_sha(x) for x in torch.cat((d['state'],d['previous'],d['scalars']),1)]
    new_keys=features(new);train_keys=set(features(old['train_rows']));val_keys=set(features(old['validation_rows']))
    overlap=dict(new_feature_matches_current_train=sum(k in train_keys for k in new_keys),
        new_feature_matches_current_validation=sum(k in val_keys for k in new_keys),rows_filtered=0,
        definition='227 draft-visible FP32 features only; not full image/text/history target query identity',
        validation_labels_used_for_training=False)
    out=job/'training';out.mkdir(exist_ok=False)
    draws={};initial={};new_data=c.d.prepared(new);old_data=c.d.prepared(old['train_rows'])
    for seed in SEEDS:
        draws[str(seed)]={'current':torch.randint(3906,(1000,512),generator=torch.Generator().manual_seed(seed)),
            'aux_current':torch.randint(3906,(1000,128),generator=torch.Generator().manual_seed(seed+200000)),
            'aux_new':torch.randint(len(new),(1000,128),generator=torch.Generator().manual_seed(seed+300000))}
        for arch in ('transformer64','mlp96'):
            torch.manual_seed(seed);model=c.models.Predictor(arch,old['normalization']);initial[f'{arch}:{seed}']=model_sha(model.model)
    current={k:v[:512] for k,v in old_data.items()};current.update(weight=old['weights'][:512],retain_mask=old['retain_mask'][:512],old_bf16=old['old_train'][:512])
    merged={k:torch.cat((old_data[k][:512],old_data[k][:128]),0) for k in ('state','previous','scalars')}
    checks={}
    for arch in ('transformer64','mlp96'):
        torch.manual_seed(SEEDS[0]);m=c.models.Predictor(arch,old['normalization']);pred=m.decode_batch(merged)
        value,_,_,_=objective(pred,current,old_data['target'][:128],m.residual_std,float(old['retain_mask'].mean()))
        value.backward();checks[arch+'_finite_backward']=bool(torch.isfinite(value)) and all(v.grad is None or bool(torch.isfinite(v.grad).all()) for v in m.model.parameters())
        checks[arch+'_parameter_count']=sum(v.numel() for v in m.model.parameters())
    if (not checks['transformer64_finite_backward'] or not checks['mlp96_finite_backward']
        or checks['transformer64_parameter_count']!=53070 or checks['mlp96_parameter_count']!=52928):
        raise ValueError('CPU objective/parameter checks failed')
    pack=dict(current=old,new_rows=new,new_sources=sources);torch.save(pack,out/'dataset.pt');torch.save(draws,out/'draws.pt')
    run_order=[dict(seed=s,condition=x,architecture='mlp96' if x.startswith('MLP96') else 'transformer64') for s in SEEDS for x in CONDITIONS]
    training=dict(status='PREPARED_NINE_RUNS_NO_TRAINING',at=stamp(),collection_plan_sha256=digest(job/'plan.json'),
        collection_terminal_sha256=digest(job/'GPU-COLLECTION.json'),conditions=CONDITIONS,seeds=SEEDS,run_order=run_order,
        current_rows=3906,new_rows=len(new),validation_rows=1890,old_current_dataset_sha256=CURRENT_SHA,
        compact_feature_overlap_diagnostic=overlap,
        updates=1000,current_batch=512,aux_batch=128,merged_forward_batch=640,lr=1e-4,weight_decay=.01,
        loss='current_C + .25 aux_DATA',aux_KD=False,current_weights_KD_and_normalization_unchanged=True,
        init_rule='TF A/B same initialized bytes; architectures only share seed and default initialization strategy',
        initial_sha256=initial,draw_sha256={s:{k:c.tensor_sha(v) for k,v in ds.items()} for s,ds in draws.items()},
        draws_sha256=digest(out/'draws.pt'),dataset_sha256=digest(out/'dataset.pt'),new_sources=sources,
        source_sha256={**p['implementation_sha256'],**p['critical_source_sha256']},
        CPU_total_limit_seconds=1800,prepare_seconds=time.monotonic()-start,CPU_affinity=[8,9,10,11],
        torch_threads=4,torch_interop=1,no_deployment=True,old_validation_is_adaptive_development=True,
        original_entry_gate=['root and each task root pass >= OLD','all and root Euler L2 <= OLD','root OLD-pass losses<=1%'],
        selection='all component median margins>=0 AND >=2/3 seeds pass AND median-root-pass seed itself passes',
        rank=['descending median root pass','ascending median root L2','ascending median all L2','declared condition order'])
    write(out/'CPU-NEW-OPERATOR-AUDIT.json',audit,exclusive=True);write(out/'CPU-TRAINING-CHECKS.json',checks,exclusive=True)
    write(out/'plan.json',training,exclusive=True,durable=True)
    print(json.dumps(dict(status=training['status'],plan_sha256=digest(out/'plan.json'),new_rows=len(new))))

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
        cache=pack['current'];val=cache['validation_rows'];data=c.d.prepared(cache['train_rows']);new=c.d.prepared(pack['new_rows'])
        data.update(weight=cache['weights'],retain_mask=cache['retain_mask'],old_bf16=cache['old_train']);fraction=float(cache['retain_mask'].mean())
        operator=c.CPUTransition()
        for spec in p['run_order']:
            seed,condition,arch=spec['seed'],spec['condition'],spec['architecture'];folder=out/f'{condition}-seed{seed}';folder.mkdir()
            active=dict(**spec,updates_completed=0);torch.manual_seed(seed);m=c.models.Predictor(arch,cache['normalization'])
            if model_sha(m.model)!=p['initial_sha256'][f'{arch}:{seed}']:raise ValueError('initialization changed')
            m.model.train();opt=torch.optim.AdamW(m.model.parameters(),lr=1e-4,weight_decay=.01);logs=[]
            ds=draws[str(seed)];aux_kind='aux_current' if condition=='TF_A_CURRENT_AUX' else 'aux_new';aux=data if aux_kind=='aux_current' else new
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
        final=dict(status='PASS_CPU_NINE_ONPOLICY_RUNS',complete_runs=9,summary=summary(results),
            elapsed_including_prepare=time.monotonic()-start+p['prepare_seconds'],CUDA_initialized=torch.cuda.is_initialized(),no_deployment=True)
        write(out/'FINAL-RESULT.json',final,exclusive=True,durable=True)
    except BaseException as exc:
        write(out/'PARTIAL-RESULT.json',dict(status='INCOMPLETE_NO_SELECTION',completed=len(results),active=active,error=repr(exc)),exclusive=True,durable=True)
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)

if __name__=='__main__':
    raise SystemExit('This copied training entry is disabled: train only after all four wave jobs pass, in a separately frozen training source.')
