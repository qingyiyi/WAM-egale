"""Fixed primary TF-cont CPU preparation/training. No collection/GPU at import."""
from __future__ import annotations
import argparse,ast,json,os,signal,statistics,sys,time
from pathlib import Path
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise RuntimeError('CPU-only: CUDA_VISIBLE_DEVICES must be empty')
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
OWN=Path(__file__).resolve().parent;EXP=OWN.parent;ROOT=OWN.parents[2]
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from tf_usable_train_v1.adapter import (SCHEMA,FINAL_STATUS,SEED,PARENT_SHA,load_parent,load_checkpoint,model_source,sha,require)
PARENT_ROOT=ROOT/'投机采样/results/runtime/p0-target-trace-draft-finetune-v1-20261015-r1'
MATH_SOURCE=EXP/'target_trace_draft_finetune_v1/train.py'
TAU=.007839680526450805

def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temporary=p.with_name(p.name+'.tmp')
    with temporary.open('w') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
    os.replace(temporary,p)

def math_functions():
    # Compile only the unchanged pure numerical/sampling functions; never import
    # the old train module's bare `adapter` namespace or run its old main.
    wanted=('draw','decode','loss_value','predict_all','errors','metrics','same_query')
    tree=ast.parse(MATH_SOURCE.read_text());nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in wanted]
    require({x.name for x in nodes}==set(wanted),'frozen numerical function inventory differs')
    ns=dict(torch=torch,statistics=statistics,require=require,TAU=TAU)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(MATH_SOURCE)+':unchanged-functions','exec'),ns)
    return {k:ns[k] for k in wanted}

def data(plan):
    require(read(plan['data_audit'])['status']=='PASS_COMPLETE_NUMERIC_AUDIT','historical target audit incomplete')
    require(read(plan['same_query_audit'])['status']=='PASS_COMPLETE_SAME_QUERY_AUX_NUMERIC_AUDIT','old same-query audit incomplete')
    pack=torch.load(plan['data'],map_location='cpu',weights_only=False);columns=pack['columns']
    keys=('state','previous','target','scalars','delta','task','episode')
    fit={k:columns[k][pack['views']['fit174']].contiguous() for k in keys}
    cal={k:columns[k][pack['views']['cal20']].contiguous() for k in keys}
    sq=math_functions()['same_query'](torch.load(plan['same_query_data'],map_location='cpu',weights_only=False))
    require(len(fit['state'])==159691 and len(cal['state'])==20237 and len(sq['state'])==14997,'original recipe cardinalities differ')
    require(not set(fit['episode'].tolist())&set(cal['episode'].tolist()) and not set(sq['episode'].tolist())&set(cal['episode'].tolist()),'original FIT/CAL overlap')
    return fit,cal,sq

def prepare(job):
    require(not job.exists(),'new isolated continuation job required')
    base=read(PARENT_ROOT/'PLAN.json');final=read(PARENT_ROOT/'FINAL-RESULT.json')
    require(final['status']=='PASS_SIX_TWO_STAGE_FINETUNING_RUNS' and final['complete_runs']==6,'complete parent six runs required')
    require(not (PARENT_ROOT/'FAILED.json').exists(),'parent job failed')
    parent=PARENT_ROOT/f'transformer64-seed{SEED}'/'checkpoint.pt'
    require(sha(parent)==PARENT_SHA,'approved child23 parent changed')
    record=read(parent.parent/'RESULT.json')
    require(record['seed']==SEED and record['total_updates']==2500 and record['checkpoint_sha256']==PARENT_SHA,'parent record differs')
    provenance=[OWN/'adapter.py',OWN/'train.py',OWN/'gpu_actual_cal.py',MATH_SOURCE,
        EXP/'target_trace_draft_finetune_v1/adapter.py',EXP/'target_trace_draft_train_v1/adapter.py',EXP/'draft_architecture_v1/models.py']
    plan=dict(schema='tf-usable-primary-continuation-plan/v1',status='PREPARED_NOT_TRAINED',condition='TF_CONT',architecture='transformer64',seed=SEED,
        data=base['data'],data_audit=base['data_audit'],same_query_data=base['same_query_data'],same_query_audit=base['same_query_audit'],
        parent_checkpoint=str(parent),parent_checkpoint_sha256=PARENT_SHA,parent_training_plan_sha256=sha(PARENT_ROOT/'PLAN.json'),
        initial_total_updates=2500,continuation_updates=500,total_updates=3000,batch=512,target_half=256,same_query_half=256,
        learning_rate=3e-5,weight_decay=.01,optimizer='new AdamW, no carried moments',normalization_refit=False,
        fit_rows=159691,fit_episodes=174,cal_rows=20237,cal_episodes=20,same_query_rows=14997,
        threshold=TAU,old_recipe_unchanged=True,training_source='same_stage2_historical_target_and_original_same_query',
        draw_rule='same frozen stage2 target/same_query balanced task->episode->row draws for primary seed23; identical old500 schedule',
        selection='fixed final update500, one primary seed23; no CAL-based checkpoint/seed search',
        cpu_threads=2,maximum_wall_seconds=900,output_limit_bytes=128*1024*1024,no_GPU=True,new_FIT_CAL_HOLD=True,
        source_sha256={str(p):sha(p) for p in provenance},parent_data_source_proof_sha256=sha(PARENT_ROOT/'DATA-SOURCE-PROOF.json'),
        data_stat={str(Path(base[k])):dict(bytes=Path(base[k]).stat().st_size,mtime_ns=Path(base[k]).stat().st_mtime_ns)
                   for k in ('data','same_query_data')},
        data_audit_sha256={str(Path(base[k])):sha(base[k]) for k in ('data_audit','same_query_audit')},
        no_new_target_forward=True,no_new_LIBERO_episode=True,deployed=False)
    # Preparing validates real source/checkpoint interfaces but does not load data,
    # run a model forward/backward, allocate a CUDA context, or train.
    parent_predictor=load_parent(parent,architecture='transformer64',device='cpu',expected_channels=range(7),expected_sha256=PARENT_SHA)
    require(sum(p.numel() for p in parent_predictor.model.parameters())==53070 and not torch.cuda.is_initialized(),'parent CPU interface differs')
    job.mkdir(parents=True,exist_ok=False);write(job/'PLAN.json',plan)
    write(job/'PREPARED.json',dict(status='PASS_CPU_PREPARED_NO_TRAINING',parent_checkpoint_sha256=PARENT_SHA,parameters=53070,CUDA_initialized=False,source_files=len(provenance)))
    print(json.dumps(dict(status='PASS_CPU_PREPARED_NO_TRAINING',job=str(job),plan_sha256=sha(job/'PLAN.json'))))

def same_cpu_decode_reference(predictor,cal,outputs):
    source=model_source()
    with torch.random.fork_rng(devices=[]):ref=source.Predictor('transformer64',{k:getattr(predictor,k) for k in ('feature_mean','feature_std','residual_mean','residual_std')})
    ref.model.load_state_dict(predictor.model.state_dict(),strict=True);ref.model.eval();values=torch.empty_like(outputs)
    with torch.inference_mode():
        for i in range(len(values)):
            state=cal['state'][i].reshape(1,112).bfloat16().float();prev=cal['previous'][i].reshape(1,112).bfloat16().float()
            sigma,nxt,time_norm=cal['scalars'][i].tolist();timestep=time_norm*1000.
            d=dict(state=state,previous=prev,scalars=torch.tensor([[sigma,nxt,timestep/1000.]],dtype=torch.float32))
            values[i]=ref.decode_batch(d).reshape(2,112).bfloat16().float()
    require(torch.equal(values,outputs) and bool(torch.isfinite(values).all()),'complete same-CPU original decode_batch parity differs')
    return True

def train(job):
    require((job/'PLAN.json').exists() and not (job/'STARTED.json').exists(),'fresh prepared job required')
    plan=read(job/'PLAN.json');require(plan['schema']=='tf-usable-primary-continuation-plan/v1' and plan['seed']==SEED
        and plan['condition']=='TF_CONT' and plan['continuation_updates']==500 and plan['total_updates']==3000
        and plan['target_half']==plan['same_query_half']==256 and plan['batch']==512
        and plan['learning_rate']==3e-5 and plan['weight_decay']==.01,'fixed continuation plan differs')
    require(not (job/'FAILED.json').exists() and not (job/'STOP_REQUESTED').exists(),'failed/stopped prepared job')
    for path,h in plan['source_sha256'].items():require(sha(path)==h,'source drift: '+path)
    for path,expected in plan['data_stat'].items():
        st=Path(path).stat();require(dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)==expected,'frozen original data stat changed')
    for path,h in plan['data_audit_sha256'].items():require(sha(path)==h,'original data audit changed')
    require(sha(PARENT_ROOT/'DATA-SOURCE-PROOF.json')==plan['parent_data_source_proof_sha256'],'parent data proof changed')
    require(sha(PARENT_ROOT/'PLAN.json')==plan['parent_training_plan_sha256'],'parent recipe changed')
    start=time.monotonic();write(job/'STARTED.json',dict(status='RUNNING_CPU_ONLY',pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),CUDA_initialized=torch.cuda.is_initialized()))
    active_updates=0
    try:
        require(torch.get_num_threads()==2 and len(os.sched_getaffinity(0))<=2 and not torch.cuda.is_initialized(),'CPU2/zeroCUDA required')
        signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('TF_CONT full CPU wall cap')));signal.alarm(900)
        funcs=math_functions();fit,cal,sq=data(plan)
        old_draws=torch.load(PARENT_ROOT/'draws.pt',map_location='cpu',weights_only=True)[str(SEED)]
        draws=dict(target=funcs['draw'](fit,SEED+700000),same_query=funcs['draw'](sq,SEED+800000))
        require(all(torch.equal(draws[k],old_draws[k]) and draws[k].shape==(500,256) for k in draws),'original500 draw schedule changed')
        torch.save(draws,job/'draws.pt')
        predictor=load_parent(plan['parent_checkpoint'],architecture='transformer64',device='cpu',expected_channels=range(7),expected_sha256=PARENT_SHA)
        norm={k:getattr(predictor,k).clone() for k in ('feature_mean','feature_std','residual_mean','residual_std')}
        cpu_rng=torch.get_rng_state().clone()
        before=torch.load(Path(plan['parent_checkpoint']).parent/'cal_predictions.pt',map_location='cpu',weights_only=True)
        require(before.shape==(20237,2,112) and bool(torch.isfinite(before).all()),'original complete CAL prediction shape/finite differs')
        predictor.model.train();optimizer=torch.optim.AdamW(predictor.model.parameters(),lr=3e-5,weight_decay=.01);logs=[]
        for update,(ti,si) in enumerate(zip(draws['target'],draws['same_query']),1):
            require(not (job/'STOP_REQUESTED').exists(),'STOP_REQUESTED')
            batch={k:torch.cat((fit[k][ti],sq[k][si]),0) for k in ('state','previous','target','scalars')}
            optimizer.zero_grad(set_to_none=True);loss=funcs['loss_value'](funcs['decode'](predictor,batch),batch,predictor.residual_std)
            require(bool(torch.isfinite(loss)),'nonfinite continuation loss');loss.backward()
            require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in predictor.model.parameters()),'nonfinite continuation gradient')
            optimizer.step();active_updates=update
            if update==1 or update%50==0:
                logs.append(dict(update=update,loss=float(loss.detach()),seconds=time.monotonic()-start));write(job/'status.json',dict(status='TRAINING_TF_CONT',update=update,seconds=time.monotonic()-start))
        predictor.model.eval();require(all(torch.equal(norm[k],getattr(predictor,k)) for k in norm),'parent normalization changed')
        folder=job/f'TF_CONT-seed{SEED}';folder.mkdir(exist_ok=False);checkpoint=folder/'checkpoint.pt';source=model_source()
        metadata={k:plan[k] for k in ('condition','seed','initial_total_updates','continuation_updates','total_updates','batch','target_half','same_query_half','learning_rate','weight_decay','normalization_refit','training_source','parent_checkpoint','parent_checkpoint_sha256')}
        metadata.update(optimizer_state_reset=True,deployed=False,plan=str(job/'PLAN.json'),plan_sha256=sha(job/'PLAN.json'))
        torch.save(dict(schema=SCHEMA,architecture='transformer64',specification=source.SPECS['transformer64'],model_state_dict=predictor.model.state_dict(),normalization=norm,metadata=metadata),checkpoint)
        loaded=load_checkpoint(checkpoint,architecture='transformer64',device='cpu',expected_channels=range(7))
        after=funcs['predict_all'](loaded,cal)
        require(torch.equal(after,funcs['predict_all'](predictor,cal)),'all20237 new loader/training predictor differs')
        same_cpu_decode_reference(loaded,cal,after)
        require(torch.equal(cpu_rng,torch.get_rng_state()) and not torch.cuda.is_initialized(),'CPU RNG/CUDA changed')
        torch.save(after,folder/'cal_predictions.pt')
        record=dict(status='PASS_TF_CONT_CELL',architecture='transformer64',condition='TF_CONT',seed=SEED,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),parent_checkpoint=plan['parent_checkpoint'],parent_checkpoint_sha256=PARENT_SHA,total_updates=3000,continuation_updates=500,
            before_cal=funcs['metrics'](funcs['errors'](before,cal),cal),after_cal=funcs['metrics'](funcs['errors'](after,cal),cal,funcs['errors'](before,cal)),
            full_CAL_new_adapter_bitwise_exact=True,same_CPU_decode_batch_bitwise_exact=True,normalization_exact_parent=True,CPU_rng_unchanged=True,CUDA_initialized=False,loss_log=logs,robot_quality_validated=False)
        write(folder/'RESULT.json',record)
        require(sum(f.stat().st_size for f in job.rglob('*') if f.is_file())<128*1024*1024,'continuation output cap')
        write(job/'selected.json',dict(schema='tf-usable-selection/v1',selection_role='fixed_final_checkpoint_needs_closed_loop_DEV',architectures={'transformer64':dict(condition='TF_CONT',seed=SEED,checkpoint=str(checkpoint),checkpoint_sha256=record['checkpoint_sha256'])},robot_quality_validated=False,deployed=False))
        final=dict(status=FINAL_STATUS,complete_runs=1,condition='TF_CONT',seed=SEED,total_updates=3000,full_CAL_new_adapter_bitwise_exact=True,same_CPU_decode_batch_bitwise_exact=True,normalization_exact_parent=True,elapsed_seconds=time.monotonic()-start,target_forwards=0,LIBERO_episodes=0,CUDA_initialized=False,robot_quality_validated=False,deployed=False)
        write(job/'FINAL-RESULT.json',final);write(job/'status.json',final);signal.alarm(0);print(json.dumps(final),flush=True)
    except BaseException as exc:
        write(job/'FAILED.json',dict(status='FAILED_TF_CONT',error=repr(exc),updates_completed=active_updates,seconds=time.monotonic()-start,CUDA_initialized=torch.cuda.is_initialized()));signal.alarm(0);raise

if __name__=='__main__':
    p=argparse.ArgumentParser();mode=p.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--train',action='store_true');p.add_argument('--job',type=Path,required=True);a=p.parse_args();job=a.job.resolve();prepare(job) if a.prepare else train(job)
