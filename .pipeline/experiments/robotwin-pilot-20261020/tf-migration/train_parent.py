"""One fixed final TF64 PARENT2000. No seed/model/checkpoint search."""
from pathlib import Path
import ast,os,json,time,traceback
os.sched_setaffinity(0,{18,19})
import torch
from draft import Transformer,Predictor,SPEC
B=Path(__file__).resolve().parent
torch.set_num_threads(2);torch.set_num_interop_threads(1)
state=dict(status='RUNNING_FIXED_PARENT2000',start=time.time(),updates_completed=0,seed=20260923,
           robot_quality_validated=False,formal_performance_permission=False,loss_log=[])
def save():(B/'TRAIN-RESULT.json').write_text(json.dumps(state,indent=2)+'\n')
try:
    assert json.loads((B/'DATA-AUDIT.json').read_text())['status']=='PASS_TARGET48_COMPLETE_SUPERVISION'
    assert not (B/'parent2000.pt').exists()
    pack=torch.load(B/'target48.pt',map_location='cpu',weights_only=True);fit=pack['data']['FIT'];cal=pack['data']['CAL']
    # Reuse the real original loss and normalizer functions without importing runners.
    tree=ast.parse((B/'original_train_math.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('normalize','data_loss')]
    assert {n.name for n in nodes}=={'normalize','data_loss'}
    ns=dict(torch=torch);exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_train_math.py','exec'),ns)
    norm=ns['normalize'](fit)
    assert all(torch.isfinite(v).all() for v in norm.values())
    torch.manual_seed(20260923);torch.cuda.manual_seed_all(20260923)
    model=Transformer().float().cuda();p=Predictor(model,norm)
    state['parameters']=sum(v.numel() for v in model.parameters())
    # Task-uniform, episode-uniform, then row-uniform: same sampling rule as LIBERO,
    # with six fixed tasks instead of ten. Draw entirely from FIT36.
    g=torch.Generator().manual_seed(20260923);tasks=torch.randint(6,(2000*512,),generator=g);draws=torch.empty_like(tasks)
    for task in range(6):
        slots=(tasks==task).nonzero().flatten();episodes=fit['episode'][fit['task']==task].unique()
        choices=torch.randint(len(episodes),(len(slots),),generator=g)
        for i,episode in enumerate(episodes):
            where=slots[choices==i];pool=((fit['task']==task)&(fit['episode']==episode)).nonzero().flatten()
            draws[where]=pool[torch.randint(len(pool),(len(where),),generator=g)]
    draws=draws.reshape(2000,512).cuda();gpu={k:fit[k].cuda() for k in ['state','previous','target','scalars']}
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    model.train()
    for i,ids in enumerate(draws):
        assert time.time()-state['start']<2400
        batch={k:v[ids] for k,v in gpu.items()};optimizer.zero_grad(set_to_none=True)
        loss=ns['data_loss'](p.decode_batch(batch),batch,p.residual_std)
        assert torch.isfinite(loss)
        loss.backward();optimizer.step();state['updates_completed']=i+1
        if i==0 or (i+1)%100==0:
            state['loss_log'].append(dict(update=i+1,loss=float(loss.detach()),seconds=time.time()-state['start']));save()
    model.eval();assert all(torch.isfinite(v).all() for v in model.parameters())
    metadata=dict(condition='PARENT2000',seed=20260923,updates=2000,batch=512,lr=1e-4,weight_decay=.01,
                  normalization='Target FIT36 only',selection='fixed final update',parameters=state['parameters'],
                  target_revision='8c9dea8abbc5c91cc9e18bc3264b8915083bbe70',representation='16config_channels x 32positions',
                  source_data=str(B/'target48.pt'),source_role='Target visited, not on-policy TF',quality_passed=False)
    torch.save(dict(schema='robotwin-tf64/v1',specification=SPEC,model_state_dict={k:v.cpu() for k,v in model.state_dict().items()},
                    normalization=norm,metadata=metadata,channels=pack['channels']),B/'parent2000.pt')
    # Deploy exactly one query at a time, FP32 draft math -> original BF16 direction.
    # Calibration is the predeclared task/episode-equal inverse empirical CDF q90.
    errors=[];weights=[];episodes=cal['episode'];rows_per={int(e):int((episodes==e).sum()) for e in episodes.unique()}
    device=torch.device('cuda:0');rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state()
    max_decode_difference=0.;n=len(cal['state'])
    with torch.inference_mode():
        for i in range(n):
            s,nsig,nt=cal['scalars'][i].tolist()
            x=cal['state'][i].reshape(16,2,16).bfloat16().to(device)
            previous=cal['previous'][i].reshape_as(x).bfloat16().to(device)
            pred=p.predict(x,previous,s,nsig,nt*1000.).reshape(2,512).float()
            teacher=cal['target'][i].to(device)
            err=(pred-teacher).square().sum(-1).sqrt()*abs(s-nsig)
            errors.append(float(err.min()));weights.append(1/(12*rows_per[int(episodes[i])]))
            if i<16:
                direct=p.decode_batch(dict(state=x.reshape(1,512).float(),previous=previous.reshape(1,512).float(),
                    scalars=torch.tensor([[s,nsig,nt]],device=device))).bfloat16().float().reshape(2,512)
                assert torch.equal(direct,pred)
    assert torch.equal(rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state())
    ordered=sorted(zip(errors,weights));cum=0.;tau=None
    for value,weight in ordered:
        cum+=weight
        if cum>=.9:tau=value;break
    assert tau is not None and tau>=0
    (B/'CALIBRATION.json').write_text(json.dumps(dict(status='FIXED_PARENT_CAL12_ENGINEERING_THRESHOLD',tau=tau,q=.9,
        algorithm='inverse weighted empirical CDF, minimal value crossing .9',weights='task/episode equal, row uniform within episode',
        rows=n,errors_min=min(errors),errors_max=max(errors),one_query_BF16_deployment=True,
        adapter_decode_observed_queries=16,adapter_decode_exact=True,CPU_GPU_model_parity='NOT_TESTED',
        B2_B14_teacher_parity='NOT_TESTED',task_success_guarantee=False,not_LIBERO_tau=True),indent=2))
    torch.save(dict(error=torch.tensor(errors),weight=torch.tensor(weights),episode=episodes),B/'cal12-errors.pt')
    state.update(status='COMPLETE_PARENT2000_AND_CAL12_NOT_ROBOT_CERTIFICATE',tau=tau,CAL_rows=n,metadata=metadata)
except BaseException as e:state.update(status='FAILED_PARENT2000_OR_CAL',error=repr(e),traceback=traceback.format_exc())
finally:state['elapsed_seconds']=time.time()-state['start'];save()
if state['status']!='COMPLETE_PARENT2000_AND_CAL12_NOT_ROBOT_CERTIFICATE':raise SystemExit(1)
