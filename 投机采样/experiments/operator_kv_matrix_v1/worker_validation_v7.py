"""Real canonical B2/video and frozen B14 operator/block comparisons, no hashes."""
from pathlib import Path
import ast
import copy
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
import types

ROOT=Path(__file__).resolve().parents[3]
EXP=ROOT/'投机采样/experiments'
MODEL=ROOT/'投机采样/wam-p0-worktree/wan_va/modules/model.py'
OLD=ROOT/'wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/operator_lab/wam_transformer_tile'
JOB=Path(os.environ['OPERATOR_MATRIX_JOB']).resolve()
PLAN=json.loads((JOB/'PLAN.json').read_text())
START=time.monotonic()


def write(name,data):
    path=JOB/name;tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def run(s,source):
    import torch
    torch.cuda.set_per_process_memory_fraction(.70,0)
    import numpy as np
    import random
    sys.path.insert(0,str(OLD))
    from wam_transformer_tile import cuda_sdpa_standalone,torch_sdpa
    sys.path.insert(0,str(EXP/'shared_prefix_torch_v2'))
    sys.path.insert(0,str(EXP/'wam_candidate_tile_v1'))
    import _shared_torch_v1 as native
    import _wam_candidate_tile as grouped
    torch.set_num_threads(2)
    model=s.transformer
    assert len(model.blocks)==30 and s._cache_mode=='original'
    assert torch.__version__=='2.9.0+cu126' and torch.version.git_version=='0fabc3ba44823f257e70ce397d989c8de5e362c1'
    assert s._p0_action_draft_mode=='off'
    # Canonical reset(None) initializes fields before exact frozen-state restore.
    # It does not encode a prompt or run a target forward; restored RNG wins.
    s.infer(dict(reset=True,prompt=None))
    p=json.loads((JOB/'SOURCE-PLAN.json').read_text())
    result=dict(status='RUNNING',torch_version=torch.__version__,torch_git_version=torch.version.git_version,
        GPU=torch.cuda.get_device_name(),rounds=7,attention_iterations=50,block_iterations=20,graph_calls=32,
        original_atol=.02,original_rtol=.02,points=[],full_query_oracles=[],target_forwards=0,
        episodes=0,formal_robot_performance=False,original_sources_modified=False,
        B14_source='2026-10-16 frozen real A_off verifier candidates; not current TF closed-loop performance',
        ordinary_dense_B14_materialization='original Torch reference layout only; strongest shared-prefix separately measured',
        timing_scope='prepared operator and real static block; no policy/task E2E or effective commit speedup')
    data_points={};control=dict(active=None,current=None,indices={})
    selected={0,8,29}

    def check():
        if time.monotonic()-START>PLAN['worker_active_cap_seconds'] or (JOB/'STOP_REQUESTED').exists():
            raise TimeoutError('bounded operator matrix stop')
        assert result['target_forwards']<=512

    def dev(x):
        if isinstance(x,torch.Tensor):return x.to(s.device)
        if isinstance(x,dict):return {k:dev(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return type(x)(dev(v) for v in x)
        return x

    def bytes_equal(a,b):
        return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))

    def error(a,b):
        a,b=a.detach().contiguous(),b.detach().contiguous();d=a.float()-b.float();flat=d.abs().flatten()
        qs=torch.quantile(flat,torch.tensor([.5,.9,.99],device=flat.device))
        out=dict(shape=list(a.shape),dtype=str(a.dtype),finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()),
            torch_equal=bool(torch.equal(a,b)),byte_equal=bool(bytes_equal(a,b)),different_elements=int(torch.count_nonzero(d)),
            elements=a.numel(),max_abs=float(flat.max()),rel_l2=float(d.norm()/b.float().norm().clamp_min(1e-30)),
            abs_p50=float(qs[0]),abs_p90=float(qs[1]),abs_p99=float(qs[2]),
            allclose_02=bool(torch.allclose(a,b,atol=.02,rtol=.02)),
            allclose_001=bool(torch.allclose(a,b,atol=.001,rtol=.001)),
            allclose_0001=bool(torch.allclose(a,b,atol=.0001,rtol=.0001)))
        if a.dtype==b.dtype==torch.bfloat16:
            def ordered(t):
                bits=t.view(torch.int16).to(torch.int32);mag=bits&32767
                return torch.where(bits<0,32768-mag,32768+mag)
            out['max_bf16_ulp']=int((ordered(a)-ordered(b)).abs().max())
        return out

    def capture_cache(cache):
        valid=cache['mask'].nonzero(as_tuple=False).flatten()
        return dict(shape=list(cache['k'].shape),slots=valid.clone(),k=cache['k'][:,valid].clone(),
            v=cache['v'][:,valid].clone(),id=cache['id'].clone(),mask=cache['mask'].clone(),is_pred=cache['is_pred'].clone())

    def pool(saved):
        out=dict(k=torch.empty(saved['shape'],device=s.device,dtype=torch.bfloat16),
                 v=torch.empty(saved['shape'],device=s.device,dtype=torch.bfloat16))
        for k in ('id','mask','is_pred'):out[k]=saved[k].clone()
        for k in ('k','v'):out[k][:,saved['slots']]=saved[k]
        return out

    def restore_history(history):
        for block,saved in zip(model.blocks,history):
            item=dict(shape=saved['pool_shape'],slots=dev(saved['slots']),k=dev(saved['k_valid']),v=dev(saved['v_valid']),
                      **{k:dev(saved[k]) for k in ('id','mask','is_pred')})
            block.attn1.attn_caches={'pos':pool(item)}

    # Pure original restoration helpers are compiled without executing digest helpers.
    snapfile=EXP/'fixed_load_cost/snapshot.py';tree=ast.parse(snapfile.read_text())
    names={'cpu','restore_cache','restore','seed'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    fields=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIELDS' for t in n.targets))
    ns=dict(torch=torch,np=np,random=random,copy=copy)
    exec(compile(ast.Module(body=[fields,*nodes],type_ignores=[]),str(snapfile)+':no-digest-pure-restoration','exec'),ns)

    def restore_state(state):
        restore_history(state['caches'])
        ns['restore'](s,state,cuda=True)
        s._p0_action_draft_mode='off'

    original_ops={}
    handles=[]
    def model_pre(module,args,kw):
        result['target_forwards']+=1;check()
        active=control['active'];control['current']=None
        if active is None:return
        inp=args[0] if args else kw['input_dict']
        if kw.get('update_cache',0)!=0:return
        kind='action' if kw.get('action_mode',False) else 'video'
        if active['batch']==14 and kind!='action':return
        key=(active['length'],active['batch'],kind)
        index=control['indices'].get(key,0);control['indices'][key]=index+1
        role='first' if index==0 else 'last'
        control['current']=dict(**active,kind=kind,role=role,query=index,
            timestep_min=float(inp['timesteps'].min()),timestep_max=float(inp['timesteps'].max()),
            candidate_rows=None if kw.get('candidate_cache_rows') is None else kw['candidate_cache_rows'].clone())

    handles.append(model.register_forward_pre_hook(model_pre,with_kwargs=True))
    for layer in selected:
        block=model.blocks[layer]
        def pre(module,args,kw,layer=layer):
            context=control['current']
            if context is None:return
            ident=(context['length'],context['batch'],context['kind'],context['role'],layer)
            data_points[ident]=dict(context=context.copy(),layer=layer,args=args,kwargs=kw,
                cache=capture_cache(module.attn1.attn_caches['pos']),attention={})
        def post(module,args,kw,y,layer=layer):
            context=control['current']
            if context is not None:
                ident=(context['length'],context['batch'],context['kind'],context['role'],layer)
                data_points[ident]['block_gold']=y.detach()
        handles += [block.register_forward_pre_hook(pre,with_kwargs=True),block.register_forward_hook(post,with_kwargs=True)]
        for part,attention in [('self',block.attn1),('cross',block.attn2)]:
            original_ops[layer,part]=attention.attn_op
            def make(original,layer,part):
                def op(q,k,v):
                    y=original(q,k,v);context=control['current']
                    if context is not None:
                        ident=(context['length'],context['batch'],context['kind'],context['role'],layer)
                        data_points[ident]['attention'][part]=dict(q=q.detach(),k=k.detach(),v=v.detach(),gold=y.detach())
                    return y
                return op
            attention.attn_op=make(attention.attn_op,layer,part)

    try:
        short=torch.load(p['short']['path'],map_location='cpu',weights_only=False,mmap=True)
        for label in ('short','middle','long'):
            check();control['active']=None
            if label=='middle':
                restore_state(short['state']);artifacts={(x['frame'],x['kind']):x for x in p['middle_prefix_artifacts']}
                s.exp_save_root=str(JOB/'middle-prefix-replay');Path(s.exp_save_root).mkdir()
                for frame in (4,8,12,16,20,24):
                    assert s.frame_st_id==frame
                    saved={kind:torch.load(artifacts[frame,kind]['path'],map_location='cpu',weights_only=False)
                           for kind in ('latents','actions','obs_data')}
                    s.infer(dict(p0_counterfactual_prime_video_cache=True,frame_st_id=frame,video_terminal_state=saved['latents']))
                    s.infer(dict(compute_kv_cache=True,obs=saved['obs_data'],state=s.postprocess_action(saved['actions'])))
                assert s.frame_st_id==28
                ns['seed'](20260922,True)
                context=copy.deepcopy(short['request']['p0_trace_context']);context['fixed_fixture']='T-init3-chunk7'
                request=dict(obs=saved['obs_data'][-1],p0_trace_context=context)
            else:
                payload=short if label=='short' else torch.load(p['long']['path'],map_location='cpu',weights_only=False,mmap=True)
                restore_state(payload['state']);request=copy.deepcopy(payload['request'])
            control['active']=dict(length=label,batch=2,source='canonical_target_full50_video20_frozen_policy_replay')
            s.exp_save_root=str(JOB/('canonical-'+label));Path(s.exp_save_root).mkdir()
            s.infer(request)
            control['active']=None
            # Real verifier fixture has its own exact condition/history, never a tiled copy of B2.
            frozen=torch.load(Path(PLAN['fixture_root'])/label/'real-verifier-fixture.pt',map_location='cpu',weights_only=False,mmap=True)
            restore_history(frozen['history'])
            for query in frozen['queries']:
                control['active']=dict(length=label,batch=14,source='frozen_real_seven_candidate_CFG2_verifier')
                inputs=dev(query['input']);kw=dev(query['kwargs']);kw.pop('input_dict',None)
                y=model(inputs,**kw)
                parity=error(y,dev(query['output']))
                result['full_query_oracles'].append(dict(length=label,query=query['index'],**parity))
                assert parity['finite'] and parity['byte_equal'],'frozen B14 query oracle not reproduced; no performance claim'
            control['active']=None
        source.save_async.__globals__['executor'].shutdown(wait=True)
        for h in handles:h.remove()
        for (layer,part),original in original_ops.items():
            getattr(model.blocks[layer],'attn1' if part=='self' else 'attn2').attn_op=original
        # Mechanical private-history splice from the existing implementation, with no digest execution.
        adapter=EXP/'single_worker_chc_gpu_v4/adapter.py'
        fn=next(n for n in ast.parse(adapter.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='_private_forward')
        namespace=dict(ast=ast,copy=copy,types=types,MODEL_SOURCE=MODEL,LeaseError=RuntimeError,__name__='operator_kv_matrix_v1')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(adapter)+':pure-private-forward','exec'),namespace)
        private=namespace['_private_forward']
        print('CAPTURE_COMPLETE',len(data_points),'target_forwards',result['target_forwards'],flush=True)

        def measure(*args,**kwargs):
            return dict(status='NOT_TIMED_SHARED_GPU_CORRECTNESS_ONLY')

        for index,(identity,item) in enumerate(sorted(data_points.items())):
            check();c=item['context'];block=model.blocks[item['layer']];saved=item['cache'];cache=pool(saved)
            block.attn1.attn_caches['pos']=cache
            row=dict(length=c['length'],batch=c['batch'],kind=c['kind'],query_role=c['role'],query_index=c['query'],
                timestep_min=c['timestep_min'],timestep_max=c['timestep_max'],layer=item['layer'],source=c['source'],
                prefix_tokens=len(saved['slots']),attention={},block={})
            for part,t in item['attention'].items():
                q,k,v,gold=[t[x].contiguous() for x in ('q','k','v','gold')]
                fns=dict(torch=lambda q=q,k=k,v=v:torch_sdpa(q,k,v),
                    standalone=lambda q=q,k=k,v=v:cuda_sdpa_standalone(q,k,v))
                if c['batch']==14 and part=='self':
                    assert tuple(q.shape)==(14,16,24,128) and k.shape[1]==len(saved['slots'])+16
                    ck,cv=k[:,-16:].contiguous(),v[:,-16:].contiguous();mapping=c['candidate_rows']
                    assert torch.equal(mapping,torch.tensor([0]*7+[1]*7,device=s.device))
                    group=2 if (len(saved['slots'])+32+63)//64==(len(saved['slots'])+16+63)//64 else 1
                    fns.update(shared=lambda q=q,ck=ck,cv=cv,mapping=mapping:native.shared(q,cache['k'],cache['v'],ck,cv,saved['slots'],mapping),
                        G1=lambda q=q,ck=ck,cv=cv:grouped.grouped(q,cache['k'],cache['v'],ck,cv,saved['slots'],1),
                        guarded=lambda q=q,ck=ck,cv=cv,group=group:grouped.grouped(q,cache['k'],cache['v'],ck,cv,saved['slots'],group))
                accuracy={name:error(fn(),gold) for name,fn in fns.items()}
                row['attention'][part]=dict(q_shape=list(q.shape),k_shape=list(k.shape),dtype=str(q.dtype),
                    reference_backend=int(torch._fused_sdp_choice(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2))),
                    dense_materialized_KV_bytes=(k.numel()+v.numel())*k.element_size(),accuracy=accuracy,
                    eager=measure(fns),graph_core=measure(fns,graph=True))
            original_forward=block.attn1.forward;original_self=block.attn1.attn_op;original_cross=block.attn2.attn_op
            engines=['torch','standalone']+(['shared','guarded'] if c['batch']==14 else [])
            def before(engine):
                block.attn1.forward=original_forward
                block.attn1.attn_op=block.attn2.attn_op=torch_sdpa if engine!='standalone' else cuda_sdpa_standalone
                if engine in ('shared','guarded'):
                    group=2 if (len(saved['slots'])+32+63)//64==(len(saved['slots'])+16+63)//64 else 1
                    def history(attn,q,k,v,pool,name,mapping,update):
                        assert pool is cache and update==0
                        if engine=='shared':return native.shared(q.contiguous(),pool['k'],pool['v'],k.contiguous(),v.contiguous(),saved['slots'],mapping)
                        return grouped.grouped(q.contiguous(),pool['k'],pool['v'],k.contiguous(),v.contiguous(),saved['slots'],group)
                    block.attn1.forward=types.MethodType(private(original_forward,history),block.attn1)
            block_fns={name:(lambda:block(*item['args'],**item['kwargs'])) for name in engines}
            accuracy={}
            try:
                for name,fn in block_fns.items():before(name);accuracy[name]=error(fn(),item['block_gold'])
                row['block']=dict(accuracy=accuracy,eager=measure(block_fns,before,block=True),
                    scope='real block with original projections/norm/FFN; shared/guard use admitted static history binding')
            finally:
                block.attn1.forward=original_forward;block.attn1.attn_op=original_self;block.attn2.attn_op=original_cross
            row['valid_KV_preserved']=all(bytes_equal(cache[k][:,saved['slots']],saved[k]) for k in ('k','v'))
            row['valid_metadata_preserved']=all(bytes_equal(cache[k][saved['slots']],saved[k][saved['slots']]) for k in ('id','mask','is_pred'))
            assert row['valid_KV_preserved'] and row['valid_metadata_preserved']
            result['points'].append(row);write('RESULT.json',result)
            print('POINT',index+1,len(data_points),identity,flush=True)
        assert len(result['points'])==54
        # Entire 30-layer frozen verifier queries, separate from the block checks.
        del data_points
        gc.collect();torch.cuda.empty_cache()
        all_methods=[b.attn1.forward for b in model.blocks]
        all_ops=[(b.attn1.attn_op,b.attn2.attn_op) for b in model.blocks]
        result['complete_model_queries']=[]
        def rng_state():
            return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
        def same_rng(a,b):
            return (a[0]==b[0] and a[1][0]==b[1][0] and np.array_equal(a[1][1],b[1][1])
                and a[1][2:]==b[1][2:] and torch.equal(a[2],b[2])
                and len(a[3])==len(b[3]) and all(torch.equal(x,y) for x,y in zip(a[3],b[3])))
        def cfg(value):
            assert value.shape[0]==14 and value.shape[-1]==s.job_config.action_dim
            cond,uncond=value[:7],value[7:]
            return uncond+s.job_config.action_guidance_scale*(cond-uncond) if s.job_config.action_guidance_scale>1 else cond
        try:
            for label in ('short','middle','long'):
                frozen=torch.load(Path(PLAN['fixture_root'])/label/'real-verifier-fixture.pt',map_location='cpu',weights_only=False,mmap=True)
                restore_history(frozen['history'])
                baseline=[capture_cache(b.attn1.attn_caches['pos']) for b in model.blocks]
                for query in frozen['queries']:
                    inputs=dev(query['input']);kw=dev(query['kwargs']);kw.pop('input_dict',None)
                    input_reference={k:v.clone() for k,v in inputs.items()}
                    gold=dev(query['output']);gold_cfg=cfg(gold)
                    engines={}
                    for engine in ('torch','standalone','shared','guarded'):
                        check()
                        for lid,b in enumerate(model.blocks):
                            b.attn1.forward=all_methods[lid]
                            b.attn1.attn_op=b.attn2.attn_op=torch_sdpa if engine!='standalone' else cuda_sdpa_standalone
                            if engine in ('shared','guarded'):
                                current_pool=b.attn1.attn_caches['pos'];slots=baseline[lid]['slots']
                                group=2 if (len(slots)+32+63)//64==(len(slots)+16+63)//64 else 1
                                def history(attn,q,k,v,pool,name,mapping,update,*,owned_pool=current_pool,valid=slots,kind=engine,g=group):
                                    assert pool is owned_pool and update==0
                                    if kind=='shared':return native.shared(q.contiguous(),pool['k'],pool['v'],k.contiguous(),v.contiguous(),valid,mapping)
                                    return grouped.grouped(q.contiguous(),pool['k'],pool['v'],k.contiguous(),v.contiguous(),valid,g)
                                b.attn1.forward=types.MethodType(private(all_methods[lid],history),b.attn1)
                        before=rng_state();result['target_forwards']+=1;check()
                        output=model(inputs,**kw)
                        after=rng_state()
                        state_ok=[]
                        for b,ref in zip(model.blocks,baseline):
                            actual=b.attn1.attn_caches['pos']
                            state_ok.append(all(bytes_equal(actual[k][:,ref['slots']],ref[k]) for k in ('k','v'))
                                and bytes_equal(actual['mask'],ref['mask'])
                                and all(bytes_equal(actual[k][ref['slots']],ref[k][ref['slots']]) for k in ('id','is_pred')))
                        raw=error(output,gold);post_cfg=error(cfg(output),gold_cfg)
                        channels=s.job_config.used_action_channel_ids
                        active=error(cfg(output)[...,channels],gold_cfg[...,channels])
                        input_ok=all(bytes_equal(inputs[k],v) for k,v in input_reference.items())
                        rng_ok=same_rng(before,after)
                        engines[engine]=dict(raw=raw,CFG=post_cfg,active_action_CFG=active,
                            all30_valid_KV_and_metadata_unchanged=all(state_ok),unchanged_layers=sum(state_ok),
                            inputs_unchanged=input_ok,RNG_unchanged=rng_ok,
                            strict_bitwise_input_output_state_pass=raw['byte_equal'] and post_cfg['byte_equal'] and all(state_ok) and input_ok and rng_ok)
                        torch.save(output.detach().cpu(),JOB/f'{label}-q{query["index"]}-{engine}-complete-output.pt')
                        if engine=='torch':assert raw['byte_equal'],'original complete frozen model oracle mismatch'
                    result['complete_model_queries'].append(dict(length=label,query=query['index'],engines=engines,
                        scope='full30-layer B14 query, CFG postprocess, all valid KV, input bytes and RNG; no scheduler/chunk/robot rollout'))
                    write('RESULT.json',result)
                    print('FULL_MODEL_QUERY',label,query['index'],{k:v['raw']['byte_equal'] for k,v in engines.items()},flush=True)
        finally:
            for lid,b in enumerate(model.blocks):
                b.attn1.forward=all_methods[lid];b.attn1.attn_op,b.attn2.attn_op=all_ops[lid]
        result.update(status='COMPLETE_54_POINT_AND_FULL30_QUERY_VALIDATION',
            elapsed_seconds=time.monotonic()-START,all_valid_KV_and_metadata_preserved=True,
            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        write('RESULT.json',result)
    except BaseException as exc:
        result.update(status='PARTIAL_OR_FAILED_MATRIX',error=repr(exc),traceback=traceback.format_exc(),elapsed_seconds=time.monotonic()-START)
        write('RESULT.json',result);raise
    finally:
        for h in handles:h.remove()
        (JOB/'IN-TIMING').unlink(missing_ok=True)


def main():
    os.sched_setaffinity(0,set(PLAN['CPU_worker']))
    sys.path.insert(0,str(ROOT/'投机采样/wam-p0-worktree'))
    from wan_va import wan_va_server as source
    def execute(server,rank,host,port):
        assert rank==0
        with source.torch.no_grad():run(server,source)
    source.run_async_server_mode=execute
    source.init_logger()
    source.main()


if __name__=='__main__':main()
