"""Three real full chunks followed by frozen verifier mechanism replay.

No simulator, no new policy, no formal performance entry. Existing O3 is used
unchanged; B2/B6 are diagnostic subsets of the same seven frozen candidates.
"""
import copy,json,os,sys,time,traceback,statistics
from pathlib import Path
OWN=Path(__file__).resolve().parent
ROOT=OWN.parents[2];EXP=OWN.parent
sys.path.insert(0,str(EXP/'prefix_acceptance_v1'))
sys.path.append(str(EXP/'megaverifier_profile_v2'))
from paths import load

def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def run(s,source,out,p):
    import torch
    from runtime import Runtime
    from selection import install
    from measured_online import control
    from template import TemplateSharedPrefix
    import buffers,state_check
    snap=load('shape_matrix_snapshot',EXP/'fixed_load_cost/snapshot.py')
    old=load('shape_matrix_prior_profile',EXP/'megaverifier_profile_v2/profile_worker.py')
    start=time.monotonic()
    torch.set_num_threads(7)
    s.infer(dict(reset=True,prompt=None))
    # The mechanism launcher starts disabled. Admission to the already-frozen
    # A_off verifier must precede the source's guarded lazy draft loader.
    # middle() explicitly disables drafting again while rebuilding real caches.
    s._p0_action_draft_mode='verifier_k2_b2_bonus'
    s._load_p0_action_draft_b2_predictor().model.to(s.device).eval()
    proxy=install(s,'prefix_then_cost')
    buffers.prepare(s._infer_action_verifier_k2_b2,s._load_p0_shadow_api())
    rt=Runtime(s,out,1800,shared=False,allow_cuda=True)
    short=torch.load(p['short']['path'],map_location='cpu',weights_only=False)
    for k in ('state','request'):assert snap.fingerprint(short[k])==p['short'][k+'_hash']
    mid=old.middle(s,rt,snap,short,p,out)
    result=dict(status='RUNNING',scope='frozen real query mechanism, not robot speed',rows=[],chunks=[],profiles=[],graph_status='UNAVAILABLE_UNMODIFIED_FULL_VERIFIER: dynamic nonzero and scalar ownership validation; operator Graph only measured separately',new_engine=False)
    def check():
        if time.monotonic()-start>1650 or (out/'STOP_REQUESTED').exists():raise TimeoutError('matrix time/stop guard')
    def compare(a,b):
        a=a.detach().cpu().contiguous();b=b.detach().cpu().contiguous();d=(a.float()-b.float()).abs()
        return dict(bitwise=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),max_abs=float(d.max()),different=int(torch.count_nonzero(d)),finite=bool(torch.isfinite(a).all()))
    def to_device(x):
        if isinstance(x,torch.Tensor):return x.to(s.device)
        if isinstance(x,dict):return {k:to_device(v) for k,v in x.items()}
        if isinstance(x,tuple):return tuple(to_device(v) for v in x)
        if isinstance(x,list):return [to_device(v) for v in x]
        return x
    def subset(x,idx):
        if isinstance(x,torch.Tensor) and x.ndim and x.shape[0]==14:return x[idx].contiguous()
        if isinstance(x,dict):return {k:subset(v,idx) for k,v in x.items()}
        if isinstance(x,tuple):return tuple(subset(v,idx) for v in x)
        if isinstance(x,list):return [subset(v,idx) for v in x]
        return x
    def timing(fn):
        fn();torch.cuda.synchronize();vals=[]
        for _ in range(6):
            check();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();t=time.perf_counter();y=fn();b.record();b.synchronize()
            vals.append(dict(wall_ms=1000*(time.perf_counter()-t),cuda_ms=a.elapsed_time(b)))
        return dict(samples=vals,median_wall_ms=statistics.median(x['wall_ms'] for x in vals),median_cuda_ms=statistics.median(x['cuda_ms'] for x in vals)),y
    try:
      for label,history,total in [('short',144,288),('middle',1008,1152),('long',2016,2160)]:
        check();rt.shared.close()
        payload=short if label=='short' else mid if label=='middle' else torch.load(p['long']['path'],map_location='cpu',weights_only=False)
        ref=mid if label=='middle' else p[label]
        for k in ('state','request'):assert snap.fingerprint(payload[k])==ref[k+'_hash']
        snap.restore(s,payload['state'])
        assert all(len(c['slots'])==history for c in payload['state']['caches'])
        s._p0_action_draft_mode='verifier_k2_b2_bonus';s._p0_action_draft_selector=proxy.effective='prefix_then_cost'
        proxy.records.clear();proxy.pending=None
        rt.shared=control(s,'O3',allow_cuda=True)
        cap=dict(active=False,count=0,first=None,last=None,history=None)
        def pre(model,args,kw):
            inp=args[0] if args else kw['input_dict']
            cap['active']=kw.get('candidate_cache_rows') is not None
            if not cap['active']:return
            assert inp['noisy_latents'].shape[0]==14 and kw['update_cache']==0
            if cap['history'] is None:cap['history']=[snap.capture_cache(b.attn1.attn_caches[s.cache_name]) for b in s.transformer.blocks]
            cap['current']=dict(input=snap.cpu(inp),kwargs=snap.cpu(kw),index=cap['count'])
        def post(model,args,kw,y):
            if not cap['active']:return
            item=cap.pop('current');item['output']=snap.cpu(y)
            if cap['first'] is None:cap['first']=item
            cap['last']=item;cap['count']+=1;cap['active']=False
        handles=[s.transformer.register_forward_pre_hook(pre,with_kwargs=True),s.transformer.register_forward_hook(post,with_kwargs=True)]
        folder=out/label;folder.mkdir();s.exp_save_root=str(folder/'policy');Path(s.exp_save_root).mkdir()
        before=dict(rt.phase);beforecount=rt.attempted
        try:returned=s.infer(copy.deepcopy(payload['request']))
        finally:
            for h in handles:h.remove()
            rt.shared.close()
        phase={k:v-before.get(k,0) for k,v in rt.phase.items() if v-before.get(k,0)}
        assert phase['committed_scheduler_update_count']==50 and len(proxy.records)==phase['tree_count']
        assert cap['count']>=2 and all(len(c['slots'])==total-16 for c in cap['history'])
        # Exact shared text is stored once per length, without changing shape.
        assert torch.equal(cap['first']['input']['text_emb'],cap['last']['input']['text_emb'])
        cap['last']['input']['text_emb']=cap['first']['input']['text_emb']
        torch.save(dict(history=cap['history'],queries=[cap['first'],cap['last']],source=p[label] if label!='middle' else dict(source='real 18-cache-forward replay'),total_KV=total),folder/'real-verifier-fixture.pt')
        write(folder/'CHUNK.json',dict(phase=phase,forwards=rt.attempted-beforecount,candidates=cap['count'],records=proxy.records,shared=rt.shared.summary(),complete_scheduler_50=True,closed_loop_episode=False))
        result['chunks'].append(dict(length=label,total_KV=total,candidates=cap['count'],phase=phase))
        for b,c in zip(s.transformer.blocks,cap['history']):snap.restore_cache(b.attn1.attn_caches[s.cache_name],c)
        refs=state_check.references(s.transformer,s.cache_name)
        rng=snap.fingerprint(snap.rng(True))
        for query in (cap['first'],cap['last']):
          inp=to_device(query['input']);kwargs=to_device(query['kwargs']);kwargs.pop('input_dict',None)
          full=lambda:s.transformer(inp,**kwargs)
          raw=full();assert compare(raw,query['output'])['bitwise'],'raw B14 vs captured O3 mismatch'
          for n in (1,3,7):
            idx=torch.tensor(list(range(n))+list(range(7,7+n)),device=s.device)
            si=subset(inp,idx);sk=subset(kwargs,idx)
            parts=[]
            for j in range(n):
                ix=torch.tensor([j,j+7],device=s.device);parts.append((subset(inp,ix),subset(kwargs,ix)))
            def sequential():
                ys=[s.transformer(a,**b) for a,b in parts]
                return torch.cat([y[:1] for y in ys]+[y[1:] for y in ys])
            batched=lambda:s.transformer(si,**sk)
            times={};ys={}
            arms=[('sequential_B2',sequential),('existing_torch_batch',batched)]
            if query['index']!=0:arms.reverse()
            for name,fn in arms:times[name],ys[name]=timing(fn)
            row=dict(length=label,total_KV=total,query_index=query['index'],n_v=n,CFG=2,shape=[2*n,16,3072],timing=times,seq_vs_batch=compare(ys['sequential_B2'],ys['existing_torch_batch']),subset_vs_full=compare(ys['existing_torch_batch'],raw[idx]),shared_n1_n3='N/A fixed B14 native contract')
            if n==7:
                c=TemplateSharedPrefix(s.transformer,enabled=True,allow_cuda=True).install()
                try:
                    enter=time.perf_counter()
                    with c.lease(request='shape-matrix',arm='S',cache_name=s.cache_name,device=s.device) as token:
                        row['shared_lease_entry_wall_ms_excluded']=1000*(time.perf_counter()-enter)
                        def shared():return s.transformer(inp,**{**kwargs,'candidate_cache_rows':token.rows})
                        # Production keeps a lease across candidate forwards.
                        # One warmup primes entries, then measure this live lease.
                        times['existing_shared_B14'],ys['existing_shared_B14']=timing(shared)
                        row['shared_timing_scope']='live existing lease; warm entry; includes per-forward checks; pack and lease entry/exit excluded'
                    row['shared_vs_raw']=compare(ys['existing_shared_B14'],raw)
                    assert row['shared_vs_raw']['bitwise']
                finally:c.close()
            result['rows'].append(row)
            print(json.dumps(dict(length=label,query=query['index'],n_v=n,wall_ms={k:v['median_wall_ms'] for k,v in times.items()})),flush=True)
          if query['index']==0:
            c=TemplateSharedPrefix(s.transformer,enabled=True,allow_cuda=True).install();hh=[];ledger=[]
            def linear_pre(name):
                def f(m,a):
                    x=a[0];ledger.append(dict(name=name,input=list(x.shape),stride=list(x.stride()),dtype=str(x.dtype),weight=list(m.weight.shape),MNK=[x.numel()//x.shape[-1],m.weight.shape[0],x.shape[-1]]))
                return f
            for name,m in s.transformer.named_modules():
                if isinstance(m,torch.nn.Linear):hh.append(m.register_forward_pre_hook(linear_pre(name)))
            try:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                    with c.lease(request='shape-profile',arm='S',cache_name=s.cache_name,device=s.device) as token:s.transformer(inp,**{**kwargs,'candidate_cache_rows':token.rows})
                    torch.cuda.synchronize()
                prof.export_chrome_trace(str(folder/'PROFILE-shared30-first.json'))
                write(folder/'LINEAR-LEDGER.json',ledger)
                result['profiles'].append(dict(length=label,linear_calls=len(ledger),trace=str(folder/'PROFILE-shared30-first.json')))
            finally:
                for h in hh:h.remove()
                c.close()
          assert state_check.equal(s.transformer,refs,s.cache_name),'all-valid KV changed'
          assert snap.fingerprint(snap.rng(True))==rng,'RNG changed'
        del refs,cap,payload;torch.cuda.empty_cache()
      rt.close();source.save_async.__globals__['executor'].shutdown(wait=True)
      result.update(status='COMPLETE_THREE_LENGTH_FULL_VERIFIER_MECHANISM',target_intents=rt.attempted,elapsed_sec=time.monotonic()-start,all_valid_KV_and_RNG_unchanged=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),formal_robot_performance=False)
    except BaseException as exc:
      result.update(status='FAILED',error=repr(exc),traceback=traceback.format_exc(),target_intents=rt.attempted);raise
    finally:write(out/'MATRIX-RESULT.json',result)

def main():
    os.sched_setaffinity(0,set(range(1,8)))
    out=Path(os.environ['WAM_SHAPE_OUT']);p=json.loads((out/'SOURCE-PLAN.json').read_text())
    sys.path.insert(0,str(ROOT/'投机采样/wam-p0-worktree'))
    from wan_va import wan_va_server as source
    def execute(s,rank,host,port):
        assert rank==0
        with source.torch.no_grad():run(s,source,out,p)
    source.run_async_server_mode=execute
    try:source.init_logger();source.main()
    except BaseException as exc:
        write(out/'WORKER-FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
