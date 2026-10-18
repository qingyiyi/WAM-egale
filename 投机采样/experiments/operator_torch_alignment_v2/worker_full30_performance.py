"""Bounded actual 30-layer verifier alignment timing, no robot/task claims."""
from pathlib import Path
import json,os,sys,time,statistics,traceback
ROOT=Path(__file__).resolve().parents[3]
JOB=Path(os.environ['OPERATOR_MATRIX_JOB']).resolve()
PLAN=json.loads((JOB/'PLAN.json').read_text())
START=time.monotonic()

def write(data):
    p=JOB/'RESULT.json';t=p.with_suffix('.pending');t.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n');t.replace(p)

def run(s):
    import torch
    from integration import installed
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.70)
    model=s.transformer
    assert len(model.blocks)==30 and s._cache_mode=='original' and s._p0_action_draft_mode=='off'
    assert torch.version.git_version=='0fabc3ba44823f257e70ce397d989c8de5e362c1'
    s.infer(dict(reset=True,prompt=None))
    result=dict(status='RUNNING',rows=[],target_forwards=0,episodes=0,
        engines=['torch_canonical_dense_B14','aligned_v2_dense_B14'],
        scope='complete actual30-layer B14 query eager, same input/history/layout; no cache optimization or robot E2E',
        resource_qualification='empty context at admission; known idle utilization anomaly remains',
        input_shape='14 independent Q16, seven candidate states CFG2',
        prompt_projection_cache=False,CUDA_Graph=False)
    def check():
        assert time.monotonic()-START<PLAN['worker_active_cap_seconds']
        assert result['target_forwards']<=128
    def dev(x):
        if isinstance(x,torch.Tensor):return x.to(s.device)
        if isinstance(x,dict):return {k:dev(v) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return type(x)(dev(v) for v in x)
        return x
    def equal(a,b):
        return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
    def call(inputs,kw):
        check();result['target_forwards']+=1
        return model(inputs,**kw)
    try:
        for label in ['short','middle','long']:
            frozen=torch.load(Path(PLAN['fixture_root'])/label/'real-verifier-fixture.pt',map_location='cpu',weights_only=False,mmap=True)
            for b,h in zip(model.blocks,frozen['history']):
                cache={k:dev(h[k]).clone() for k in ['id','mask','is_pred']}
                cache.update(k=torch.empty(h['pool_shape'],dtype=torch.bfloat16,device=s.device),v=torch.empty(h['pool_shape'],dtype=torch.bfloat16,device=s.device))
                slots=dev(h['slots'])
                cache['k'][:,slots]=dev(h['k_valid']);cache['v'][:,slots]=dev(h['v_valid'])
                b.attn1.attn_caches={'pos':cache}
            for query in frozen['queries']:
                check();inputs=dev(query['input']);kw=dev(query['kwargs']);kw.pop('input_dict',None);gold=dev(query['output'])
                # Verification/warmup is outside measured regions.
                first=call(inputs,kw)
                with installed(model):second=call(inputs,kw)
                assert equal(first,gold) and equal(second,gold),'same actual input full30 numeric gate'
                events={n:[] for n in ['torch','aligned']};walls={n:[] for n in events}
                a,z=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                (JOB/'IN-TIMING').write_text('No management API/profiler during paired measurements\n')
                try:
                    for rid in range(3):
                        for engine in ['torch','aligned'] if rid%2==0 else ['aligned','torch']:
                            check();ev=[];wa=[]
                            from contextlib import nullcontext
                            with installed(model) if engine=='aligned' else nullcontext():
                                for _ in range(1):
                                    torch.cuda.synchronize();begin=time.perf_counter();a.record();y=call(inputs,kw);z.record();z.synchronize()
                                    ev.append(a.elapsed_time(z));wa.append((time.perf_counter()-begin)*1000)
                            events[engine].append(ev);walls[engine].append(wa)
                finally:(JOB/'IN-TIMING').unlink(missing_ok=True)
                kv_ok=[]
                for b,h in zip(model.blocks,frozen['history']):
                    c=b.attn1.attn_caches['pos'];slots=dev(h['slots'])
                    kv_ok.append(equal(c['k'][:,slots],dev(h['k_valid'])) and equal(c['v'][:,slots],dev(h['v_valid'])) and all(equal(c[k],dev(h[k])) for k in ['id','mask','is_pred']))
                med={k:statistics.median(statistics.median(r) for r in v) for k,v in events.items()}
                row=dict(length=label,query_index=query['index'],raw_event_ms=events,raw_wall_ms=walls,
                    median_ms=med,aligned_over_torch=med['aligned']/med['torch'],original_and_new_output_byte_equal=True,
                    all30_valid_KV_metadata_unchanged=all(kv_ok),rounds=3,samples_per_engine_round=1)
                assert row['all30_valid_KV_metadata_unchanged']
                result['rows'].append(row);write(result)
                print('FULL30_PERFORMANCE',label,query['index'],med,flush=True)
        result.update(status='COMPLETE_SIX_REAL_FULL30_PERFORMANCE',elapsed_seconds=time.monotonic()-START,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        result.update(status='PARTIAL_FULL30_PERFORMANCE',error=repr(exc),traceback=traceback.format_exc());raise
    finally:write(result)

def main():
    os.sched_setaffinity(0,set(PLAN['CPU_worker']))
    sys.path.insert(0,str(ROOT/'投机采样/wam-p0-worktree'))
    from wan_va import wan_va_server as source
    def execute(server,rank,host,port):
        with source.torch.no_grad():run(server)
    source.run_async_server_mode=execute;source.init_logger();source.main()

if __name__=='__main__':main()
