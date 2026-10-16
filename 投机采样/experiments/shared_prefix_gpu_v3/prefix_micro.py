"""Bounded GPU micro diagnostic; never substituted for target/verifier speed."""
import json
import math
import os
import statistics
import time


def record_intent(rt, purpose):
    rt.check_stop()
    count=getattr(rt,'direct_micro_count',0)+1
    if count>rt.p['direct_attention_micro_call_cap']:raise RuntimeError('direct attention micro cap')
    rt.direct_micro_count=count
    path=rt.job/'worker/direct-attention-micro.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(dict(intent=count,purpose=purpose,monotonic=time.monotonic()))+'\n');f.flush();os.fsync(f.fileno())
    return count


def run_micro(rt):
    t=rt.t
    from prefix_attention import shared_attention
    from prefix_compare import numerical
    def call(fn, timed=False):
        record_intent(rt,'synthetic-kernel-check-or-micro')
        if not timed: return fn()
        rt.sync();begin=t.cuda.Event(enable_timing=True);end=t.cuda.Event(enable_timing=True)
        begin.record();out=fn();end.record();rt.sync()
        return out,begin.elapsed_time(end)
    result=[];bench=[];seed=t.Generator().manual_seed(7241)
    for length,strided,amplitude in ((0,False,1.),(65,False,1.),(2144,False,1.),(65,True,1.),(2144,True,1.),(65,False,8.)):
        shape=(14,16,24,128); pool=(2,2*length+3,24,128)
        q=t.randn(shape,dtype=t.bfloat16,generator=seed).to(rt.device)*amplitude
        ck=t.randn(shape,dtype=t.bfloat16,generator=seed).to(rt.device)
        cv=t.randn(shape,dtype=t.bfloat16,generator=seed).to(rt.device)
        pk=t.full(pool,float('nan'),dtype=t.bfloat16,device=rt.device);pv=t.full_like(pk,float('nan'))
        slots=t.arange(length,dtype=t.long,device=rt.device)*2+1
        rows=t.tensor([0]*7+[1]*7,dtype=t.long,device=rt.device)
        pk[:,slots]=t.randn((2,length,24,128),dtype=t.bfloat16,generator=seed).to(rt.device)
        pv[:,slots]=t.randn((2,length,24,128),dtype=t.bfloat16,generator=seed).to(rt.device)
        if strided:
            q,ck,cv,pk,pv=(x.transpose(1,2).contiguous().transpose(1,2) for x in (q,ck,cv,pk,pv))
        def original():
            hk=pk[:,slots].index_select(0,rows);hv=pv[:,slots].index_select(0,rows)
            return t.nn.functional.scaled_dot_product_attention(q.transpose(1,2),t.cat((hk,ck),1).transpose(1,2),t.cat((hv,cv),1).transpose(1,2)).transpose(1,2)
        def shared():
            return shared_attention(q,pk,pv,ck,cv,slots,rows,enabled=True,allow_cuda=True,maps_validated=True)
        a=call(original);b=call(shared);rt.sync()
        row=numerical(b.cpu(),a.cpu());row.update(length=length,strided=strided,amplitude=amplitude)
        if not strided and length in (65,2144) and amplitude==1:
            def fp64_oracle():
                from torch.nn.attention import sdpa_kernel, SDPBackend
                kk=t.cat((pk[:,slots].index_select(0,rows),ck),1).double()
                vv=t.cat((pv[:,slots].index_select(0,rows),cv),1).double()
                with sdpa_kernel(SDPBackend.MATH):
                    return t.nn.functional.scaled_dot_product_attention(q.double().transpose(1,2),kk.transpose(1,2),vv.transpose(1,2)).transpose(1,2).to(q.dtype)
            gold=call(fp64_oracle);rt.sync()
            row['S_vs_FP64_joint_oracle']=numerical(b.cpu(),gold.cpu())
            row['original_SDPA_vs_FP64_joint_oracle']=numerical(a.cpu(),gold.cpu())
            del gold
        # Diagnostic corruption ceilings, not target acceptance tolerances.
        if not row['finite'] or row['relative_RMS_error']>.01 or row['max_abs_error']>.03125:
            raise ValueError('kernel diagnostic numerical ceiling '+str(row))
        if length==2144 and not strided:
            old=b.clone();ck[0].add_(.25);cv[0].add_(1)
            new=call(shared);rt.sync()
            if not t.equal(old[1:],new[1:]) or t.equal(old[0],new[0]):raise ValueError('kernel other13 isolation')
            ck[0].sub_(.25);cv[0].sub_(1)
        result.append(row)
        if rt.p['diagnostic_nonbitwise_approved'] and not strided and length in (65,2144) and amplitude==1:
            # R includes materialization + concat + original SDPA. S reads
            # existing prefix2; metadata is precomputed for BOTH arms here.
            for fn in (original,shared):
                for _ in range(4):call(fn)
            data={'R':[],'S':[]}
            for block in range(4):
                for arm in (('R','S') if block%2==0 else ('S','R')):
                    fn=original if arm=='R' else shared
                    samples=[]
                    for _ in range(8):
                        out,ms=call(fn,timed=True);samples.append(ms)
                    data[arm].extend(samples)
            bench.append(dict(length=length,R_ms=statistics.median(data['R']),S_ms=statistics.median(data['S']),
                              raw_ms=data,scope='single-layer micro: Q/current/prefix preloaded, metadata precomputed, no Transformer',
                              persistent_intent_io_outside_micro_event_span=True,not_full_verifier_speed=True))
        del q,ck,cv,pk,pv,slots,rows,a,b
    rt.sync()
    return dict(status='PASS_KERNEL_DIAGNOSTIC',numerical=result,bench=bench,direct_attention_micro_calls=rt.direct_micro_count,
                direct_attention_micro_cap=rt.p['direct_attention_micro_call_cap'],target_forwards=0,
                CFG_rows=2,current_rows=14,history14_materialized_on_S=False,fullKV14_materialized_on_S=False,
                bitwise_is_separate_from_math=True,target_acceptance_not_proven=True,
                performance_skipped_without_diagnostic_authorization=not rt.p['diagnostic_nonbitwise_approved'])
