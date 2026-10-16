"""Real long-KV attention and actual output-projection replay, not a task benchmark."""
import argparse,json,os,signal,sys,time,traceback
from pathlib import Path

OWN=Path(__file__).resolve().parent
ROOT=OWN.parents[2]
OBS=ROOT/'投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534/observation'
MODEL=ROOT/'models/lingbot-va-posttrain-libero-long-modelscope-20260922/transformer'

def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    out=Path(sys.argv[1]);start=time.monotonic();os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    import torch
    from safetensors import safe_open
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    sys.path.insert(0,str(OWN.parent/'shared_prefix_torch_v2'))
    import _shared_torch_v1 as native
    device=torch.cuda.get_device_properties(0)
    assert device.major==8 and device.minor==0 and device.multi_processor_count==108
    result=dict(status='RUNNING',scope='real long2160 attention and attn1 output Linear only',torch=torch.__version__,cuda=torch.version.cuda,GPU=device.name,profile_is_not_timing=True,full_block=False,full_verifier=False,full_chunk=False,full_episode=False,rows=[],profiles=[],calls=0)
    def check():
        if time.monotonic()-start>540 or (out/'STOP_REQUESTED').exists():raise TimeoutError('bounded replay stop')
        if result['calls']>=12000:raise RuntimeError('operator call cap')
    def call(fn):
        check();result['calls']+=1;return fn()
    def numbers(a,b):
        d=(a.float()-b.float()).abs()
        return dict(bitwise=torch.equal(a.view(torch.uint8),b.contiguous().view(torch.uint8)),max_abs=float(d.max()),different_elements=int(torch.count_nonzero(d)),finite=bool(torch.isfinite(a).all()))
    def measure(fn):
        for _ in range(3):call(fn)
        torch.cuda.synchronize()
        # A graph contains one operator/path. 6 rounds, 30 replay iterations each.
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):call(fn)
        torch.cuda.current_stream().wait_stream(stream)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):output=fn()
        result['calls']+=1
        vals=[]
        for _ in range(6):
            check();a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(30):graph.replay();result['calls']+=1
            b.record();b.synchronize();vals.append(a.elapsed_time(b)/30)
        return dict(cuda_graph_ms=vals,median_ms=float(__import__('statistics').median(vals)),graph_single_path=True)
    try:
        rng=torch.get_rng_state().clone();crng=torch.cuda.get_rng_state().clone()
        index=json.loads((MODEL/'diffusion_pytorch_model.safetensors.index.json').read_text())['weight_map']
        for layer in (0,8,29):
            check();payload=torch.load(OBS/f'layer{layer:02d}-tensors.pt',map_location='cpu',weights_only=True)
            meta=json.loads((OBS/f'layer{layer:02d}-metadata.json').read_text())
            q,ck,cv,pk,pv=[payload[k].cuda() for k in ('q','current_k','current_v','prefix_k','prefix_v')]
            assert q.shape==(14,16,24,128) and pk.shape[0]==2 and pk.shape[1]+16==2160
            fullpk=torch.full((2,2160,24,128),float('nan'),device='cuda',dtype=q.dtype);fullpv=torch.full_like(fullpk,float('nan'))
            slots=torch.tensor(meta['valid_slots'],device='cuda',dtype=torch.long)
            fullpk[:,slots]=pk;fullpv[:,slots]=pv
            base=payload['original_out'].cuda().flatten(2,3)
            tensors={}
            for suffix in ('weight','bias'):
                key=f'blocks.{layer}.attn1.to_out.0.{suffix}'
                with safe_open(str(MODEL/index[key]),framework='pt',device='cpu') as f:tensors[suffix]=f.get_tensor(key).cuda()
            weight,bias=tensors['weight'],tensors['bias']
            versions=[x._version for x in (q,ck,cv,pk,pv,fullpk,fullpv,slots,base,weight,bias)]
            sdpa=lambda q,k,v:torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2)).transpose(1,2).contiguous()
            fullrows=torch.tensor([0]*7+[1]*7,device='cuda')
            ref=call(lambda:sdpa(q,torch.cat((pk[fullrows],ck),1),torch.cat((pv[fullrows],cv),1)))
            frozen=numbers(ref,payload['original_out'].cuda())
            if not frozen['bitwise']:raise ValueError('current torch disagrees with frozen original output')
            for n in (1,3,7):
                ids=list(range(n))+list(range(7,7+n));idx=torch.tensor(ids,device='cuda')
                qs,ks,vs=q[idx].contiguous(),ck[idx].contiguous(),cv[idx].contiguous()
                rows=torch.tensor([0]*n+[1]*n,device='cuda')
                kd,vd=torch.cat((pk[rows],ks),1),torch.cat((pv[rows],vs),1)
                singles=[(qs[[i,i+n]].contiguous(),kd[[i,i+n]].contiguous(),vd[[i,i+n]].contiguous()) for i in range(n)]
                def seq():return [sdpa(*x) for x in singles]
                seqouts=call(seq);seqfull=torch.cat([x[:1] for x in seqouts]+[x[1:] for x in seqouts])
                dense=lambda:sdpa(qs,kd,vd)
                actual=call(dense)
                paths={'sequential_B2_core':seq,'batched_core':dense,'batched_with_history_pack':lambda:sdpa(qs,torch.cat((pk[rows],ks),1),torch.cat((pv[rows],vs),1))}
                shared_compare=None
                if n==7:
                    shared=lambda:native.shared(qs,fullpk,fullpv,ks,vs,slots,rows)
                    shared_compare=numbers(call(shared),actual);paths['existing_shared_core']=shared
                times={}
                # Reverse arm order on alternate layer; all samples retained.
                names=list(paths)
                if layer==8:names.reverse()
                for name in names:times[name]=measure(paths[name])
                linear_input=base[idx].contiguous();linear=lambda:torch.nn.functional.linear(linear_input,weight,bias)
                lin=call(linear);linparts=[call(lambda i=i:torch.nn.functional.linear(linear_input[[i,i+n]],weight,bias)) for i in range(n)]
                linseq=torch.cat([x[:1] for x in linparts]+[x[1:] for x in linparts])
                linear_time=measure(linear)
                row=dict(layer=layer,n_v=n,CFG=2,Q=16,KV_total=2160,history=2144,q_shape=list(qs.shape),q_stride=list(qs.stride()),dtype=str(qs.dtype),attention_each_head_MNK=[16,2160,128],projection_MNK=[2*n*16,3072,3072],projection_weight_source=index[f'blocks.{layer}.attn1.to_out.0.weight'],projection_input='actual frozen original attention output',frozen_output=frozen,seq_vs_batched=numbers(seqfull,actual),subset_vs_full=numbers(actual,ref[idx].contiguous()),existing_shared=shared_compare,timing=times,projection=linear_time,projection_seq_vs_batch=numbers(linseq,lin))
                result['rows'].append(row)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
                    with torch.profiler.record_function('REAL_ATTENTION'):call(dense)
                    with torch.profiler.record_function('REAL_OUT_PROJECTION'):call(linear)
                    torch.cuda.synchronize()
                events=[dict(name=e.key,count=e.count,input_shapes=e.input_shapes) for e in prof.key_averages(group_by_input_shape=True) if e.key in ('aten::linear','aten::addmm','aten::mm','aten::matmul','aten::bmm','aten::scaled_dot_product_attention','aten::_scaled_dot_product_flash_attention')]
                kernels=sorted({e.name for e in prof.events() if str(e.device_type).endswith('CUDA')})
                result['profiles'].append(dict(layer=layer,n_v=n,aten=events,kernels=kernels))
                if layer==0 and n==7:prof.export_chrome_trace(str(out/'PROFILE-layer00-n7.json'))
                print(json.dumps(dict(layer=layer,n_v=n,done=True)),flush=True)
            assert versions==[x._version for x in (q,ck,cv,pk,pv,fullpk,fullpv,slots,base,weight,bias)]
        result.update(status='COMPLETE_LONG_FIXTURE_MECHANISM_ONLY',CPU_RNG_unchanged=torch.equal(rng,torch.get_rng_state()),CUDA_RNG_unchanged=torch.equal(crng,torch.cuda.get_rng_state()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_sec=time.monotonic()-start,input_versions_unchanged=True,short_medium_coverage=False)
    except BaseException as e:
        result.update(status='FAILED_REPLAY',error=repr(e),traceback=traceback.format_exc(),elapsed_sec=time.monotonic()-start)
        raise
    finally:
        write(out/'REPLAY-RESULT.json',result)

if __name__=='__main__':main()
