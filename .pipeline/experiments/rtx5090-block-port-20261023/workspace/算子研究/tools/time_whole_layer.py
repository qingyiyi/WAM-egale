"""Real complete-layer source entry: output/state comparison and paired timing.

This is the requested measurement, not a smoke run. No synthetic input, SHA,
hardware counters, model training, simulator, or production permission change.
"""
import argparse, gc, importlib.util, json, os, statistics, subprocess, time, traceback
from contextlib import contextmanager
from pathlib import Path
import torch
from tools.block_loader import ROOT, FIXTURES, load_block
from tools.fixtures import BlockFixture, decode, visible_snapshot, tensors
from tools.metrics import compare, state_equal, byte_equal
from tools.cached_cross import cached_cross
from wam_ops.optimized.prepared import PreparedBlock
from wam_ops.optimized.whole_layer import WholeLayerBlock

def device_tree(x):
    if isinstance(x, torch.Tensor): return x.cuda()
    if isinstance(x, dict): return {k:device_tree(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return type(x)(device_tree(v) for v in x)
    return x

def resource():
    def query(fields):
        return subprocess.check_output(['nvidia-smi', fields, '--format=csv,noheader,nounits'],text=True,timeout=15).strip()
    return dict(gpu=query('--query-gpu=uuid,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used'),
                processes=query('--query-compute-apps=gpu_uuid,pid,used_gpu_memory'))

@contextmanager
def capture(block):
    """Observe the original Torch modules, outside every timing region."""
    values, handles = {}, []
    modules = dict(block.named_modules())
    outputs = {'norm1':'norm1','norm3':'norm3','attn1.to_q':'self_q_linear',
        'attn1.to_k':'self_k_linear','attn1.to_v':'self_v','attn1.norm_q':'self_q_norm',
        'attn1.norm_k':'self_k_norm','attn1.to_out.0':'self_out','attn2.to_q':'cross_q_linear',
        'attn2.to_k':'cross_k_linear','attn2.to_v':'cross_v','attn2.norm_q':'cross_q',
        'attn2.norm_k':'cross_k','attn2.to_out.0':'cross_out',
        'ffn.net.0.proj':'ffn1','ffn.net.0':'gelu','ffn.net.2':'ffn2'}
    for name,key in outputs.items():
        handles.append(modules[name].register_forward_hook(
            lambda m,a,y,key=key: values.__setitem__(key,y.detach().clone())))
    for name,key in [('attn1.to_q','modulated1'),('attn2.to_q','norm2'),('ffn.net.0.proj','modulated3')]:
        handles.append(modules[name].register_forward_pre_hook(
            lambda m,a,key=key: values.__setitem__(key,a[0].detach().clone())))
    for name,key in [('norm2','residual1'),('norm3','residual2')]:
        handles.append(modules[name].register_forward_pre_hook(
            lambda m,a,key=key: values.__setitem__(key,a[0].detach().to(torch.bfloat16).clone())))
    original = [block.attn1.attn_op,block.attn2.attn_op]
    def attention(i,q,k,v):
        y=original[i](q,k,v)
        values['self_attention' if i==0 else 'cross_attention']=y.detach().clone()
        if i==0:
            values['self_q_rope']=q.detach().clone()
            values['self_k_rope']=k[:,-q.shape[1]:].detach().clone()
        return y
    block.attn1.attn_op=lambda q,k,v:attention(0,q,k,v)
    block.attn2.attn_op=lambda q,k,v:attention(1,q,k,v)
    try: yield values
    finally:
        for h in handles:h.remove()
        block.attn1.attn_op,block.attn2.attn_op=original

def load_extension(path,name='wam_whole_fa2_cuda'):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--binary',type=Path,required=True);p.add_argument('--process',type=int,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    job=json.loads((ROOT.parent/'JOB.json').read_text())
    from datetime import datetime
    deadline=datetime.fromisoformat(job['absolute_cleanup_deadline']).timestamp()-90
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:4])
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    start=time.time();result=dict(status='RUNNING',rows=[],calls=0,process=a.process,
        gpu_scope='one complete real B14 block0, warm final crossKV cache',
        baseline='original Torch class bodies; static dense history admission for both eager and Graph',
        source=str(ROOT),cache_build='canonical B14 projection outputs deduplicated to two conditions',binary=str(a.binary),sha='NOT_COMPUTED_USER_NO_SHA',
        formal_performance_permission=False,utilization_cause='UNKNOWN',resources=[resource()])
    def save(): (a.output/'RESULT.json').write_text(json.dumps(result,indent=2))
    def tick():
        result['calls']+=1
        if result['calls']>10000 or time.time()>deadline:raise TimeoutError('registered forward/time budget')
    try:
        if result['resources'][0]['processes']:raise RuntimeError('visible GPU already occupied')
        ext=load_extension(a.binary);block=load_block();bind=WholeLayerBlock(block)
        torch.cuda.synchronize();result['resources'].append(resource())
        owner=result['resources'][-1]['processes'].splitlines()
        if len(owner)!=1:raise RuntimeError('cannot identify one newly created CUDA context')
        result['properties']=dict(torch=torch.__version__,cuda=torch.version.cuda,
            name=torch.cuda.get_device_name(),sm_count=torch.cuda.get_device_properties(0).multi_processor_count,
            context_observation=owner)
        stream=torch.cuda.Stream()
        with torch.inference_mode(),torch.cuda.stream(stream):
            for length in ['short','middle','long']:
                current=resource();result['resources'].append(current)
                if [x.rsplit(',',1)[0] for x in current['processes'].splitlines()] != [x.rsplit(',',1)[0] for x in owner]:
                    raise RuntimeError('visible context set changed')
                path=FIXTURES/f'input-{length}-B14-action-layer0-first.pt'
                f=BlockFixture(path,device_tree(decode(torch.load(path,map_location='cpu',mmap=True,weights_only=False))))
                args,kw=f.arguments();cache=f.install_cache(block);initial=visible_snapshot(cache)
                immutables={k:t.clone() for k,t in tensors((args,kw))}
                cpu_rng,gpu_rng=torch.get_rng_state().clone(),torch.cuda.get_rng_state().clone()
                with capture(block) as refs:
                    ref=block(*args,**kw);tick()
                refs['output']=ref
                bind.prepare_context(args[1],precompute=True)
                # Preserve the current canonical B14 GEMM rounding when building
                # the warm cache. Deduplicate its *outputs*, not its input batch.
                bind.context_kv=[refs[key].index_select(0,bind.context_roots) for key in
                    ['cross_k_linear','cross_k','cross_v']]
                for key,value in zip(['cross_k_linear','cross_k','cross_v'],bind.context_kv):
                    if not torch.equal(refs[key],value.index_select(0,bind.context_rows)):
                        raise RuntimeError('canonical B14 context outputs cannot be deduplicated exactly')
                full_kv=[v.index_select(0,bind.context_rows) for v in bind.context_kv]
                cl=[cache[k] for k in ['k','v','id','mask','is_pred']]
                common=(bind.weights,*args,cl,kw['candidate_cache_rows'],kw.get('update_cache',0),bind.eps)
                def own(debug=False):return ext.forward(*common,debug,bind.context_rows,bind.context_roots,bind.context_kv)
                values=own(True);tick();torch.cuda.synchronize()
                numeric={k:compare(refs[k].reshape(-1),values[k].reshape(-1)) for k in refs if k in values}
                row=dict(case=length,total_KV=int(initial['slots'].numel())+16,fixture=str(path),
                    precision=numeric,all_shared_outputs_exact=all(v['bitwise_pass'] for v in numeric.values()),
                    resources=values['resources'].tolist(),ffn_pipeline_state=values['ffn_pipeline_state'].tolist(),
                    historical_A100_output=compare(f.data['block_gold'],ref),own_vs_historical_A100=compare(f.data['block_gold'],values['output']),samples=[],before=current,
                    context_rows=bind.context_rows.tolist(),context_roots=bind.context_roots.tolist())
                result['rows'].append(row);save()
                if not all(v['finite_pass'] for v in numeric.values()):raise RuntimeError('nonfinite current output')
                prepared=PreparedBlock(block,args,kw,cache)
                with prepared.activate('dense',False),cached_cross(block,full_kv):
                    baseline=block(*args,**kw);tick()
                    row['prepared_baseline_vs_canonical']=compare(ref,baseline)
                    normal=own(False)['output'];tick()
                    row['normal_vs_debug']=compare(values['output'],normal)
                    row['timing_qualification']='FINITE_EXACT' if (row['all_shared_outputs_exact'] and
                        row['prepared_baseline_vs_canonical']['bitwise_pass'] and row['normal_vs_debug']['bitwise_pass']) else 'DIAGNOSTIC_NUMERIC_MISMATCH_NOT_ADMITTED'
                    ops={'Torch':lambda:block(*args,**kw),'Own':lambda:own(False)['output']}
                    graphs={};graph_out={}
                    for name,op in ops.items():
                        for _ in range(3):op();tick()
                        g=torch.cuda.CUDAGraph()
                        with torch.cuda.graph(g,stream=stream):graph_out[name]=op();tick()
                        graphs[name]=g
                    for mode in ['eager','graph']:
                        for _ in range(60):
                            for name,op in ops.items():
                                op() if mode=='eager' else graphs[name].replay();tick()
                        for rep in range(24):
                            order=['Torch','Own'] if (rep+a.process)%2==0 else ['Own','Torch']
                            for name in order:
                                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                                torch.cuda.synchronize();wall=time.perf_counter();begin.record(stream)
                                for _ in range(10):
                                    y=ops[name]() if mode=='eager' else graphs[name].replay();tick()
                                end.record(stream);end.synchronize()
                                row['samples'].append(dict(mode=mode,rep=rep,arm=name,
                                    event_ms=begin.elapsed_time(end)/10,wall_ms=(time.perf_counter()-wall)*100))
                        save()
                    row['graph_output']={name:compare(ref,y) for name,y in graph_out.items()}
                row['valid_KV']=state_equal(initial,visible_snapshot(cache))
                row['inputs_unchanged']=all(byte_equal(immutables[k],t) for k,t in tensors((args,kw)))
                row['RNG_unchanged']=byte_equal(cpu_rng,torch.get_rng_state()) and byte_equal(gpu_rng,torch.cuda.get_rng_state())
                row['after']=resource()
                row['median_ms']={mode:{arm:statistics.median(s['event_ms'] for s in row['samples'] if s['mode']==mode and s['arm']==arm) for arm in ['Torch','Own']} for mode in ['eager','graph']}
                save()
                del values,refs,ref,normal,baseline,graphs,graph_out,ops,f,args,kw,cache,cl,initial,immutables,prepared
                block.attn1.attn_caches={};bind.context_kv=[];gc.collect();torch.cuda.empty_cache()
        result['status']='COMPLETE_EXACT' if all(r['timing_qualification']=='FINITE_EXACT' and r['valid_KV']['pass_all'] and r['inputs_unchanged'] and r['RNG_unchanged'] for r in result['rows']) else 'COMPLETE_DIAGNOSTIC_NOT_QUALIFIED'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result['lifecycle_seconds']=time.time()-start;save()
    print(result['status'],result.get('error',''),flush=True)
    if result['status']=='FAILED_OR_PARTIAL':raise SystemExit(1)

if __name__=='__main__':main()
