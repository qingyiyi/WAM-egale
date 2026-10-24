"""Actual B14 complete-block comparison and fused-stage attribution, all eager.

Normal whole-block timing decides latency. Embedded globaltimer intervals and
standalone Torch eager regions are diagnostic boundaries, never additive.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime
import gc
import json
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture, decode, visible_snapshot, tensors
from tools.metrics import compare, state_equal, byte_equal
from tools.time_whole_layer import capture, device_tree, load_extension, resource
from tools.cached_cross import cached_cross
from wam_ops.optimized.whole_layer import WholeLayerBlock
import wam_ops.optimized.prepared as prepared_module

P = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--process', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--stages', action='store_true')
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--label',required=True)
    ap.add_argument('--module-name', required=True)
    ap.add_argument('--profile', type=Path)
    ap.add_argument('--profile-module-name', default='wam_whole_warp_specialized_profile_cuda')
    ap.add_argument('--reference', type=Path, default=P/'A.so')
    ap.add_argument('--reference-module-name', default='wam_whole_warp_specialized_cuda')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    job = json.loads((P/'JOB.json').read_text())
    deadline = datetime.fromisoformat(job['absolute_cleanup_deadline']).timestamp()-90
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])
    start = time.time()
    result = dict(status='RUNNING', process=args.process, rows=[], calls=0,
                  graph=False, scope='three frozen B14 block0 fixtures; warm crossKV',
                  source=str(P), candidate=args.label, registered_job=job, before=resource(), stage_boundary=
                  'own embedded globaltimer incl barriers; Torch standalone eager CUDA events')

    def save():
        (args.output/'RESULT.json').write_text(json.dumps(result, indent=2))

    def tick():
        result['calls'] += 1
        if time.time() > deadline or result['calls'] > 6000:
            raise TimeoutError('registered wall/call limit')

    def timed(op, count=5):
        b, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall = time.perf_counter()
        b.record()
        for _ in range(count):
            y = op()
        e.record()
        e.synchronize()
        return dict(event_ms=b.elapsed_time(e)/count,
                    wall_ms=(time.perf_counter()-wall)*1000/count), y

    try:
        if result['before']['processes']:
            raise RuntimeError('visible GPU context already exists')
        normal_path = args.candidate
        referenceA=load_extension(args.reference,args.reference_module_name)
        guard_path = next((P.parent.parent/'eager-stage-comparison/build/guarded').glob('*.so'))
        normal = load_extension(normal_path, args.module_name)
        if normal is referenceA or normal.__name__ == referenceA.__name__:
            raise RuntimeError('candidate and A must have independent extension identities')
        result['loaded_modules'] = {k:dict(name=v.__name__, file=v.__file__)
                                    for k,v in [('A',referenceA),('candidate',normal)]}
        guard = load_extension(guard_path, 'wam_candidate_cuda')
        prof = load_extension(args.profile or next((P/'build/profile').glob('*.so')),
                              args.profile_module_name) if args.stages else None
        if prof is not None:
            result['loaded_modules']['profile'] = dict(name=prof.__name__, file=prof.__file__)
        old_extension = prepared_module.extension
        def selected_extension(name):
            if name != 'wam_candidate_cuda':
                raise ValueError(name)
            return guard
        prepared_module.extension = selected_extension
        block = load_block()
        bind = WholeLayerBlock(block)
        torch.cuda.synchronize()
        result['after_load'] = resource()
        owner = result['after_load']['processes'].splitlines()
        if len(owner) != 1:
            raise RuntimeError('expected one CUDA owner')
        result['properties'] = dict(torch=torch.__version__, cuda=torch.version.cuda,
                                    name=torch.cuda.get_device_name(),
                                    multiprocessors=torch.cuda.get_device_properties(0).multi_processor_count)
        with torch.inference_mode():
            for length in ['short', 'middle', 'long']:
                before = resource()
                if [x.rsplit(',',1)[0] for x in before['processes'].splitlines()] != [x.rsplit(',',1)[0] for x in owner]:
                    raise RuntimeError('visible CUDA context changed')
                path = FIXTURES/f'input-{length}-B14-action-layer0-first.pt'
                f = BlockFixture(path, device_tree(decode(torch.load(
                    path, map_location='cpu', mmap=True, weights_only=False))))
                a, kw = f.arguments()
                cache = f.install_cache(block)
                initial = visible_snapshot(cache)
                inputs = {k:v.clone() for k,v in tensors((a,kw))}
                rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
                with capture(block) as ref:
                    ref['output'] = block(*a, **kw)
                    tick()
                # Actual public cache builder; never seed it from golden tensors.
                bind.prepare_context(a[1], precompute=True)
                fullkv = [v.index_select(0, bind.context_rows) for v in bind.context_kv]
                cache_exact = {key:torch.equal(ref[key],v) for key,v in zip(
                    ['cross_k_linear','cross_k','cross_v'],fullkv)}
                if not all(cache_exact.values()):
                    raise RuntimeError('public cache differs from canonical B14')
                cl = [cache[k] for k in ['k','v','id','mask','is_pred']]
                common = (bind.weights,*a,cl,kw['candidate_cache_rows'],0,bind.eps)
                def own(lib=normal, debug=False):
                    tick()
                    return lib.forward(*common,debug,bind.context_rows,bind.context_roots,bind.context_kv)
                debug = own(debug=True)
                missing = set(ref) - set(debug)
                if missing:
                    raise RuntimeError(f'candidate missing captured tensors: {sorted(missing)}')
                exact = {k:compare(ref[k].reshape(-1),debug[k].reshape(-1))
                         for k in ref if k in debug}
                splitk_commit = debug['ffn_splitk_commit'].tolist()
                splitk_state = debug['ffn_pipeline_state'].tolist()
                if not (len(splitk_commit)==96 and all(v==7 for v in splitk_commit) and splitk_state[4:8]==[168]*4 and splitk_state[8:12]==[24]*4):
                    raise RuntimeError('split-K full contribution/count gate failed')
                tma_counts = debug['projection_tma_counts'].tolist()
                if tma_counts != [[[48,48,10,48] for _ in range(96)] for _ in range(6)]:
                    raise RuntimeError('continuous TMA count gate failed')
                role_counts = debug['projection_role_counts'].tolist()
                if role_counts != [[([[48,48,96]]+[[10,10,10]]*3+[[48,48,10]]*4) for _ in range(96)] for _ in range(6)]:
                    raise RuntimeError('six projections warp-role count gate failed')
                row = dict(projection_role_count_pass=True, projection_tma_counts=tma_counts,ffn_splitk_commit=splitk_commit,ffn_pipeline_state=splitk_state,case=length, total_KV=int(initial['slots'].numel())+16,
                           precision=exact, public_cache=cache_exact,
                           samples=[], resources=debug['resources'].tolist(), before=before)
                result['rows'].append(row)
                save()
                if not all(v['bitwise_pass'] for v in exact.values()):
                    raise RuntimeError('normal numerical gate failed')
                prepared = prepared_module.PreparedBlock(block,a,kw,cache)
                with prepared.activate('dense',False), cached_cross(block,fullkv):
                    def torch_op():
                        tick(); prepared.engine='dense'
                        return block(*a,**kw)
                    def guarded_op():
                        tick(); prepared.engine='guarded'
                        return block(*a,**kw)
                    ops = {'Torch':torch_op,'Guarded':guarded_op,
                           'Own':lambda:own()['output'],'A':lambda:own(referenceA)['output']}
                    row['whole_precision'] = {label:compare(ref['output'],op())
                                              for label,op in ops.items()}
                    if not all(v['bitwise_pass'] for v in row['whole_precision'].values()):
                        raise RuntimeError('four-arm whole-block numerical gate failed')
                    for _ in range(12):
                        for op in ops.values(): op()
                    labels = list(ops)
                    for repeat in range(16):
                        order = labels if (repeat+args.process)%2==0 else labels[::-1]
                        for label in order:
                            measured,y = timed(ops[label])
                            row['samples'].append(dict(rep=repeat,arm=label,**measured))
                    row['median_ms'] = {label:statistics.median(x['event_ms'] for x in row['samples'] if x['arm']==label) for label in ops}
                    row['post_timing_precision'] = {label:compare(ref['output'],op()) for label,op in ops.items()}
                    if not all(v['bitwise_pass'] for v in row['post_timing_precision'].values()):
                        raise RuntimeError('post-timing output changed')

                if args.stages:
                    profile_y = own(prof)
                    row['instrumented_precision'] = compare(ref['output'],profile_y['output'])
                    if not row['instrumented_precision']['bitwise_pass']:
                        raise RuntimeError('instrumentation changes output')
                    row['profile_resources'] = own(prof,True)['resources'].tolist()
                    row['regions_ms'], row['instrumentation_samples'] = [], []
                    for _ in range(15): own(); own(prof)
                    for repeat in range(16):
                        for label,lib in ([('normal',normal),('profile',prof)] if repeat%2==0 else [('profile',prof),('normal',normal)]):
                            measured,y = timed(lambda:own(lib),5)
                            row['instrumentation_samples'].append(dict(rep=repeat,arm=label,**measured))
                            if label=='profile':
                                ticks = y['stage_ticks'].cpu()
                                spans = (ticks[:,:,1].max(1).values-ticks[:,:,0].min(1).values).double()/1e6
                                row['regions_ms'].append(spans.tolist())

                    table = block.scale_shift_table[None]+a[2].float()
                    shift,scale,gate,cshift,cscale,cgate = [x.squeeze(1) for x in table.permute(0,2,1,3).chunk(6,1)]
                    def rope(x):
                        x=x.unflatten(2,(24,128))
                        z=torch.view_as_complex(x.double().reshape(*x.shape[:3],-1,2))
                        return torch.view_as_real(z*a[3]).flatten(3).to(x.dtype)
                    def dense_self():
                        hk=cache['k'][:,initial['slots']].index_select(0,kw['candidate_cache_rows'])
                        hv=cache['v'][:,initial['slots']].index_select(0,kw['candidate_cache_rows'])
                        return block.attn1.attn_op(ref['self_q_rope'],torch.cat((hk,ref['self_k_rope']),1),torch.cat((hv,ref['self_v'].unflatten(2,(24,128))),1))
                    def guarded_self():
                        return guard.grouped(ref['self_q_rope'],cache['k'],cache['v'],
                            ref['self_k_rope'],ref['self_v'].unflatten(2,(24,128)),
                            initial['slots'],prepared_module.guarded_group(initial['slots'].numel()))
                    operations = {
                        'table':lambda:block.scale_shift_table[None]+a[2].float(),
                        1:lambda:(block.norm1(a[0].float())*(1+scale)+shift).to(a[0].dtype),
                        2:lambda:(block.attn1.to_q(ref['modulated1']),block.attn1.to_k(ref['modulated1']),block.attn1.to_v(ref['modulated1'])),
                        3:lambda:rope(block.attn1.norm_q(ref['self_q_linear'])),
                        4:lambda:rope(block.attn1.norm_k(ref['self_k_linear'])),
                        5:dense_self,
                        6:lambda:block.attn1.to_out[0](ref['self_attention'].flatten(2)),
                        7:lambda:(a[0].float()+ref['self_out']*gate).to(a[0].dtype),
                        8:lambda:block.norm2(ref['residual1'].float()).to(a[0].dtype),
                        9:lambda:block.attn2.to_q(ref['norm2']),
                        10:lambda:block.attn2.norm_q(ref['cross_q_linear']),
                        11:lambda:block.attn2.attn_op(ref['cross_q'].unflatten(2,(24,128)),fullkv[1].unflatten(2,(24,128)),fullkv[2].unflatten(2,(24,128))),
                        12:lambda:block.attn2.to_out[0](ref['cross_attention'].flatten(2)),
                        13:lambda:ref['residual1']+ref['cross_out'],
                        14:lambda:(block.norm3(ref['residual2'].float())*(1+cscale)+cshift).to(a[0].dtype),
                        15:lambda:(ref['residual2'].float()+block.ffn(ref['modulated3']).float()*cgate).to(a[0].dtype),
                        'guarded_self':guarded_self,
                        'cold_crossKV':lambda:(lambda k:(k,block.attn2.norm_k(k),block.attn2.to_v(a[1])))(block.attn2.to_k(a[1])),
                    }
                    row['torch_stages'] = []
                    for sid,op in operations.items():
                        for _ in range(10):op()
                        samples=[]
                        for repeat in range(16):
                            measured,y=timed(op,20)
                            samples.append(measured)
                        row['torch_stages'].append(dict(region=sid,samples=samples,
                            median_ms=statistics.median(x['event_ms'] for x in samples)))
                        if time.time()>deadline:raise TimeoutError('stage deadline')

                row['valid_KV'] = state_equal(initial,visible_snapshot(cache))
                row['inputs_unchanged'] = all(byte_equal(inputs[k],v) for k,v in tensors((a,kw)))
                row['RNG_unchanged'] = byte_equal(rng[0],torch.get_rng_state()) and byte_equal(rng[1],torch.cuda.get_rng_state())
                row['after'] = resource()
                if not row['valid_KV']['pass_all'] or not row['inputs_unchanged'] or not row['RNG_unchanged']:
                    raise RuntimeError('state changed')
                if [x.rsplit(',',1)[0] for x in row['after']['processes'].splitlines()] != [x.rsplit(',',1)[0] for x in owner]:
                    raise RuntimeError('visible CUDA context changed')
                save()
                del debug,ref,f,a,kw,cache,cl,initial,inputs,fullkv,prepared
                block.attn1.attn_caches={};bind.context_kv=[]
                gc.collect();torch.cuda.empty_cache()
        result['status']='COMPLETE_EXACT_EAGER'
    except BaseException as exc:
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result['lifecycle_seconds']=time.time()-start
        save()
    print(result['status'],result.get('error',''),flush=True)
    if result['status']!='COMPLETE_EXACT_EAGER':raise SystemExit(1)


if __name__=='__main__':main()
