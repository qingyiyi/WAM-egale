"""Diagnostic-only partial comparison and within-CTA intervals, no robot claims."""
from pathlib import Path
import os,json,time,traceback,importlib.util
import torch
from tools.block_loader import FIXTURES,load_block
from tools.fixtures import BlockFixture,decode,visible_snapshot
from tools.replay import device_tree
from tools.metrics import compare,state_equal
from wam_ops.optimized.whole_layer import WholeLayerBlock
B=Path(os.environ['WAM_RUN_DIR']);R=B.parent;start=time.time()
os.sched_setaffinity(0,{4,5});torch.set_num_threads(2);torch.set_num_interop_threads(1)
def load(name,path):
    s=importlib.util.spec_from_file_location(name+'.wam_whole_fa2_cuda',path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
so='wam_whole_fa2_cuda.cpython-312-x86_64-linux-gnu.so'
r=dict(status='RUNNING',calls=0,rows=[],scope='software intervals from warp0 and MMA warp3 representatives, not all-warp critical path or GPU hardware counters')
def save():(B/'DIAGNOSTIC.json').write_text(json.dumps(r,indent=2)+'\n')
try:
    original=load('original',R/'A'/so);ad=load('a_diagnostic',R/'T2-A-diag-build'/so)
    bc=load('bc_diagnostic',R/'T2-BC-diag-build'/so)
    block=load_block('cuda');bind=WholeLayerBlock(block)
    with torch.inference_mode():
        for length in ['short','middle','long']:
            p=FIXTURES/f'input-{length}-B14-action-layer0-first.pt'
            f=BlockFixture(p,device_tree(decode(torch.load(p,map_location='cpu',mmap=True,weights_only=False))))
            a,k=f.arguments();f.install_cache(block);bind.prepare_context(a[1],precompute=True)
            cache=block.attn1.attn_caches[k['cache_name']];snapshot=visible_snapshot(cache)
            values=[cache[n] for n in ('k','v','id','mask','is_pred')]
            inputs=[x.clone() for x in a];cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state().clone()
            def call(ext,flag=None):
                r['calls']+=1
                z=ext.forward(bind.weights,*a,values,k['candidate_cache_rows'],0,bind.eps,True,bind.context_rows,bind.context_roots,bind.context_kv,*([] if flag is None else [flag]))
                torch.cuda.synchronize();return z
            anchor=call(original);base=call(ad)
            checks={key:compare(anchor[key].reshape(-1),base[key].reshape(-1)) for key in anchor.keys()&base.keys() if key not in ['resources','ffn_pipeline_timestamps','ffn_pipeline_state']}
            assert all(x['bitwise_pass'] for x in checks.values())
            row=dict(case=length,anchor_diagnostic_checks=checks,resources_A=anchor['resources'].tolist(),resources_A_diag=base['resources'].tolist(),samples=[])
            for repeat in range(6):
                for arm,flag in [('B',False),('C',True)]:
                    z=call(bc,flag)
                    checks={key:compare(base[key].reshape(-1),z[key].reshape(-1)) for key in ['first_partial','ffn2','output'] if key in base}
                    assert 'first_partial' in checks and all(x['bitwise_pass'] for x in checks.values())
                    t=z['local_trace'].tolist();assert len(t)==96 and all(min(v)>0 for v in t)
                    row['samples'].append(dict(repeat=repeat,arm=arm,checks=checks,resources=z['resources'].tolist(),trace_ns=t))
            row['cache']=state_equal(snapshot,visible_snapshot(cache));row['inputs_unchanged']=all(torch.equal(x,y) for x,y in zip(inputs,a))
            row['rng_unchanged']=torch.equal(cpu,torch.get_rng_state()) and torch.equal(gpu,torch.cuda.get_rng_state())
            assert row['cache']['pass_all'] and row['inputs_unchanged'] and row['rng_unchanged']
            r['rows'].append(row);save()
    r['status']='COMPLETE_FINITE_PARTIAL_AND_LOCAL_INTERVALS'
except BaseException as e:r.update(status='FAILED',error=repr(e),traceback=traceback.format_exc())
finally:r['lifecycle_seconds']=time.time()-start;save()
print(r['status'],r.get('error',''),flush=True)
if r['status'].startswith('FAILED'):raise SystemExit(1)
