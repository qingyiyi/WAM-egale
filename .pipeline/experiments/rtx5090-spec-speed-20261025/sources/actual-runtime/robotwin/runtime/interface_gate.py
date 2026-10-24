"""Eight complete policy calls from two frozen real initial observations.

Original T versus wrapper T; native candidate versus Q32 shared candidate.
Not an environment episode or success-rate test. No synthetic/random fixture.
"""
from pathlib import Path
import os,json,time,random,traceback
os.sched_setaffinity(0,{20,21})
import numpy as np
import torch
from common import setup,R
B=Path(__file__).resolve().parent;E=B/'interface-gate';E.mkdir(exist_ok=False)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
state=dict(status='LOADING',start=time.time(),policy_calls=0,teacher_attempted=0,teacher_returned=0,comparisons=[],
           complete_episodes=0,formal_performance_permission=False)
def save():(E/'RESULT.json').write_text(json.dumps(state,indent=2)+'\n')
def snapshot(server):
    result=[]
    for block in server.transformer.blocks:
        c=block.attn1.attn_caches[server.cache_name];mask=c['mask'].clone();valid=mask.nonzero().squeeze(-1)
        result.append({k:(c[k][:,valid] if k in ['k','v'] else c[k]).cpu().clone() for k in ['k','v','id','mask','is_pred']})
    return result

def compare(a,b):
    if isinstance(a,torch.Tensor):return a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(compare(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(compare(x,y) for x,y in zip(a,b))
    return a==b

try:
    assert json.loads((B/'TRAIN-RESULT.json').read_text())['status']=='COMPLETE_PARENT2000_AND_CAL12_NOT_ROBOT_CERTIFICATE'
    assert json.loads((B/'Q32-BUILD-STATUS.json').read_text())['status']=='BUILT_Q32_NOT_VALIDATED'
    source,s=setup(E)
    base_forward=s.transformer.forward
    def counting():
        old=s.transformer.forward
        def call(*args,**kwargs):
            assert state['teacher_attempted']<1500
            state['teacher_attempted']+=1
            out=old(*args,**kwargs);state['teacher_returned']+=1
            return out
        s.transformer.forward=call
    counting()
    manifest=json.loads((R/'runs/target-first6-3/INPUT-MANIFEST.json').read_text())
    cases=[x for x in manifest if x['task'] in ['place_object_basket','handover_block']]
    assert len(cases)==2
    inputs=[]
    for case in cases:
        d=torch.load(case['initial'],map_location='cpu',weights_only=False);o=d['initial_observation']
        obs={'observation.images.cam_high':o['observation']['head_camera']['rgb'],
             'observation.images.cam_left_wrist':o['observation']['left_camera']['rgb'],
             'observation.images.cam_right_wrist':o['observation']['right_camera']['rgb'],
             'observation.state':o['joint_action']['vector'],'task':case['instruction']}
        inputs.append((case,obs))
    def run(case,obs,runtime=None,spec=False,shader='shared'):
        assert state['policy_calls']<8
        random.seed(0);np.random.seed(0);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        s.infer(dict(reset=True,prompt=case['instruction'],save_visualization=False))
        if runtime:runtime.shader=shader;runtime.begin_chunk(spec)
        state['policy_calls']+=1;state['status']='RUNNING_COMPLETE_POLICY_CALL';save();start=time.time()
        out=s.infer(dict(obs=obs,prompt=case['instruction'],save_visualization=False,video_guidance_scale=5,action_guidance_scale=1))
        torch.cuda.synchronize()
        result=dict(action=out['action'],KV=snapshot(s),CPU=torch.get_rng_state(),CUDA=torch.cuda.get_rng_state_all())
        if runtime:result['audit']=runtime.end_chunk()
        state.setdefault('calls',[]).append(dict(case=case['task'],spec=spec,shader=shader,elapsed=time.time()-start,
            teacher_attempted=state['teacher_attempted'],scheduler_updates=(result.get('audit',{}).get('counters',{}).get('committed_scheduler_update_count') if spec else 50)))
        save();return result
    original=[run(c,o) for c,o in inputs]
    s.transformer.forward=base_forward # remove the first observer before rebinding
    from runtime import Runtime
    tau=json.loads((B/'CALIBRATION.json').read_text())['tau']
    runtime=Runtime(s,source,B/'parent2000.pt',tau,E/'adapter');counting()
    for (case,obs),ref in zip(inputs,original):
        current=run(case,obs,runtime,False);same=compare(ref,{k:current[k] for k in ref})
        state['comparisons'].append(dict(case=case['task'],pair='canonical-T vs wrapper-T',all_output_effective_KV_RNG_exact=same));save();assert same
        native=run(case,obs,runtime,True,'native');shared=run(case,obs,runtime,True,'shared')
        keys=['action','KV','CPU','CUDA'];same=compare({k:native[k] for k in keys},{k:shared[k] for k in keys})
        tree_equal=native['audit']['trees']==shared['audit']['trees']
        counter_equal=native['audit']['counters']==shared['audit']['counters']
        state['comparisons'].append(dict(case=case['task'],pair='native-B14 vs Q32-shared-B14',all_output_effective_KV_RNG_exact=same,
            tree_scores_selection_exact=tree_equal,counters_equal=counter_equal));save();assert same and tree_equal and counter_equal
        torch.save(dict(case=case,action=shared['action'],audit=shared['audit']),E/(case['task']+'-observed.pt'))
    assert state['policy_calls']==8 and len(state['comparisons'])==4
    state.update(status='PASS_FINITE_EIGHT_COMPLETE_POLICY_CALLS_NOT_ROBOT_QUALITY',
        B2_B14_teacher_parity='NOT_ASSERTED',all_valid_KV_compared_live=True,full_KV_persisted=False,
        reset_bank_full_episode_equivalence='NOT_TESTED',initial_observations_only=True)
except BaseException as e:state.update(status='FAILED_OR_PARTIAL_INTERFACE',error=repr(e),traceback=traceback.format_exc())
finally:state['end']=time.time();state['elapsed']=time.time()-state['start'];save()
if state['status']!='PASS_FINITE_EIGHT_COMPLETE_POLICY_CALLS_NOT_ROBOT_QUALITY':raise SystemExit(1)
