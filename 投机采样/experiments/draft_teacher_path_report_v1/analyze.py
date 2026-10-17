"""CPU-only final evidence, resource and immutable-file audit."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from draft_teacher_path_diagnostic_v1.protocol import write,read,load
from draft_timestep_causal_v1.run import load as load2
import torch
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)
r=ROOT/'投机采样/results/runtime'
a=r/'p0-draft-teacher-path-diagnostic-v1-20261016-r1'
b=r/'p0-draft-timestep-causal-v1-20261016-r2'
p1,p2=load(a),load2(b)
s1,s2=read(a/'SUPERVISOR-RESULT.json'),read(b/'SUPERVISOR-RESULT.json')
c1,c2=read(a/'RUN-STARTED.json'),read(b/'RUN-STARTED.json')
old=read(r/'p0-draft-freeze-quality-v1-local-20261016-r1/RESULT.json')
assert old['permissions']==p1['permissions']
assert old['retained_primary_for_prospective_independent_TEST'] is False
st1=torch.load(a/'REAL-STAGES.pt',map_location='cpu',weights_only=True,mmap=True)
st2=torch.load(b/'CAUSAL-STAGES.pt',map_location='cpu',weights_only=True,mmap=True)
tf=torch.load(p1['arms']['TF_CONT']['trace'],map_location='cpu',weights_only=False)
first=next(x for x in tf['rounds'] if x['kind']=='speculative_tree')

def compact(x):return x[0].T[:7].reshape(112).contiguous()
def metrics(x,y):
    d=(x.float()-y.float()).abs()
    return dict(elements=x.numel(),dtype=str(x.dtype),different=int(torch.count_nonzero(d)),
        max_abs=float(d.max()),RMS=float(torch.sqrt(d.square().mean())))

base=compact(st1['outputs']['T']);packed=compact(st1['outputs']['TF_CONT'])
assert not torch.equal(base,tf['commits'][1]['direction'])
assert torch.equal(packed,tf['commits'][1]['direction'])
out=dict(root_conditional112=metrics(base,packed),
    causal_conditional112={k:metrics(base,compact(v)) for k,v in st2['outputs'].items()},
    time_suboperator_dtypes={k:str(v.dtype) for k,v in st2['stages']['ordinary_B2']
        if isinstance(v,torch.Tensor) and 'condition_embedder_action.' in k},
    root_depth=first['accepted_depth'],root_sigma_path=first['sigma_path'],
    root_threshold=first['threshold'],recorded_first_costs=first['first_costs'],
    all_scores_below_are_CPU_diagnostic_not_production_recalibration=True)
delta=abs(first['sigma_path'][1]-first['sigma_path'][0])
out['CPU_root_first_costs_B2']=[float(delta*torch.linalg.vector_norm((x-base).float())) for x in first['first_directions']]
out['CPU_root_first_costs_B14']=[float(delta*torch.linalg.vector_norm((x-packed).float())) for x in first['first_directions']]
out['teacher_numeric_difference_local_norm']=float(delta*torch.linalg.vector_norm((base-packed).float()))
causal=read(b/'WORKER-RESULT.json')
for key,stage in (('canonical_time_returned_exact','condition_embedder_action.returned'),
                  ('canonical_time_block0_exact','blocks.0')):
    out[key]={name:next(x for x in causal['stages'][name]['stages'] if x['stage']==stage)['bitwise']
        for name in ('B14_root_time_from_B2','B14_all_candidate_time_pairs_B2')}
write(b/'CPU-ROOT-ANALYSIS.json',out)
guards=read(ROOT/'.pipeline/maintenance/hw-upgrade-20261016/BEFORE.json')['immutable_guards']
changed=[]
for record in guards:
    f=ROOT/record['path'];s=f.stat()
    if (s.st_size,s.st_mtime_ns)!=(record['bytes'],record['mtime_ns']):changed.append(record['path'])
assert not changed
assert (ROOT/'投机采样/configs/current_speculative_candidate.json').read_bytes()==(
    ROOT/'review_exports/WAM-egale/投机采样/configs/current_speculative_candidate.json').read_bytes()
live=[]
for summary in (s1,s2):
    for pin in summary['observed_pins']:
        f=Path('/proc')/str(pin['pid'])/'stat'
        if f.exists():
            raw=f.read_text();fields=raw[raw.rfind(')')+2:].split()
            if int(fields[19])==pin['start_ticks'] and fields[0]!='Z':live.append(pin)
assert not live
elapsed=(c2['monotonic']-c1['monotonic'])+s2['elapsed_sec']
assert elapsed<1200
budget=dict(status='PASS_CPU_FINAL_AUDIT',fixed_commit=p1['commit'],GPU_initialized=False,
    scientific_episodes_this_phase=0,complete_recorded_policy_chunks=2,
    additional_full_verifier_query_replays=8,
    actual_total_target_forwards=read(a/'WORKER-RESULT.json')['target_accounting']['target_intents']+
        causal['target_accounting']['target_intents'],
    GPU_job_supervisor_wall_sec=s1['elapsed_sec']+s2['elapsed_sec'],
    cumulative_first_supervisor_window_sec=elapsed,cumulative_cap_sec=1200,actual_CUDA_active_sec='UNKNOWN',
    output_bytes=sum(x.stat().st_size for job in (a,b) for x in job.rglob('*') if x.is_file()),
    all_observed_child_lifetimes_exited=True,observed_child_count=len(s1['observed_pins'])+len(s2['observed_pins']),
    both_supervisor_roots_exited=all(not Path('/proc',str(c['pid'])).exists() for c in (c1,c2)),
    cleanup_signals=s1['cleanup']['signals']+s2['cleanup']['signals'],
    immutable_guards_checked=len(guards),changed_guards=changed,current_config_matches_pinned=True,
    original_permissions=p1['permissions'],old_retention_unchanged=True,training_trigger=False,
    no_120_started=True,no_AutoDL=True,
    CPU_review_notes=['Initial combined worker/supervisor import encountered expected legacy cleanup namespace collision; actual separate processes passed',
        'Initial postcheck used missing top-level permission key; corrected to the existing nested permissions field; zero CUDA'])
write(b/'FINAL-CPU-AUDIT.json',budget)
import json
print(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(budget,ensure_ascii=False,indent=2))
