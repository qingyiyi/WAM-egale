"""CPU preparation for one bounded real-recorded-observation Q02 replay."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

OWN=Path(__file__).resolve().parent
EXP=OWN.parent
ROOT=OWN.parents[2]
sys.path.insert(0,str(EXP))
RUNTIME=ROOT/'投机采样/results/runtime'
AUDIT=RUNTIME/'p0-wam-quality-diagnostic-v1-20261016-r1'
FRESH=RUNTIME/'p0-target-trace-draft-fresh-dev-v1-20261015-r2'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def require(v,s):
    if not v:raise ValueError(s)

def inputs():
    from target_trace_draft_fresh_dev_v1.protocol import load as original_load
    p=original_load(FRESH)
    d=read(RUNTIME/'p0-fresh-dev-terminal-diagnostic-v1-20261015-r2/DIAGNOSTIC.json')
    require(d['status']=='COMPLETE_INDEPENDENT_TERMINAL30_DIAGNOSTIC'
        and d['original_terminal_mode']=='ORIGINAL_FAILED_CAMERA_ONLY_PROVED'
        and d['full_observed_episodes']==30 and d['recorded_rule_violation_count']==0,
        'actual complete original failed-camera DEV evidence required')
    require(read(AUDIT/'RESULT.json')['status']=='COMPLETE_CPU_EVIDENCE_Q02_PREPARATION','actual CPU audit required')
    m=read(AUDIT/'Q02-INPUT-MANIFEST.json')
    require(m['task']==7 and m['init']==24 and m['arm']=='FINETUNE_TF64' and len(m['chunks'])==20,
            'only actual current TF task7 init24 input chain accepted')
    for x in m['chunks']:
        for s in x['files'].values():
            v=Path(s['path']).stat()
            require((v.st_size,v.st_mtime_ns)==(s['bytes'],s['mtime_ns']),'recorded input artifact changed')
    require(m['current_checkpoint']==p['selected']['transformer64'],'current TF checkpoint differs')
    return p,m

def query_selection():
    c=read(FRESH/'episodes/FINETUNE_TF64-t7-i24/COUNTER.json')
    out=[]
    for chunk in range(10):
        ts=[x for x in c['acceptance'] if x['context']['rollout_chunk']==chunk]
        bins=[('high',lambda x:x>=.6),('transition',lambda x:.3<=x<.6),('low',lambda x:x<.01)]
        if chunk in (0,9):bins.append(('mid',lambda x:.1<=x<.3))
        for name,predicate in bins:
            i=next(i for i,t in enumerate(ts) if predicate(t['sigma']))
            out.append(dict(chunk=chunk,tree_index=i,sigma_region=name,sigma=ts[i]['sigma']))
    require(len(out)==32 and len({(x['chunk'],x['tree_index']) for x in out})==32,'fixed32 distinct real tree selection')
    return out

def prepare(job):
    job=Path(job).resolve();require(job.parent==RUNTIME and not job.exists(),'fresh isolated runtime directory required')
    p,m=inputs();queries=query_selection()
    needed=[AUDIT/n for n in ('RESULT.json','EPISODE-AUDIT.json','SOURCE-EXPOSURE.json','Q02-INPUT-MANIFEST.json','RISK-ASSESSMENT.json')]
    needed += [FRESH/'plan.json',FRESH/'FAILED.json',FRESH/'FINAL-OWNED-CLEANUP.json',
        FRESH/'episodes/FINETUNE_TF64-t7-i24/COUNTER.json',FRESH/'episodes/FINETUNE_TF64-t7-i24/RESET.json',
        FRESH/'arms/FINETUNE_TF64/FULL-REAL-CAL-PARITY.json',FRESH/'arms/FINETUNE_TF64/TARGET-INITIAL-RNG.pt']
    own_files=[OWN/n for n in ('replay_protocol.py','replay_worker.py','run_replay.py','resource_probe.py')]
    require(all(x.is_file() for x in own_files),'complete worker/supervisor sources required before prepare')
    plan=dict(schema='wam-quality-real-input-replay/v1',job=str(job),source_job=str(FRESH),input_manifest=m,
        reference_plan=p,source_sha256={**p['source_sha256'],**{str(x):sha(x) for x in own_files}},
        prerequisite_sha256={str(x):sha(x) for x in needed},query_selection=queries,
        baseline_mode='original_dense_RAW_B14_requiring_exact_match_to_recorded_O3',
        baseline_chunks=10,comparison_chunks=10,total_full_policy_cache_chunk_cap=20,
        two_pass_rule='first10 complete baseline chunks must PASS before resetting and replaying same10 with probes',
        packed_query_cap=32,real_candidate_node_query_cap=224,extra_target_forward_cap=256,total_target_forward_cap=4000,
        primary_numeric_comparison='same-current-query denseB14 versus seven sequential denseB2; original gate unchanged',
        same_query_requires_original_action_video_and_acceptance_path_bitwise=True,
        root_cause_status='UNKNOWN',D01_allowed=False,
        max_run_wall_sec=2700,active_wall_sec=2400,cleanup_reserve_sec=300,
        initial_free_GiB=22.,runtime_free_GiB=.5,output_budget_GiB=2.,disk_floor_GiB=5.,
        CPU_threads=2,CPU_limit=7,seed=p['seed'],master_port=29617,
        no_EGL=True,no_new_sampling=True,no_training=True,no_smoke=True,performance_claim=False,
        no_per_query_KV_saved=True,no_AUTODL=True,no_automatic_retry=True,
        source_files_relocated=False,original_failed_status_modified=False)
    job.mkdir();(job/'queries').mkdir();write(job/'plan.json',plan);return plan

def load(job):
    job=Path(job).resolve();p=read(job/'plan.json')
    require(p['schema']=='wam-quality-real-input-replay/v1' and p['job']==str(job),'replay plan identity differs')
    require((p['baseline_chunks'],p['comparison_chunks'],p['total_full_policy_cache_chunk_cap'],p['packed_query_cap'],
        p['extra_target_forward_cap'],p['total_target_forward_cap'],p['max_run_wall_sec'],p['active_wall_sec'],p['cleanup_reserve_sec'])
        ==(10,10,20,32,256,4000,2700,2400,300),'fixed replay bounds changed')
    require(p['query_selection']==query_selection() and p['input_manifest']==inputs()[1],'fixed real input/query selection changed')
    for name in ('source_sha256','prerequisite_sha256'):
        for path,expected in p[name].items():require(sha(path)==expected,'frozen source/prerequisite changed:'+path)
    for s in p['reference_plan']['target_weight_stats']:
        x=Path(s['path']).stat();require((x.st_size,x.st_mtime_ns)==(s['bytes'],s['mtime_ns']),'target weight file metadata changed')
    return p

if __name__=='__main__':
    q=argparse.ArgumentParser();q.add_argument('--job',type=Path,required=True);q.add_argument('--prepare',action='store_true');a=q.parse_args()
    p=prepare(a.job) if a.prepare else load(a.job)
    print(json.dumps(dict(status='PREPARED_CPU_ONLY' if a.prepare else 'PASS_FROZEN_REPLAY_PLAN',
        job=p['job'],total_chunk_cap=20,packed_query_cap=32,real_node_queries_cap=224,GPU_used=False),ensure_ascii=False))
