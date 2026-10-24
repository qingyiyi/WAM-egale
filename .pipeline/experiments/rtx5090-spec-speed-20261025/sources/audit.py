"""CPU audit of returned real logs, never a synthetic model/simulator test."""
from pathlib import Path
import json,os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
import numpy as np
import torch
torch.set_num_threads(2)
P=Path(__file__).resolve().parent
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b)
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
report=dict(scope='Only persisted actual finite panel evidence; no new GPU calls',suites={},full_KV_and_all_outputs_equivalence='UNKNOWN_NOT_SAVED',B2_B14_teacher_equivalence='NOT_ESTABLISHED',permissions={k:False for k in ['engineering_quality_screen','independent_test_complete','noninferiority_established','formal_performance_permission']})
for suite in ['long','robotwin','goal','spatial','object']:
    r=P/'raw'/suite
    if not r.exists():continue
    result=dict(policies=0,committed_updates=0,route_errors=[],counter_errors=[],reset_RNG_pairs=[],first_two_S_decision_pairs=[],robotwin_first_48_submitted_command_pairs=[])
    policies={}
    for server in r.glob('server-*'):
        s=server/'STATUS.json'
        if not s.exists():continue
        state=json.loads(s.read_text())
        if suite!='robotwin' and state.get('single_worker_engine')!='existing best LIBERO noFSDP':continue
        f=server/'requests.jsonl'
        if not f.exists():continue
        for line in f.read_text().splitlines():
            if not line.strip():continue
            row=json.loads(line)
            if row.get('kind')!='policy' or not row.get('complete'):continue
            result['policies']+=1;result['committed_updates']+=row['valid_commits_g']
            expected='T' if row['arm']=='T' or row['arm']=='P' and row['ordinal']%3==2 else 'S'
            if row['actual_route']!=expected or row['scheduled']!=expected:result['route_errors'].append((row['case'],row['arm'],row['ordinal']))
            c=row['counters'];forwards=row['forwards'];bad=[]
            if row['valid_commits_g']!=50 or sum(x['g'] for x in row['rounds'])!=50:bad.append('legal_g')
            if expected=='T':
                if row['scheduler']!=50 or forwards.get('action_B2_update0',{}).get('returned')!=50:bad.append('canonical50')
            else:
                if sum(c.get(k,0) for k in ['ordinary_target_scheduler_update_count','committed_draft_scheduler_update_count','bonus_target_scheduler_update_count'])!=50:bad.append('commit_partition')
                n=c.get('tree_count',0)
                if c.get('logical_candidate_rows')!=7*n or c.get('verifier_cfg_rows')!=14*n:bad.append('candidate_CFG_rows')
                if forwards.get('action_B14_update0',{}).get('returned',0)!=c.get('verifier_forward_count',0):bad.append('B14_count')
            if forwards.get('action_B2_update1',{}).get('returned')!=1:bad.append('terminal_cache_forward')
            if bad:result['counter_errors'].append(dict(case=row['case'],arm=row['arm'],ordinal=row['ordinal'],errors=bad))
            policies[row['case'],row['arm'],row['ordinal']]=row
        rng=server/'reset-RNG'
        for folder in rng.glob('*'):
            for arm in ['C','P']:
                target=folder/'T.pt';spec=folder/(arm+'.pt')
                if target.exists() and spec.exists():
                    result['reset_RNG_pairs'].append(dict(case=folder.name,arm=arm,exact=bool(equal(torch.load(target,map_location='cpu',weights_only=False),torch.load(spec,map_location='cpu',weights_only=False))),scope='Immediately before original reset; subsequent full KV not saved'))
    cases={c for c,a,i in policies}
    for case in cases:
        for index in [0,1]:
            a=policies.get((case,'C',index));b=policies.get((case,'P',index))
            if not a or not b:continue
            def math_tree(tree):return {k:v for k,v in tree.items() if k!='context'}
            result['first_two_S_decision_pairs'].append(dict(case=case,ordinal=index,counts_exact=a['counters']==b['counters'],tree_scores_and_routes_exact=[math_tree(t) for t in a['trees']]==[math_tree(t) for t in b['trees']],scope='Saved scores/routes only, not full latent/KV/output equality'))
    result['status']='PASS_FINITE_RECORDED_PROTOCOL' if not result['route_errors'] and not result['counter_errors'] else 'FAILED_RECORDED_PROTOCOL'
    result['route_counter_status']=result['status']
    result['reset_RNG_status']='EXACT_FINITE_SAVED_PAIRS' if result['reset_RNG_pairs'] and all(x['exact'] for x in result['reset_RNG_pairs']) else 'DIFFERENCE_OR_MISSING'
    result['first_two_S_decision_status']='EXACT_FINITE_SAVED_PAIRS' if result['first_two_S_decision_pairs'] and all(x['counts_exact'] and x['tree_scores_and_routes_exact'] for x in result['first_two_S_decision_pairs']) else 'DIFFERENCE_OR_MISSING'
    if suite=='robotwin':
        for panel in sorted(r.glob('panel-*')):
            sf=panel/'STATUS.json'
            if not sf.exists():continue
            attempts=json.loads(sf.read_text()).get('attempts',[])
            by_case={}
            for row in attempts:
                if row.get('complete') and row['arm'] in ['C','P']:
                    by_case.setdefault(row['case']['task'],{})[row['arm']]=row
            for case,arms in by_case.items():
                if set(arms)!={'C','P'}:continue
                commands={}
                for arm,row in arms.items():
                    f=panel/'episodes'/Path(row['output']).name/'commands.jsonl'
                    commands[arm]=[json.loads(line)['command'] for line in f.read_text().splitlines()[:48]]
                enough=len(commands['C'])==len(commands['P'])==48
                result['robotwin_first_48_submitted_command_pairs'].append(dict(case=case,commands_per_arm={a:len(v) for a,v in commands.items()},exact=commands['C']==commands['P'] if enough else None,scope='First 16 plus 32 submitted commands, before first PF2 target chunk; persisted commands only, not all internal states'))
    prefork_difference=any(not x['counts_exact'] or not x['tree_scores_and_routes_exact'] for x in result['first_two_S_decision_pairs']) or any(x['exact'] is False for x in result['robotwin_first_48_submitted_command_pairs'])
    if result['route_errors'] or result['counter_errors']:
        result['status']='FAILED_RECORDED_ROUTE_OR_COUNTERS'
    elif prefork_difference:
        result['status']='FAILED_RECORDED_PREFORK_ISOLATION'
    else:
        result['status']='PASS_FINITE_RECORDED_SUBSET_FULL_STATE_UNKNOWN'
    report['suites'][suite]=result
(P/'AUDIT-RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:dict(status=v['status'],policies=v['policies'],legal_updates=v['committed_updates'],RNG_pairs=len(v['reset_RNG_pairs']),RNG_exact=sum(x['exact'] for x in v['reset_RNG_pairs'])) for k,v in report['suites'].items()},ensure_ascii=False))
