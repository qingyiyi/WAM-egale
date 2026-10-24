"""Audit complete natural attempts and keep timing, length and failures distinct."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,json,statistics,time,sys
HERE=Path(__file__).resolve().parent;RAW=HERE/'raw'
ALLOW_LIVE='--allow-live' in sys.argv
def lines(path):
    rows=[]
    if not path.exists():return rows
    for line in path.read_text().splitlines():
        if line.strip():
            try:rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not ALLOW_LIVE:raise
    return rows
def percentile(xs,q):
    if not xs:return None
    xs=sorted(xs);x=(len(xs)-1)*q;i=int(x);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(x-i)
def divide(a,b):return a/b if b else None
output=dict(at=time.time(),scope='Existing frozen algorithms on5090 engineering timing; exposed DEV subset, not independent TEST or inherited quality qualification',permissions={k:False for k in ['engineering_quality_screen','independent_test_complete','noninferiority_established','formal_performance_permission']},suites={},quantile_algorithm='linear interpolation at (N-1)*q (numpy default linear)')
episode_rows=[];incomplete=[];excluded_complete=[]
for suite in ['long','robotwin','goal','spatial','object']:
    root=RAW/suite
    if not root.exists():continue
    requests=defaultdict(list)
    for folder in sorted(root.glob('server-*')):
        sf=folder/'STATUS.json'
        if not sf.exists():continue
        state=json.loads(sf.read_text())
        # Default-FSDP initial Long observation is a separate engine, never the reference.
        if suite!='robotwin' and state.get('single_worker_engine')!='existing best LIBERO noFSDP':continue
        for r in lines(folder/'requests.jsonl'):
            if r.get('complete') and r.get('kind')=='policy':requests[(r['case'],r['arm'])].append(r)
    episodes={}
    for panel in sorted(root.glob('panel-*')):
        f=panel/'STATUS.json'
        if not f.exists():continue
        for r in json.loads(f.read_text()).get('attempts',[]):
            if not r.get('complete'):
                incomplete.append(dict(suite=suite,panel=panel.name,index=r['index'],arm=r['arm'],teacher_before=r.get('teacher_before'),teacher_after=r.get('teacher_after'),complete=False));continue
            terminal=r['terminal'];arm=r['arm']
            if suite=='robotwin':
                case=r['case']['task']+'-valid049';valid_steps=terminal.get('environment_take_action_cnt');submitted=terminal['raw_submitted_command_count'];horizon=terminal['task_step_limit'];warm=terminal.get('warm_episode_seconds')
            else:
                case=f'{suite}-task{terminal["task_index"]}-init49';valid_steps=submitted=terminal['env_steps'];horizon=800;warm=terminal['elapsed_sec']
            req=requests.get((case,arm),[])
            # An episode without matching final-engine request coverage is kept outside the main table.
            count=terminal.get('policy_chunks',terminal.get('infer_chunks'))
            if len(req)!=count:
                excluded_complete.append(dict(suite=suite,panel=panel.name,index=r['index'],case=case,arm=arm,matched_request_count=len(req),terminal_policy_chunks=count,reason='Different registered engine or missing request coverage; retained outside main comparison'))
                continue
            if any(x['valid_commits_g']!=50 for x in req):raise ValueError('Illegal committed-update count')
            key=(case,arm)
            if key in episodes:raise ValueError('Duplicate complete same-engine case/arm; do not pick best')
            directory=root/panel.name/'episodes'/Path(r['output']).name
            audit=(json.loads((directory/'INITIAL-PAIR-AUDIT.json').read_text()) if suite!='robotwin' else terminal.get('initial_checks'))
            row=dict(suite=suite,case=case,arm=arm,success=terminal['success'],actual_valid_environment_steps=valid_steps,raw_submitted_commands=submitted,failure_penalty=submitted if terminal['success'] else horizon,policy_chunks=count,valid_updates=sum(x['valid_commits_g'] for x in req),action_seconds=sum(x['action_accounted_seconds'] for x in req),server_policy_seconds=sum(x['policy_seconds'] for x in req),client_policy_seconds=(sum(x['elapsed'] for x in terminal['requests'] if x['route']=='policy') if suite=='robotwin' else terminal['policy_infer_wall_sec']),warm_episode_seconds=warm,client_lifecycle_seconds=r['lifecycle_seconds'],initial_pair_audit=audit,S_chunks=sum(x['scheduled']=='S' for x in req),T_chunks=sum(x['scheduled']=='T' for x in req),tree_count=sum(x['counters'].get('tree_count',0) for x in req),accepted_depth_0=sum(x['counters'].get('accepted_depth_0',0) for x in req),accepted_depth_1=sum(x['counters'].get('accepted_depth_1',0) for x in req),accepted_depth_2=sum(x['counters'].get('accepted_depth_2',0) for x in req))
            row['registered_failure_penalty']=row['failure_penalty']
            row['valid_step_failure_penalty']=valid_steps if terminal['success'] else horizon
            if suite=='robotwin':
                reset_seconds=sum(x['elapsed'] for x in terminal['requests'] if x['route']=='reset')
                cache_seconds=sum(x['elapsed'] for x in terminal['requests'] if x['route']=='cache')
            else:
                reset_seconds=terminal['reset_infer_wall_sec'];cache_seconds=terminal['cache_infer_wall_sec']
            row.update(reset_seconds=reset_seconds,cache_seconds=cache_seconds,policy_nonaction_seconds=row['client_policy_seconds']-row['action_seconds'],warm_other_seconds=warm-row['client_policy_seconds']-reset_seconds-cache_seconds,startup_close_other_seconds=row['client_lifecycle_seconds']-warm)
            row['round_count']=sum(len(x['rounds']) for x in req)
            row['round_host_seconds']=sum(z['host_seconds'] for x in req for z in x['rounds'])
            row['action_boundary_seconds']=sum(x['extra_action_setup_terminal_drain_seconds'] for x in req)
            row['route_setup_seconds']=sum(x['route_setup_seconds'] for x in req)
            if abs(row['round_host_seconds']+row['action_boundary_seconds']+row['route_setup_seconds']-row['action_seconds'])>1e-6:
                raise ValueError('Action accounting did not close')
            episodes[key]=row;episode_rows.append(row)
    common=sorted(set(c for c,a in episodes) & {c for c,a in episodes if a=='T'} & {c for c,a in episodes if a=='C'} & {c for c,a in episodes if a=='P'})
    result=dict(common_three_arm_cases=common,N=len(common),all_complete_engine_episodes=len(episodes),arms={},comparisons={})
    for arm in 'TCP':
        rows=[episodes[(c,arm)] for c in common]
        sums={k:sum(r[k] for r in rows) for k in ['actual_valid_environment_steps','raw_submitted_commands','failure_penalty','registered_failure_penalty','valid_step_failure_penalty','policy_chunks','valid_updates','action_seconds','round_count','round_host_seconds','action_boundary_seconds','route_setup_seconds','server_policy_seconds','client_policy_seconds','client_lifecycle_seconds','reset_seconds','cache_seconds','policy_nonaction_seconds','warm_other_seconds','startup_close_other_seconds','S_chunks','T_chunks','tree_count','accepted_depth_0','accepted_depth_1','accepted_depth_2']}
        warm=[r['warm_episode_seconds'] for r in rows];sums['warm_episode_seconds']=sum(warm) if warm and all(x is not None for x in warm) else None
        result['arms'][arm]=dict(N=len(rows),success=sum(r['success'] for r in rows),**sums,action_cost_per_valid_update=divide(sums['action_seconds'],sums['valid_updates']),unclosed_round_host_cost_per_valid_update=divide(sums['round_host_seconds'],sums['valid_updates']),mean_legal_updates_per_round=divide(sums['valid_updates'],sums['round_count']),policy_cost_per_chunk=divide(sums['client_policy_seconds'],sums['policy_chunks']))
        total=result['arms'][arm]
        total['successes_per_measured_client_hour_including_failures']=divide(total['success']*3600,total['client_lifecycle_seconds'])
        total['all_attempt_client_seconds_per_success']=divide(total['client_lifecycle_seconds'],total['success'])
    for arm in 'CP':
        target=result['arms']['T'];spec=result['arms'][arm]
        both=[c for c in common if episodes[c,'T']['success'] and episodes[c,arm]['success']]
        ratios=[divide(episodes[c,arm]['actual_valid_environment_steps'],episodes[c,'T']['actual_valid_environment_steps']) for c in both]
        result['comparisons'][arm]=dict(N=len(common),both_success=len(both),target_win=sum(episodes[c,'T']['success'] and not episodes[c,arm]['success'] for c in common),spec_win=sum(not episodes[c,'T']['success'] and episodes[c,arm]['success'] for c in common),action_update_speedup=divide(target['action_cost_per_valid_update'],spec['action_cost_per_valid_update']),policy_chunk_speedup=divide(target['policy_cost_per_chunk'],spec['policy_cost_per_chunk']),all_complete_client_lifecycle_speedup=divide(target['client_lifecycle_seconds'],spec['client_lifecycle_seconds']),all_complete_warm_episode_speedup=divide(target['warm_episode_seconds'],spec['warm_episode_seconds']) if target['warm_episode_seconds'] is not None and spec['warm_episode_seconds'] is not None else None,both_success_client_lifecycle_speedup=divide(sum(episodes[c,'T']['client_lifecycle_seconds'] for c in both),sum(episodes[c,arm]['client_lifecycle_seconds'] for c in both)),length_ratio_P50=percentile(ratios,.5),length_ratio_P90=percentile(ratios,.9),length_ratio_max=max(ratios) if ratios else None,paired_failure_penalty_delta=spec['failure_penalty']-target['failure_penalty'])
    for arm in 'CP':
        result['comparisons'][arm]['all_attempt_success_throughput_speedup']=divide(result['arms'][arm]['successes_per_measured_client_hour_including_failures'],result['arms']['T']['successes_per_measured_client_hour_including_failures'])
    output['suites'][suite]=result
output['incomplete_attempts']=incomplete
output['excluded_complete_episodes']=excluded_complete
(HERE/'RESULTS.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
with (HERE/'EPISODES.csv').open('w',newline='') as f:
    keys=[k for k in episode_rows[0] if k!='initial_pair_audit'] if episode_rows else ['suite','case','arm']
    writer=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');writer.writeheader();writer.writerows(episode_rows)
for suite,result in output['suites'].items():print(suite,json.dumps(dict(N=result['N'],success={a:x['success'] for a,x in result['arms'].items()},comparisons=result['comparisons']),ensure_ascii=False))
