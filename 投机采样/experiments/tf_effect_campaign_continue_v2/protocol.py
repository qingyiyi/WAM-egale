"""One prospective, pending-only continuation after the frozen DEV budget stop.

The old campaign stays FAILED/PARTIAL. Its complete robot terminal outcomes
are never rerun or replaced, including failures and terminals with audit gaps.
Imports do not prepare or launch any job.
"""
from datetime import datetime, timezone
import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys

OWN=Path(__file__).resolve().parent;EXP=OWN.parent
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from tf_effect_campaign_v1 import protocol as base
from tf_effect_collector_audit_v1.audit import live_pin

ROOT,RT,COMMIT,PERMISSIONS=base.ROOT,base.RT,base.COMMIT,base.PERMISSIONS
read,write,sha,require,environment=base.read,base.write,base.sha,base.require,base.environment
ORIGINAL=RT/'p0-tf-effect-dev96-v1-20261017-r1'
ONLY_JOB=RT/'p0-tf-effect-dev96-continue-v2-20261017-r1'
WORK_STOP='2026-10-17T09:35:00+08:00'
GPU_STOP='2026-10-17T09:40:00+08:00'
REPORT_STOP='2026-10-17T09:55:00+08:00'
USER_STOP='2026-10-17T10:00:00+08:00'
AUTH=ROOT/'.pipeline/memory/global/requirements/decision-本轮TF验证截止与晨报时间.md'
CONTINUATION='tf-effect-pending-only-continuation/v2'
EPISODE_FILES=('COUNTER.json','FINISH_SERVER.json','FINISH_READY.json','CLIENT-CLEANUP.json',
    'FINAL-SERVER-STATE.json','RESET.json','TEACHER-INDEX.json','client/episodes.jsonl',
    'client/initial_observations.npz','client/initial_state.npz','client/final_state.npz',
    'client/commands.npy','client/repro_metadata.json','client/COMMAND-MAPPING.json')


def ordered_cases(p):
    return [r for arm in p['arms'] for r in p['cases'] if r['arm']==arm]


def file_stat(path):
    st=Path(path).stat();return dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)


def validate_stopped_predecessor(path):
    path=Path(path).resolve(strict=True)
    require(path==ORIGINAL,'only the original one DEV96 attempt may be continued; no second continuation')
    p=base.load(path)
    require(p['collect'] is False and p['cases']==base.cases(False)
        and p['permissions']==PERMISSIONS and not p.get('continuation_schema'), 'frozen original96 DEV required')
    for name in ('FAILED.json','SUPERVISOR-RESULT.json','CLOCK-RESULT.json','FINAL-OWNED-CLEANUP.json','T-READY.json'):
        require((path/name).is_file(),'predecessor has not reached its own terminal/cleanup: '+name)
    s=read(path/'SUPERVISOR-RESULT.json');c=read(path/'CLOCK-RESULT.json');cleanup=read(path/'FINAL-OWNED-CLEANUP.json')
    require(s['status']=='FAILED_OR_PARTIAL' and s.get('failure') is not None
        and cleanup['all_roots_reaped'] is True and c['all_observed_owned_live_processes_exited'] is True
        and c['error'] is None and c['no_foreign_signals'] is True,
        'only cleanly stopped original budget PARTIAL is eligible')
    now=datetime.now(timezone.utc);old_work=datetime.fromisoformat(p['absolute_work_stop'])
    require(now>=old_work,'do not prepare while original campaign is still within its work window')
    error=str(s['failure'].get('error',''))
    direct=error in ("RuntimeError('registered work deadline')",'RuntimeError("registered work deadline")',
                    "ValueError('registered work deadline')",'ValueError("registered work deadline")')
    clock_stop=None
    if (path/'CLOCK-STOP.json').exists():
        clock_stop=read(path/'CLOCK-STOP.json')
        require(clock_stop['reason']=='registered complete TF campaign deadline'
            and datetime.fromisoformat(clock_stop['at'])>=old_work,
            'predecessor clock stop was not its registered budget deadline')
    owned_stop=error in ("SystemExit('owned stop')",'SystemExit("owned stop")') and clock_stop is not None
    deadline_stop_request=error in ("ValueError('STOP_REQUESTED')",'ValueError("STOP_REQUESTED")') and clock_stop is not None
    require(direct or owned_stop or deadline_stop_request,'non-budget model/client/device/audit failure cannot trigger automatic continuation')
    require(c['external_stop_reason'] is None or (c['external_stop_reason']=='registered complete TF campaign deadline'
        and clock_stop is not None),'external stop was not the original registered budget deadline')
    pins=list(s.get('observed_pins',[]))+[x['pin'] for x in c.get('process_observations',[])]
    if (path/'ROOT-LAUNCH.json').exists():pins.append(read(path/'ROOT-LAUNCH.json')['pin'])
    if (path/'CLOCK-LAUNCH.json').exists():pins.append(read(path/'CLOCK-LAUNCH.json')['pin'])
    unique={(x['pid'],x['start_ticks'],x['uid']):x for x in pins}
    observations=[live_pin(x) for x in unique.values()]
    require(observations and not any(x['live'] for x in observations),'old owned root/clock/worker lifetime is still live')
    return p,dict(status='PASS_ORIGINAL_BUDGET_STOP_AND_OWNED_EXIT',failure_error=error,
        supervisor_wall_sec=s.get('elapsed_sec'),clock_wall_sec=c.get('elapsed_quality_process_seconds'),
        process_observations=observations,original_failure_unchanged=True)


def inventory_terminals(predecessor,p):
    entries=[];pending=[];attempt_debits=[];known=0;frozen={}
    supervisor=read(predecessor/'SUPERVISOR-RESULT.json') if (predecessor/'SUPERVISOR-RESULT.json').is_file() else {}
    completed_keys=set(supervisor.get('completed') or [])
    for row in ordered_cases(p):
        key=row['key'];out=predecessor/'episodes'/key;record_path=out/'client/episodes.jsonl'
        records=[];record_parse_error=None
        if record_path.exists():
            try:records=[json.loads(x) for x in record_path.read_text().splitlines() if x.strip()]
            except json.JSONDecodeError as exc:record_parse_error=repr(exc)
            require(len(records)<=1,'multiple episode attempts cannot be silently selected: '+key)
        record=records[0] if records else None
        terminal_path=record_path;terminal_kind='episodes_jsonl'
        full=(record is not None and record.get('status')=='completed'
            and isinstance(record.get('success'),bool) and record.get('task_index')==row['task']
            and record.get('episode_index')==row['init'] and record.get('libero_horizon')==800
            and record.get('max_policy_chunks') is None and record.get('truncated_after_max_policy_chunks') is False
            and (record.get('success') is True or record.get('horizon_reached') is True))
        if record is not None and record.get('status')=='completed':
            require(full,'a completed robot record violates original terminal contract; never rerun it to replace the outcome')
        if not full:
            # The source writes command mapping before appending episodes.jsonl.
            # A deadline between those writes must not replace a complete loss.
            mapping_path=out/'client/COMMAND-MAPPING.json'
            if mapping_path.is_file():
                mapping=read(mapping_path)
                if mapping.get('complete_episode') is True:
                    require(isinstance(mapping.get('success'),bool) and isinstance(mapping.get('env_steps'),int)
                        and mapping['env_steps']>0 and mapping.get('termination_reason')==('success' if mapping['success'] else 'full_horizon'),
                        'completed command mapping has inconsistent terminal outcome')
                    contexts=[x['context'] for x in mapping.get('chunks',[])]
                    require(contexts and all(c['task_index']==row['task'] and c['episode_index']==row['init']
                        and c['parallel_worker_id']==key and c['campaign_id']==predecessor.name for c in contexts),
                        'completed mapping terminal belongs to another episode')
                    record=dict(success=mapping['success'],env_steps=mapping['env_steps'])
                    terminal_path=mapping_path;terminal_kind='command_mapping';full=True
            if not full:
                for name in ('FINISH_SERVER.json','FINISH_READY.json'):
                    finish_path=out/name
                    if not finish_path.is_file():continue
                    finish=read(finish_path)
                    if finish.get('full_episode_terminal') is True:
                        require(finish.get('status')=='PASS' and finish.get('key')==key and finish.get('arm')==row['arm']
                            and isinstance(finish.get('success'),bool) and isinstance(finish.get('env_steps'),int)
                            and finish['env_steps']>0,'finished original terminal receipt is inconsistent')
                        record=dict(success=finish['success'],env_steps=finish['env_steps'])
                        terminal_path=finish_path;terminal_kind='finish';full=True;break
        require(full or key not in completed_keys,
            'old supervisor proved a completed robot episode but its original outcome receipts are missing; do not rerun or guess')
        counter_path=out/'COUNTER.json';counter=None;counter_parse_error=None
        if counter_path.is_file():
            try:counter=read(counter_path)
            except json.JSONDecodeError as exc:counter_parse_error=repr(exc)
        if counter is not None:
            count=counter['target_forwards']
            require(isinstance(count,int) and 0<count<=p['per_episode_target_forward_cap'], 'original counter outside cap')
            known+=count
        elif out.exists():
            attempt_debits.append(dict(key=key,source_job=str(predecessor),counter_missing=True,
                lower_known=0,upper_debited=p['per_episode_target_forward_cap'],quality_terminal=bool(full),
                counter_parse_error=counter_parse_error,
                actual_attempt_target_forwards='UNKNOWN; conservative original per-episode cap debited'))
        if not full:
            pending.append(key)
            continue
        missing=[name for name in EPISODE_FILES if not (out/name).is_file()]
        if terminal_kind!='episodes_jsonl':missing.append('canonical_completed_episode_record_missing_or_nonterminal')
        if record_parse_error:missing.append('client/episodes.jsonl invalid JSON: '+record_parse_error)
        if counter_parse_error:missing.append('COUNTER.json invalid JSON: '+counter_parse_error)
        arm_missing=[];arm=row['arm']
        for path in (predecessor/f'{arm}-READY.json',predecessor/'arms'/arm/'TARGET-INITIAL-RNG.pt',
                     predecessor/'arms'/arm/'OWNED-CLEANUP.json'):
            if not path.is_file():arm_missing.append(str(path))
            else:frozen[str(path)]=file_stat(path)
        if arm in p['gpu_cal_jobs']:
            cal=Path(p['gpu_cal_jobs'][arm])/'RESULT.json'
            if not cal.is_file():arm_missing.append(str(cal))
            else:frozen[str(cal)]=file_stat(cal)
        elif arm=='TF_CONT':
            cal=predecessor/'arms'/arm/'CAL-REUSE.json'
            if not cal.is_file():arm_missing.append(str(cal))
            else:frozen[str(cal)]=file_stat(cal)
        if counter is not None:
            for artifact in counter.get('save_async',{}).get('kept_artifacts',[]):
                if not Path(artifact['path']).is_file():missing.append(artifact['path'])
                else:frozen[artifact['path']]=file_stat(artifact['path'])
        else:missing.append('saved-action/latent inventory UNKNOWN without COUNTER')
        audit_status='COMPLETE_AUDIT' if not missing and not arm_missing else 'TERMINAL_QUALITY_WITH_AUDIT_GAP'
        # COMPLETE_AUDIT means required receipts exist. The independent report
        # must still run the full original numerical/state/camera auditor.
        entries.append(dict(case=copy.deepcopy(row),key=key,source_job=str(predecessor),quality_terminal=True,
            episode_record=str(terminal_path),episode_record_sha256=sha(terminal_path),terminal_record_kind=terminal_kind,
            success=record['success'],env_steps=record['env_steps'],failure_penalty=record['env_steps'] if record['success'] else 800,
            audit_status=audit_status,audit_status_is_receipt_presence_only=True,
            missing_receipts=missing,missing_arm_receipts=arm_missing,
            no_outcome_replacement=True))
        for name in EPISODE_FILES:
            path=out/name
            if path.is_file():frozen[str(path)]=file_stat(path)
    require(len(entries)+len(pending)==96 and len(set(pending))==len(pending),'exact96 terminal/pending partition required')
    return entries,pending,known,attempt_debits,frozen


def prepare(job,predecessor):
    job=Path(job).resolve();predecessor=Path(predecessor).resolve(strict=True)
    require(job==ONLY_JOB and not job.exists(),'one fixed fresh continuation namespace; never retry/second continuation')
    p,stop=validate_stopped_predecessor(predecessor)
    now=datetime.now(timezone.utc);hard=datetime.fromisoformat(GPU_STOP);work=datetime.fromisoformat(WORK_STOP)
    require((work-now).total_seconds()>p['model_ready_wall_sec']+p['per_episode_wall_sec']+180,
        'insufficient time to admit a full model setup and untruncated episode before09:35')
    require(AUTH.is_file() and '2026-10-17 10:00' in AUTH.read_text(),'current one-night10:00 authorization evidence missing')
    entries,pending,known,debits,frozen=inventory_terminals(predecessor,p)
    require(pending,'no missing robot terminal episodes; no GPU continuation needed')
    previous_bytes=sum(x.stat().st_size for x in predecessor.rglob('*') if x.is_file())
    debit=known+sum(x['upper_debited'] for x in debits)
    remaining_forwards=p['job_target_forward_cap']-debit;remaining_storage=p['storage_cap_bytes']-previous_bytes
    require(remaining_forwards>=p['per_episode_target_forward_cap'] and remaining_storage>64*2**20,
        'original cumulative forward/storage cap has no sufficient remaining allowance')
    budget=dict(original_target_forward_cap=p['job_target_forward_cap'],predecessor_target_forwards_lower_known=known,
        predecessor_target_forwards_upper_debited=debit,unclosed_attempt_debits=debits,new_target_forward_cap=remaining_forwards,
        original_storage_cap_bytes=p['storage_cap_bytes'],predecessor_storage_bytes_actual=previous_bytes,
        new_storage_cap_bytes=remaining_storage,predecessor_supervisor_wall_sec=stop['supervisor_wall_sec'],
        predecessor_clock_wall_sec=stop['clock_wall_sec'],original_created_at=p['created_at'],
        prior_absolute_work_stop=p['absolute_work_stop'],prior_absolute_cleanup_deadline=p['absolute_cleanup_deadline'],
        new_absolute_work_stop=WORK_STOP,new_absolute_cleanup_deadline=GPU_STOP,report_by=REPORT_STOP,user_deadline=USER_STOP,
        no_budget_reset=True,meaning='new separately authorized bounded window; original cap/failure stays unchanged; all prior cost debited',
        partial_attempt_actual_forward_count='UNKNOWN where no COUNTER; bounded above, never treated as zero',
        master_budget_origin=p.get('global_budget_origin'),prior_global_cleanup_deadline=p.get('global_cleanup_deadline'))
    manifest=dict(schema='tf-effect-imported-robot-terminals/v2',predecessor_job=str(predecessor),
        full_original_case_count=96,entries=entries,pending_keys=pending,
        imported_complete_robot_terminals=len(entries),audit_gap_keys=[x['key'] for x in entries if x['audit_status']!='COMPLETE_AUDIT'],
        includes_all_complete_success_and_failure=True,terminal_outcomes_never_rerun=True,
        receipt_presence_is_not_full_numerical_audit=True,predecessor_stop=stop)
    q=copy.deepcopy(p);q.update(job=str(job),continuation_schema=CONTINUATION,predecessor_job=str(predecessor),
        imported_terminal_manifest=str(job/'IMPORTED-TERMINALS.json'),pending_keys=pending,cumulative_budget=budget,
        goal='Only finish originally fixed96 cases without a complete robot terminal; preserve every old outcome and PARTIAL',
        created_at=now.isoformat(),max_run_wall_sec=math.floor((hard-now).total_seconds()),cleanup_reserve_sec=300,
        absolute_work_stop=WORK_STOP,absolute_cleanup_deadline=GPU_STOP,global_cleanup_deadline=REPORT_STOP,
        user_report_deadline=USER_STOP,job_target_forward_cap=remaining_forwards,storage_cap_bytes=remaining_storage,
        planned_full_episodes=len(pending),full_original_case_count=96,terminal_status='REGISTERED_NOT_STARTED',
        evidence_path=str(job),user_deadline_amendment=str(AUTH),no_second_continuation=True,no_new_training=True,
        target_initial_rng_reference=str(predecessor/'T-READY.json'),continuation_only=True,
        gpu_cal_jobs={arm:str(job/'cal'/arm) for arm in p['gpu_cal_jobs']},
        source_terminal_stat=frozen)
    # Source cases, weights, seed, threshold, engine and helper modules remain
    # identical; the compatible original server still sees the full96 cases.
    require(q['cases']==p['cases'] and q['selected']==p['selected'] and q['seed']==p['seed']
        and q['mode']==p['mode'] and q['threshold']==p['threshold'],'scientific configuration changed')
    q['source_sha256'].update({str(x):sha(x) for x in OWN.glob('*.py')})
    receipts=[predecessor/name for name in ('plan.json','PREREGISTRATION.json','FAILED.json','SUPERVISOR-RESULT.json',
        'CLOCK-RESULT.json','FINAL-OWNED-CLEANUP.json','T-READY.json')]+[AUTH]
    if (predecessor/'CLOCK-STOP.json').exists():receipts.append(predecessor/'CLOCK-STOP.json')
    q['evidence_sha256'].update({str(x):sha(x) for x in receipts})
    q['evidence_sha256'].update({e['episode_record']:e['episode_record_sha256'] for e in entries})
    job.mkdir();write(job/'IMPORTED-TERMINALS.json',manifest)
    q['evidence_sha256'][str(job/'IMPORTED-TERMINALS.json')]=sha(job/'IMPORTED-TERMINALS.json')
    q.pop('preregistration_sha256',None);write(job/'PREREGISTRATION.json',q)
    write(job/'plan.json',{**q,'preregistration_sha256':sha(job/'PREREGISTRATION.json')})
    return q


def load(job,verify=True):
    p=base.load(job,verify=verify)
    require(Path(p['job'])==ONLY_JOB and p.get('continuation_schema')==CONTINUATION
        and p['predecessor_job']==str(ORIGINAL) and p['no_second_continuation'] is True,
        'one authorized independent continuation only')
    require(p['absolute_work_stop']==WORK_STOP and p['absolute_cleanup_deadline']==GPU_STOP
        and p['global_cleanup_deadline']==REPORT_STOP and p['user_report_deadline']==USER_STOP,
        'fixed09:35/09:40/09:55/10:00 boundaries changed')
    manifest=read(p['imported_terminal_manifest']);order=[r['key'] for r in ordered_cases(p)]
    imported=[x['key'] for x in manifest['entries']]
    require(p['pending_keys']==manifest['pending_keys']==[k for k in order if k not in set(imported)]
        and len(imported)+len(p['pending_keys'])==96,'terminal/pending partition or original execution order changed')
    require(p['job_target_forward_cap']==p['cumulative_budget']['new_target_forward_cap']
        and p['storage_cap_bytes']==p['cumulative_budget']['new_storage_cap_bytes'],'remaining cumulative budgets changed')
    if verify:
        validate_stopped_predecessor(ORIGINAL)
        for f,s in p['source_terminal_stat'].items():require(file_stat(f)==s,'an imported original terminal artifact changed')
    return p


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--predecessor',type=Path,default=ORIGINAL);parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args();require(args.prepare,'preparation is explicit; this module never launches work')
    p=prepare(args.job,args.predecessor)
    print(dict(status='REGISTERED_PENDING_ONLY_CONTINUATION',job=p['job'],pending=len(p['pending_keys']),GPU_started=False))
