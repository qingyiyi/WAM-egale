"""Bind a new independent collection to the closed overnight ledger.

Historical GPU oracles are provenance only; the changed source must run its
own complete OFF/ON native proof before the first full episode.
"""
import json
import math
from datetime import datetime
from pathlib import Path

from experiment import ROOT, digest

AUDIT = ROOT / '.pipeline/research/speculative-sampling/evidence/fusion-withdraw-draft-onpolicy-20261014/NIGHT-FINAL-ACCOUNTING-AND-OWNED-CLEANUP.json'
AUDIT_SHA = 'cf40010a64d2516939be2ae19f6244444eb8efacc89f8c6fb717ec93e4b03ea6'
CARRY = Path('/var/tmp/cxz-wam-quality-20261014/p0-overnight-refresh-binding-v1-20261015-prepared/GPU-COLLECTION.json')
CARRY_SHA = 'fbcac9f16f1ad5fcecf1fb95f5e7bd07d40b64e0820992ae0c6f76da0e185031'
CANDIDATE_SHA = '03e1354fc6bf4e9838607590466e3de70efc8b10a959f96665502bae68d3de49'
PRIOR_SECONDS = 40311.19392736256
PRIOR_INTENTS = 192771
PRIOR_MODELS = 38


def validate_prior_identity_audits(job, assignments, terminal):
    entries=terminal.get('episode_identity_audits')
    if not isinstance(entries,list) or len(entries)!=len(assignments):
        raise ValueError('previous wave identity audits absent')
    expected_ids={x['id'] for x in assignments}
    if {x.get('id') for x in entries}!=expected_ids:
        raise ValueError('previous wave identity audit IDs changed')
    for item in assignments:
        out=Path(job)/'campaigns'/item['arm']/'workers'/item['id']
        audit_path=out/'EPISODE-IDENTITY-AUDIT.json'
        entry=next(x for x in entries if x['id']==item['id'])
        if (Path(entry.get('path','')).resolve()!=audit_path.resolve()
            or entry.get('task')!=item['task'] or entry.get('init')!=item['init']
            or digest(audit_path)!=entry.get('sha256')):
            raise ValueError('previous wave audit file identity/hash changed')
        audit=json.loads(audit_path.read_text())
        ep=[json.loads(x) for x in (out/'client/episodes.jsonl').read_text().splitlines() if x.strip()]
        if (len(ep)!=1
            or audit.get('status')!='PASS_COMPLETE_EPISODE_IDENTITY_AND_SAME_QUERY_COVERAGE'
            or audit.get('task')!=item['task'] or audit.get('init')!=item['init']
            or audit.get('episode_attempt_id')!=ep[0].get('episode_attempt_id')
            or audit.get('infer_chunks')!=ep[0].get('infer_chunks')
            or audit.get('env_steps')!=ep[0].get('env_steps')
            or audit.get('trees',0)<=0 or audit.get('rows')!=3*audit['trees']):
            raise ValueError('previous wave audit payload/episode changed')
        client_files=audit.get('client_files')
        required=[out/'client'/name for name in ('episodes.jsonl','repro_metadata.json','chunk_events.jsonl',
            'commands.npy','initial_state.npz','final_state.npz','run_metadata.json')]
        if (not isinstance(client_files,list) or len(client_files)!=len(required)
            or {str(x.resolve()) for x in required}!={str(Path(v.get('path','')).resolve()) for v in client_files}):
            raise ValueError('previous wave client evidence file set changed')
        for file in client_files:
            if digest(file['path'])!=file.get('sha256'):
                raise ValueError('previous wave client evidence changed')
        for key,pattern in (('label_files','*.pt'),('chunk_marker_files','CHUNK-*.json')):
            files=sorted((out/'teacher').glob(pattern))
            registered=audit.get(key)
            if (not isinstance(registered,list) or len(files)!=len(registered)
                or {str(f.resolve()) for f in files}!={str(Path(v.get('path','')).resolve()) for v in registered}):
                raise ValueError('previous wave label or marker file set changed')
            for file in registered:
                if digest(file['path'])!=file.get('sha256'):
                    raise ValueError('previous wave label or marker content changed')
        if len(audit['label_files'])!=audit['trees'] or len(audit['chunk_marker_files'])!=audit['infer_chunks']:
            raise ValueError('previous wave audit coverage changed')
    return True


def authority(p):
    if p['phase'] != 'long8_same_query_train_init17' or p['attempts_max'] != 1:
        raise ValueError('new independent one-attempt phase required')
    wave = p.get('wave_index')
    if type(wave) is not int or not 0 <= wave < 4:
        raise ValueError('one of four waves required')
    if wave == 0 and (p['previous_pass_terminal'] != str(CARRY) or p['previous_pass_terminal_sha256'] != CARRY_SHA):
        raise ValueError('first wave must bind final overnight terminal')
    if (p['retry_of'] is not None or p['retry_of_sha256'] is not None
        or p['source_carry'] != p['previous_pass_terminal']
        or p['source_carry_sha256'] != p['previous_pass_terminal_sha256']):
        raise ValueError('v4 accepts only a direct preceding PASS terminal; no automatic retry')
    if p['source_carry_audit'] != str(AUDIT) or p['source_carry_audit_sha256'] != AUDIT_SHA:
        raise ValueError('night audit identity changed')
    if p['GPU_UUID'] != 'GPU-754f96ec-4567-7af2-7048-7e8957a6eb36':
        raise ValueError('physical A100 GPU identity changed')
    if datetime.fromisoformat(p['GPU_absolute_stop_at']).utcoffset() is None:
        raise ValueError('explicit timezone-aware absolute stop required')
    if p['aggregate_GPU_seconds_cap'] != PRIOR_SECONDS + 6000 or p['aggregate_target_cap'] != PRIOR_INTENTS + 33280 or p['aggregate_model_launches_cap'] != 50:
        raise ValueError('bounded new additive caps required')
    if p['aggregate_prior_model_launches'] < PRIOR_MODELS or p['aggregate_prior_model_launches']+3>50:
        raise ValueError('model launch cap exhausted')
    return p


def validate_carry(p):
    authority(p)
    if digest(AUDIT) != AUDIT_SHA or digest(CARRY) != CARRY_SHA:
        raise ValueError('closed overnight ledger hash changed')
    audit = json.loads(AUDIT.read_text())
    terminal = json.loads(CARRY.read_text())
    if (audit['status'] != 'PASS_NIGHT_LEDGER_AND_PINNED_OWNED_CLEANUP'
        or audit['latest_terminal'] != str(CARRY)
        or audit['latest_terminal_sha256'] != CARRY_SHA
        or audit['aggregate_protocol_GPU_seconds'] != PRIOR_SECONDS
        or audit['aggregate_charged_target_intents'] != PRIOR_INTENTS
        or audit['aggregate_night_model_launches'] != PRIOR_MODELS):
        raise ValueError('final global ledger mismatch')
    if (terminal['status'] != 'PASS_OVERNIGHT_REFRESH_COLLECTION'
        or terminal['aggregate_GPU_seconds'] != PRIOR_SECONDS
        or terminal['aggregate_target_intents'] != PRIOR_INTENTS
        or not terminal['owned_roots_exited']
        or terminal['cleanup']['status'] != 'PASS_OWNED_CLEANUP'
        or terminal['cleanup'].get('remaining_owned_GPU_pids') != []
        or terminal.get('counter_errors')):
        raise ValueError('last source phase not terminal and closed')
    if not math.isfinite(float(terminal['aggregate_GPU_seconds'])):
        raise ValueError('invalid closed elapsed total')
    anchor=Path(p['previous_pass_terminal'])
    if digest(anchor)!=p['previous_pass_terminal_sha256']:
        raise ValueError('preceding PASS terminal hash changed')
    latest=json.loads(Path(p['source_carry']).read_text())
    if digest(p['source_carry'])!=p['source_carry_sha256']:
        raise ValueError('latest carry hash changed')
    if Path(p['source_carry'])!=anchor:
        raise ValueError('latest carry must equal preceding PASS terminal')
    if p['wave_index']:
        path=anchor
        if path.name!='GPU-COLLECTION.json':
            raise ValueError('previous wave terminal hash invalid')
        expected_index=p['wave_index']-1
        seen=set()
        while path!=CARRY:
            if path in seen or len(seen)>4:raise ValueError('wave carry cycle/depth')
            seen.add(path)
            value=json.loads(path.read_text())
            plan_path=path.parent/'plan.json';parent=json.loads(plan_path.read_text())
            expected_pair=(2+2*expected_index,3+2*expected_index)
            expected_assignments=[[dict(id=f'collect-t{t}-i17',arm='COLLECT',task=t,init=17,split='train',cap=4000)
                                   for t in expected_pair]]
            if (value['status']!='PASS_GPU_FULL_EPISODE_COLLECTION'
                or value['plan_sha256']!=digest(plan_path)
                or value['completed_episodes']!=2
                or value.get('charged_model_launches')!=3
                or not value['owned_roots_exited']
                or value['cleanup']['status']!='PASS_OWNED_CLEANUP'
                or value['cleanup'].get('remaining_owned_GPU_pids')!=[]
                or value.get('counter_errors')
                or parent['phase']!='long8_same_query_train_init17'
                or not Path(parent['job']).name.startswith('p0-morning-onpolicy-long8-v4-')
                or parent['wave_index']!=expected_index
                or parent['assignments']!=expected_assignments):
                raise ValueError('previous wave incomplete or assignment changed')
            for item in parent['assignments'][0]:
                ep=path.parent/'campaigns'/'COLLECT'/'workers'/item['id']/'client'/'episodes.jsonl'
                rows=[json.loads(x) for x in ep.read_text().splitlines() if x.strip()]
                if (len(rows)!=1 or rows[0]['status']!='completed' or rows[0]['task_index']!=item['task']
                    or rows[0]['episode_index']!=17 or rows[0]['truncated_after_max_policy_chunks']):
                    raise ValueError('previous wave episode identity/coverage invalid')
            validate_prior_identity_audits(path.parent,parent['assignments'][0],value)
            prior_path=Path(parent['source_carry'])
            if digest(prior_path)!=parent['source_carry_sha256']:
                raise ValueError('previous wave chain hash changed')
            prior=json.loads(prior_path.read_text())
            if (parent['aggregate_prior_GPU_seconds']!=prior['aggregate_GPU_seconds']
                or parent['aggregate_prior_target_intents']!=prior['aggregate_target_intents']
                or prior['aggregate_GPU_seconds']+value['new_GPU_seconds']!=value['aggregate_GPU_seconds']
                or prior['aggregate_target_intents']+value['charged_target_intents']!=value['aggregate_target_intents']
                or value['aggregate_model_launches']!=(parent['aggregate_prior_model_launches']+value['charged_model_launches'])):
                raise ValueError('wave ledger arithmetic changed')
            path=prior_path;expected_index-=1
        if expected_index!=-1:raise ValueError('wave carry did not reach overnight anchor at expected depth')
    if p['wave_index']==0 and anchor!=CARRY:
        raise ValueError('first wave anchor differs from overnight terminal')
    expected_models=latest.get('aggregate_model_launches',PRIOR_MODELS)
    if (p['aggregate_prior_GPU_seconds']!=latest['aggregate_GPU_seconds']
        or p['aggregate_prior_target_intents']!=latest['aggregate_target_intents']
        or p['aggregate_prior_model_launches']!=expected_models):
        raise ValueError('latest passed wave usage not carried forward')
    return latest


def validate_oracles(p):
    if (p['new_numeric_OFF_ON_required'] is not True
        or p['previous_native_oracles_current_proof'] is not False
        or p['new_checkpoint_sha256'] != CANDIDATE_SHA
        or digest(p['new_checkpoint']) != CANDIDATE_SHA):
        raise ValueError('exact TF generator and fresh native proof required')
    return dict(status='HISTORICAL_NATIVE_PROOF_NOT_CURRENT_GATE',
                required_new_full_chunks=4, required_new_target_cap=320,
                previous_GPU_oracles_reused=False)
