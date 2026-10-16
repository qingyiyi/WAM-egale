"""CPU checks only, including actual Torch imports and resource negative cases."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import copy
from types import SimpleNamespace
from unittest.mock import patch
from experiment import OWN,VA,CLIENT,ROOT,plan,digest,write
from shared_resource import SharedResources,charged_reservations

def identity_negative_checks(p):
    from identity_gate import validate
    import hashlib
    import torch
    item=p['assignments'][0][0];task=item['task'];init=item['init'];attempt='cpu-fixture-attempt'
    episode=dict(status='completed',task_index=task,episode_index=init,episode_attempt_id=attempt,
                 truncated_after_max_policy_chunks=False,success=False,horizon_reached=True,
                 env_steps=12,infer_chunks=1)
    repro=dict(command_count=12,actual_EGL_initialization_attempts=1,
               no_policy_chunk_truncation=True,init_state_sha256='a'*64)
    summary=dict(errors=[],extra_teacher_forwards=0,saved_trees=1)
    provenance=dict(task=task,init=init,split='train',draft_sha256=p['new_checkpoint_sha256'],
                    plan_sha256=digest(p['job']+'/plan.json'),
                    target_checkpoint_id=p['frozen_reference']['target_checkpoint_id'])
    markers=[dict(complete=True,attempted_trees=1,saved_trees=1,
                  nodes_per_tree=3,provenance=provenance,
                  context=dict(task_index=task,episode_index=init,episode_attempt_id=attempt,rollout_chunk=0))]
    state=torch.zeros(112,dtype=torch.float32);state_hash=hashlib.sha256(state.numpy().tobytes()).hexdigest()
    rows=[dict(node=n,task_index=task,episode_index=init,episode_attempt_id=attempt,
               split='train',source_state_dtype='torch.bfloat16',state_sha256=state_hash,
               target_query_state_sha256=state_hash,rollout_chunk=0,tree_step_index=0,
               state=state.clone(),previous=state.clone(),target=state.clone())
          for n in range(3)]
    payloads=[dict(schema_version='bonus-prefix-same-query-teacher/v2',extra_teacher_forwards=0,
                   captures_all_three_predictor_nodes=True,rows=rows,
                   contract=dict(mode='verifier_k2_b2_bonus',bonus_enabled=True,selector='prefix_then_cost',
                                 threshold=p['frozen_reference']['threshold'],action_updates_per_chunk=50,
                                 used_action_channel_ids=list(range(7)),cfg_scale=1.,cache_mode='original',
                                 provenance=provenance))]
    events=[dict(event_schema_version='wam-p0-chunk-event/v1',parallel_worker_id=item['id'],
                 task_index=task,episode_index=init,episode_attempt_id=attempt,
                 rollout_chunk=0,action_shape=[7,4,4],env_steps_before=0,
                 env_steps_after=12,env_steps_delta=12)]
    expected=dict(plan_sha256=digest(p['job']+'/plan.json'),draft_sha256=p['new_checkpoint_sha256'],
                  target_checkpoint_id=p['frozen_reference']['target_checkpoint_id'],
                  threshold=p['frozen_reference']['threshold'])
    assert validate(item,episode,repro,summary,markers,payloads,events,**expected)['rows']==3
    def rejects(which,change):
        args=copy.deepcopy([item,episode,repro,summary,markers,payloads,events])
        change(args)
        try:validate(*args,**expected)
        except ValueError:return which
        raise AssertionError('identity gate accepted '+which)
    return [
        rejects('wrong_episode_task',lambda a:a[1].__setitem__('task_index',9 if task!=9 else 2)),
        rejects('wrong_command_count',lambda a:a[2].__setitem__('command_count',11)),
        rejects('wrong_row_attempt',lambda a:a[5][0]['rows'][1].__setitem__('episode_attempt_id','other')),
        rejects('wrong_teacher_source',lambda a:a[5][0]['contract']['provenance'].__setitem__('target_checkpoint_id','other')),
        rejects('wrong_marker_task',lambda a:a[4][0]['context'].__setitem__('task_index',9 if task!=9 else 2)),
        rejects('wrong_marker_provenance',lambda a:a[4][0]['provenance'].__setitem__('target_checkpoint_id','other')),
        rejects('missing_third_node',lambda a:a[5][0]['rows'].pop()),
        rejects('extra_teacher_forward',lambda a:a[5][0].__setitem__('extra_teacher_forwards',1)),
        rejects('nonfinite_label_tensor',lambda a:a[5][0]['rows'][0]['target'].__setitem__(0,float('nan'))),
        rejects('incorrect_actual_state_hash',lambda a:a[5][0]['rows'][0].__setitem__('state_sha256','f'*64)),
        rejects('missing_chunk_event',lambda a:a[6].clear()),
        rejects('wrong_chunk_event_identity',lambda a:a[6][0].__setitem__('task_index',9 if task!=9 else 2)),
        rejects('discontinuous_chunk_steps',lambda a:a[6][0].__setitem__('env_steps_before',1)),
    ]

def checks(job,p):
    passed=[]
    for f in OWN.glob('*.py'):compile(f.read_text(),str(f),'exec')
    passed.append('source_compile')
    passed.extend(identity_negative_checks(p))
    import inspect
    from supervisor import Campaign
    assert 'validate_identity' in inspect.getsource(Campaign.wave)
    wave_source=inspect.getsource(Campaign.wave)
    assert wave_source.index('validate_identity(')<wave_source.index('self.completed+=len(specs)')
    assert wave_source.index('CLIENT-ADMISSION-INTENT.json')<wave_source.index('self.spawn(cmd')
    passed.append('identity_gate_called_by_actual_supervisor_wave')
    passed.append('completed_only_after_both_identity_audits_and_durable_preclient_intent')
    assert len(p['assignments'])==1 and len(p['assignments'][0])==2 and sum(v['cap'] for v in p['assignments'][0])==8000 and p['fixed_target_cap']==320 and p['model_worker_cap']==3 and p['target_cap']==8320
    assert len(p['fixed_slots'])*len(p['snapshots'])==4
    assert {v['init'] for v in p['assignments'][0]}=={17}
    assert [v['task'] for v in p['assignments'][0]]==[2+2*p['wave_index'],3+2*p['wave_index']]
    passed.extend(('two_independent_EGL_and_complete_train_split','four_oracles_plus_two_full_episode_caps'))
    assert p['aggregate_prior_GPU_seconds']+p['GPU_hard_limit_seconds']<=p['aggregate_GPU_seconds_cap'] and p['aggregate_prior_target_intents']+8320<=p['aggregate_target_cap']
    passed.append('aggregate_reservation_caps')
    assert not SharedResources.startup_ready({'memory_free_bytes':2*2**30-1},[])
    assert SharedResources.startup_ready({'memory_free_bytes':2*2**30},[])
    assert SharedResources.startup_ready({'memory_free_bytes':2*2**30},[999])
    passed.extend(('runtime_floor_boundary','foreign_process_allowed_only_for_nonperformance_collection'))
    with tempfile.TemporaryDirectory(prefix='onpolicy-CPU-negative-') as temp:
        root=Path(temp);folder=root/'worker';folder.mkdir();entry=dict(out=str(folder),cap=320)
        from retry_guard import inspect_workers,conservative_model_launches
        assert conservative_model_launches(0,[object()]*3)==3
        assert conservative_model_launches(3,[object()]*3)==3
        passed.append('popen_before_pid_pin_failure_charges_reserved_model_launches')
        pair=p['assignments']
        observation=inspect_workers(root,pair,client_admission_started=False)
        assert observation['no_client_or_rollout_started']
        one=root/'campaigns'/pair[0][0]['arm']/'workers'/pair[0][0]['id']
        (one/'client').mkdir(parents=True)
        (one/'client/episodes.jsonl').write_text(json.dumps(dict(status='completed',task_index=pair[0][0]['task'],
            episode_index=17,episode_attempt_id='already-run'))+'\n')
        observation=inspect_workers(root,pair,client_admission_started=False)
        assert not observation['no_client_or_rollout_started'] and observation['observed_completed_client_episodes']==1
        (one/'client/episodes.jsonl').unlink()
        assert inspect_workers(root,pair,client_admission_started=True)['no_client_or_rollout_started'] is False
        (root/'CLIENT-ADMISSION-INTENT.json').write_text('{}')
        assert inspect_workers(root,pair,client_admission_started=False)['client_started'] is True
        passed.extend(('one_completed_client_never_whole_pair_retry_even_if_accepted_zero',
                       'durable_client_admission_intent_blocks_retry'))
        assert charged_reservations(root,[entry])['charged_target_intents']==320
        (folder/'COUNTER-CLOSED.json').write_text('{bad-json')
        v=charged_reservations(root,[entry]);assert v['charged_target_intents']==320 and v['counter_errors']
        good=dict(status='CLOSED',cap=320,target_intents=8,host_returned=8,failed_forwards=0,active_forwards=0,request_active=False)
        (folder/'COUNTER-CLOSED.json').write_text(json.dumps(good));assert charged_reservations(root,[entry])['charged_target_intents']==8
        good['host_returned']=7;(folder/'COUNTER-CLOSED.json').write_text(json.dumps(good))
        assert charged_reservations(root,[entry])['charged_target_intents']==320
        passed.extend(('missing_closure_full_charge','broken_json_full_charge','valid_exact_closure','invalid_counter_full_charge'))
        good['host_returned']=8;(folder/'COUNTER-CLOSED.json').write_text(json.dumps(good))
        assert charged_reservations(root,[entry],failed_phase=True)['charged_target_intents']==320
        (folder/'FINISH_FAILED.json').write_text('{}')
        assert charged_reservations(root,[entry])['charged_target_intents']==320
        passed.extend(('valid_CLOSED_late_cleanup_failure_full_phase_charge','valid_CLOSED_worker_late_failure_full_reservation_charge'))
        q=object.__new__(SharedResources);q.job=root;q.p=p;q.deadline=None;q.guard=lambda:None;q.latest_foreign=[999]
        good=dict(memory_free_bytes=33*2**30,compute=[dict(pid=999)],graphics=[],memory_used_bytes=1,
                  utilization_percent=25)
        samples=iter([{**good,'memory_free_bytes':33*2**30-1}]+[good]*3);q.inspect=lambda:next(samples)
        with patch('shared_resource.time.sleep',lambda _:None):q.idle()
        assert len((root/'capacity-samples.jsonl').read_text().splitlines())==4
        passed.append('shared_three_consecutive_33GiB_capacity_samples_ignore_stale_utilization')
    from oracle_reuse import validate_oracles,validate_carry,authority
    assert validate_oracles(p)['required_new_full_chunks']==4 and not p['previous_native_oracles_current_proof']
    validate_carry(p)
    import copy
    for bad in ('new_checkpoint_sha256','aggregate_prior_target_intents'):
        q=copy.deepcopy(p);q[bad]='wrong' if isinstance(q[bad],str) else q[bad]-1
        try:validate_oracles(q) if bad=='new_checkpoint_sha256' else validate_carry(q)
        except ValueError:pass
        else:raise AssertionError('bad oracle/carry accepted')
    passed.extend(('new_four_native_OFF_ON_chunks_required_legacy_not_current_proof','candidate_substitution_rejected','carry_reset_rejected'))
    q=copy.deepcopy(p);q['retry_of']='fake-FAILED.json'
    try:authority(q)
    except ValueError:pass
    else:raise AssertionError('v4 automatic retry unexpectedly accepted')
    passed.append('v4_auto_retry_prohibited_without_new_reviewed_budget_and_chain')
    from supervisor import Campaign
    q=object.__new__(Campaign);q.job=job;q.p=p
    for v in p['assignments'][0]:
        e=q.environment('COLLECT',job/'campaigns/COLLECT/workers'/v['id'])
        assert e['DRAFT_TRAIN_INIT']=='17' and e['DRAFT_TRAIN_TASK']==str(v['task'])
        assert e['CUDA_VISIBLE_DEVICES']=='0' and e['MUJOCO_GL']=='egl' and e['WAM_P0_ACTION_DRAFT_B2_CHECKPOINT']==p['new_checkpoint']
    passed.append('actual_two_server_environments_and_checkpoint')
    import inspect
    fixed_source=inspect.getsource(Campaign.fixed)
    assert "'fixed_worker.py'" in fixed_source and 'new_native_cold_install_proof' in fixed_source and 'validate_oracles' not in fixed_source
    cold=json.loads((job/'preflight/CPU-COLD.json').read_text())
    teacher=json.loads((job/'preflight/CPU-COLLECTOR.json').read_text())
    assert cold['status']=='PASS_REAL_COLD_CTOR_INSTALL_ORIGINAL_RESET_PREDICT' and cold['constructor_not_new_or_already_reset_fixture'] and cold['server_geometry_never_privately_assigned_or_deleted']
    assert teacher['status']=='PASS_CPU_TEACHER_NO_MUTATION_FULL50_BF16' and len(teacher['pairs'])==6
    assert not cold['CUDA_initialized'] and not teacher['CUDA_initialized']
    passed.extend(('actual_cold_ctor_install_public_reset_predict_and_invalid_layout','six_real_CPU_teacher_full50_no_mutation_pairs','new_native_precheck_before_full_episodes'))
    for port in p['ports']:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    import disk_guard
    disk_guard.entry(shutil.disk_usage(job).free,p);disk_guard.runtime(shutil.disk_usage(job).free,0,p)
    passed.extend(('four_distinct_available_ports','actual_disk_entry_runtime'))
    try:plan(job,gpu=True)
    except (ValueError,FileNotFoundError):pass
    else:raise ValueError('GPU default gate unexpectedly open')
    passed.append('GPU_closed_without_root_authorization')
    assert not any('megaverifier' in str(k) or '/archives/' in str(k) for k in p['critical_source_sha256'])
    selection=str((OWN.parent/'bonus_acceptance_v2/selection.py').resolve())
    assert selection in p['critical_source_sha256'] and digest(selection)==p['critical_source_sha256'][selection]
    for dependency in ('bonus_e2e_v1/template.py','overnight_onpolicy_v1/candidate_predictor.py',
                       'fixed_load_cost/snapshot.py','draft_architecture_v1/models.py'):
        file=str((OWN.parent/dependency).resolve())
        assert file in p['critical_source_sha256'] and digest(file)==p['critical_source_sha256'][file]
    passed.append('fresh_source_manifest_excludes_withdrawn_operator_code')
    passed.append('dynamic_GPU_and_CPU_helper_sources_frozen')
    return passed

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);a=ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES')!='' or os.sched_getaffinity(0)!={8}:raise ValueError('CPU8 only')
    job=a.job.resolve();p=plan(job);passed=checks(job,p)
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1'}
    output=subprocess.run([VA,'-B',str(OWN/'cpu_runtime_check.py')],env=env,text=True,capture_output=True,check=True,timeout=180)
    (job/'preflight/REAL-IMPORT.stdout').write_text(output.stdout);(job/'preflight/REAL-IMPORT.stderr').write_text(output.stderr)
    smoke=json.loads(output.stdout.splitlines()[-1]);write(job/'preflight/REAL-IMPORT.json',smoke,exclusive=True)
    if smoke['status']!='PASS_REAL_TORCH_CPU_IMPORT_AND_OBJECTIVE' or smoke['CUDA_initialized']:raise ValueError('real import/objective check failed')
    code='import sys;sys.path.insert(0,'+repr(str(ROOT/'投机采样/scripts'))+');import p0_libero_multichunk_client;import libero.libero.benchmark;print("PASS_CLIENT_IMPORT")'
    with (job/'preflight/CLIENT-IMPORT.log').open('xb') as f:
        subprocess.run([CLIENT,'-B','-c',code],env={**env,'MUJOCO_GL':'egl','PYOPENGL_PLATFORM':'egl'},stdout=f,stderr=subprocess.STDOUT,check=True,timeout=60)
    plan(job)
    write(job/'preflight/CPU-GATE.json',dict(status='PASS',plan_sha256=digest(job/'plan.json'),checks=passed,
        actual_Torch_import_and_objective=smoke,actual_Torch_artifact_sha256=digest(job/'preflight/REAL-IMPORT.json'),
        collector_fixture_sha256=digest(job/'preflight/CPU-COLLECTOR.json'),actual_client_CPU_import=True,
        cold_ctor_fixture_sha256=digest(job/'preflight/CPU-COLD.json'),previous_GPU_oracles_not_current_proof=True,
        new_native_OFF_ON_required_before_episodes=True,
        CUDA_initialized=False,GPU_started=False,root_authorization_required=True),exclusive=True,durable=True)
    print(json.dumps(dict(status='PASS',checks=len(passed),plan_sha256=digest(job/'plan.json'),CPU_gate_sha256=digest(job/'preflight/CPU-GATE.json'))))

if __name__=='__main__':main()
