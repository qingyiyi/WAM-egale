"""CPU-only fresh plan; reuses audited evidence, never edits old plans."""
import argparse
import json
from pathlib import Path
import socket

from task_contract import ROOT, RESEARCH, OWN, PREFIX, REVISION, implementation, digest, write, stamp

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    args=ap.parse_args();job=args.job.resolve()
    if job.parent!=RESEARCH/'results/runtime' or not job.name.startswith(PREFIX):
        raise ValueError('fresh private job')
    job.mkdir()
    old=RESEARCH/'results/runtime/p0-prompt-kv-gpu-v4-20261013-011114'
    p4=json.loads((old/'plan.json').read_text())
    E=ROOT/'.pipeline/research/speculative-sampling/evidence/prompt-kv-cpu-20261012'
    audit=json.loads((E/'FINAL-AUDIT.json').read_text())
    if (audit['status']!='PASS_FINAL_PROMPT_GPU_HW_HISTORY_BYTES_BUDGET_AND_CLEANUP_AUDIT'
            or audit['aggregate_target_intents']!=6641
            or audit['aggregate_GPU_seconds']!=4994.309834191576 or not audit['cleanup_pass']):
        raise ValueError('audited carried GPU budget required')
    frozen={**p4['frozen_sha256'],**p4['implementation_sha256'],
            **json.loads((E/'NEW-EVIDENCE-FREEZE.json').read_text())['files']}
    reference=RESEARCH/'results/runtime/p0-speculative-e2e-v1-20261013-084008'
    prior=json.loads((reference/'plan.json').read_text())
    carry=json.loads((reference/'RECONCILED-CLEANUP.json').read_text())
    if (carry['status']!='PASS_FAILED_JOB_OWNED_ORPHAN_CLEANUP'
            or carry['aggregate_target_intents']!=17551 or carry['aggregate_GPU_seconds']!=5705.285101825371):
        raise ValueError('v1 failed-job reconciliation required')
    frozen.update(prior['implementation_sha256'])
    for file in reference.rglob('*'):
        if file.is_file():frozen[str(file)]=digest(file)
    skipped=RESEARCH/'results/runtime/p0-speculative-e2e-v2-20261013-090115'
    diskfail=json.loads((skipped/'FAILED.json').read_text())
    if diskfail['new_GPU_seconds'] or diskfail['charged_target_intents'] or diskfail['model_launches']:
        raise ValueError('preserved v2 must be a zero-GPU startup gate failure')
    frozen.update(json.loads((skipped/'plan.json').read_text())['implementation_sha256'])
    for file in skipped.rglob('*'):
        if file.is_file():frozen[str(file)]=digest(file)
    extra=[old/'plan.json',old/'FINAL_RESULT.json',E/'FINAL-AUDIT.json',E/'NEW-EVIDENCE-FREEZE.json',
           ROOT/'.pipeline/research/speculative-sampling/PROMPT-KV-GPU-RESULT-20261013.md',
           ROOT/'.pipeline/state.yaml',ROOT/'.pipeline/rules.yaml',
           RESEARCH/'scripts/p0_libero_multichunk_client.py',
           RESEARCH/'scripts/p0_live_device_egl_audit.py',
           OWN.parent/'fixed_load_cost/snapshot.py',OWN.parent/'prompt_kv_gpu_v4/exit_guard.py',
           Path(p4['frozen_A_off'])]
    for file in extra:frozen[str(file)]=digest(file)
    # Mutable derived HW surfaces are append governed, not immutable inputs.
    for path in frozen:
        if Path(path).name in ('PROGRESS.md','RESEARCH.md','log.md','log.yaml','knowledge.compact.md'):
            raise ValueError('mutable HW surface accidentally frozen')
    ref=dict(p4['frozen_reference'])
    ref.update(draft_checkpoint=p4['frozen_A_off'],draft_checkpoint_sha256=p4['frozen_A_off_sha256'],
               status='new full-episode plan; latest trained frozen A_off and original target')
    ports=[19731,19732,29671,29672,29673]
    for port in ports:
        with socket.socket() as s:s.bind(('127.0.0.1',port))
    p=dict(job=str(job),created_at=stamp(),execution_revision=REVISION,
           arms=['D','S'],inits=[3,4],full_episode_cap=4,EGL_initialization_cap=4,
           parallel_EGL_workers=2,model_worker_cap=4,no_retries=True,training=False,RL=False,
           trace_control_changed=False,profile_OFF=True,full_trace_OFF=True,teacher_OFF=True,
           prompt_shared=False,rowproof_reuse=False,per_episode_forward_cap=4000,probe_forward_cap=0,
           GPU_hard_limit_seconds=5400,aggregate_target_cap=35000,aggregate_GPU_seconds_cap=18000,
           aggregate_prior_target_intents=17551,aggregate_prior_GPU_seconds=5705.285101825371,
           disk_initial_min_bytes=16*2**30,disk_runtime_floor_bytes=12*2**30,
           new_output_bytes_cap=2**30,
           startup_disk_gate_decision='fresh four-episode outputs capped at 1 GiB, reserve 16 GiB; previous 40 GiB zero-GPU gate failure preserved',
           reference_target_job=str(reference),reference_target_episodes=2,full_episode_data_total=6,
           reused_numerical_probe=str(reference/'probe/PROBE-RESULT.json'),
           previous_aggregate_target_cap=35000,new_target_cap_decided_by='unchanged finite cap; autonomous four-episode completion',
           frozen_A_off=p4['frozen_A_off'],frozen_A_off_sha256=p4['frozen_A_off_sha256'],
           frozen_reference=ref,target_weight_inventory=p4['target_weight_inventory'],
           snapshot_reference=p4['snapshot_reference'],frozen_sha256=frozen,
           implementation_sha256=implementation(),ports=ports,
           primary_metric='client.elapsed_sec: reset/prompt/all model calls/environment to true success or horizon',
           secondary_metric='outer_episode_wall_sec includes env construction and close',
           excluded_time='model bootstrap, final output/accounting drain and offline comparisons',
           pair_init_and_client_server_seed=20260922,seed_before_env_construction=True,
           complete_LIBERO_horizon=800,stabilization_steps=5,
           nominal_trace_length_ratio_goal=1.05,trace_length_goal_decided_by='agent interpretation of approximately baseline',
           two_init_pilot_only=True,old_evidence_and_failures_preserved=True,
           production_source_modified=False,counter_per_forward_disk_IO=False,
           conservative_full_reservation_charge_if_no_terminal_closure=True)
    write(job/'plan.json',p,exclusive=True,durable=True)
    write(job/'AUTHORIZATION.json',dict(authorized=True,plan_sha256=digest(job/'plan.json'),
         timestamp=stamp(),GPU_seconds_cap=18000,new_GPU_seconds_cap=5400,aggregate_target_cap=35000,
         no_repeat_permission_required=True,
         user_text=['我先睡觉了，你确认cpu没问题以后，直接把gpu的也做到没问题，然后给我汇报结果，是否有加速效果，是正向优化还是负优化。中途你自己决定就行，我只要结果',
                    'gpu消耗上限你可以暂时拓展到5小时，凌晨没人用',
                    '你整理一下，我们先做到这里，目前能确定单chunck确实能够有加速了，我们先测一个e2e的投机加速比，然后来研究如何控制trace长度吧，至少和原来差不多',
                    '完整任务/episode（推荐）；完整任务轨迹的环境步数'],
         agent_decision='fresh four-episode continuation with two immutable target episodes reused; cumulative time remains five hours; no prior plan modified'),
         exclusive=True,durable=True)
    print(json.dumps(dict(job=str(job),plan_sha256=digest(job/'plan.json'),frozen_files=len(frozen)),ensure_ascii=False))

if __name__=='__main__':main()
