"""Create a fresh authorized job, not a launcher or GPU probe."""
from pathlib import Path
import json
from task_contract import ROOT,OWN,REVISION,PREFIX,STAGES,stamp,write,digest,validate


def main():
    proposal=ROOT/'.pipeline/research/speculative-sampling/evidence/single-worker-chc-plan-20261012/PLAN.json'
    old=json.loads(proposal.read_text())
    baseline_file=ROOT/'.pipeline/research/speculative-sampling/evidence/single-worker-chc-gpu-v4-entry-20261012/BASELINE.json'
    before=json.loads(baseline_file.read_text())
    runtime_old=json.loads((ROOT/'投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/plan.json').read_text())
    frozen={**before['baseline_sha256'],**before['HW_raw_record_baseline_sha256'],str(baseline_file):digest(baseline_file)}
    for f,h in frozen.items():
        if digest(f)!=h:raise ValueError('historical source/protected changed: '+f)
    # Also preserve the CPU-result evidence, without freezing writable HW views.
    for relative in ['SINGLE-WORKER-CHC-CPU-RESULT-20261012.md',
                     'evidence/single-worker-chc-cpu-20261012/CPU-RESULT.json',
                     'evidence/single-worker-chc-cpu-20261012/CPU-tests-004.json',
                     'evidence/single-worker-chc-cpu-20261012/SOURCE-AUDIT.json',
                     'evidence/single-worker-chc-cpu-20261012/validation.json']:
        path=ROOT/'.pipeline/research/speculative-sampling'/relative;frozen[str(path)]=digest(path)
    implementation={str(f):digest(f) for f in OWN.iterdir() if f.is_file() and f.suffix in ('.py','.mjs','.md')}
    name=PREFIX+stamp().replace('-','').replace(':','').replace('T','-')[:15]
    job=ROOT/'投机采样/results/runtime'/name;job.mkdir()
    for rel in ['preflight','worker/inbox','worker/outbox','results','events','traces']:(job/rel).mkdir(parents=True,exist_ok=True)
    p={k:old[k] for k in ['batch','layers','rows','bonus','selector','threshold','cache_mode','frozen_A_off',
        'frozen_A_off_sha256','snapshot_reference','main_orders']}
    p.update(schema_version=1,job=str(job),created_at=stamp(),lane='speculative-sampling NOT tile',
        status='AUTHORIZED_CPU_PENDING',gpu_run_approved=True,execution_revision=REVISION,
        authorization_source='user: repair until GPU verification passes; background private bugfix plus read-only shape subagent while GPU is idle',
        authorization_text=before['authorization_text'],
        training=False,RL=False,joint_training=False,GPU_model_worker_cap=1,GPU_attempt_cap=1,
        hard_target_forward_cap=900,GPU_hard_limit_minutes=45,new_full_episode_cap=0,EGL_initialization_cap=0,
        complete_policy_chunk_cap=0,executed_environment_step_cap=0,CPU_preflight_required=True,no_retries=True,
        no_resident_peer=True,stage_caps=STAGES,model_python=runtime_old['model_python'],worktree=runtime_old['worktree'],
        frozen_reference=runtime_old['frozen_reference'],ports=[20045],
        frozen_sha256=frozen,implementation_sha256=implementation,target_weight_inventory=before['target_weight_inventory_stat_only'],
        proposal_file=str(proposal),baseline_file=str(baseline_file),unallocated5_are_not_retry_permission=True,
        stable_idle_confirmation_max_seconds=180,resource_wait_explicitly_authorized=True,resource_poll_interval_sec=180,
        resource_wait_not_GPU_timer=True,no_unapproved_long_resource_queue=True,cleanup_reserve_seconds=15,
        GPU_timer_includes_first_model_spawn_through_cleanup=True,old_GPU_FAIL_unchanged=True,
        counter_persistence_enabled=True,GPU_compute_mode_changed=False,
        counter_cost_charged_inside_each_R_C_HC_forward_span=True,
        counter_semantics='write-ahead budget intents / confirmed host forward returns, not CUDA completion',
        shared_Default_mode_not_a_hardware_reservation=True,
        aggregate_GPU_forward_budget_this_continuation=900,
        aggregate_GPU_time_limit_minutes_this_continuation=45,
        aggregate_GPU_attempts_already_used_this_continuation=0,
        previous_v3_failed_at_CPU_not_reused=True,
        shape_explanation_subagent_read_only=True)
    validate(p);write(job/'plan.json',p,exclusive=True)
    write(job/'SOURCE-FREEZE.json',dict(frozen_sha256=frozen,implementation_sha256=implementation,
        target_weight_inventory_stat_only=p['target_weight_inventory'],old_GPU_FAIL_not_rewritten=True),exclusive=True)
    (job/'tested-source').mkdir()
    for f in OWN.iterdir():
        if str(f) in implementation:(job/'tested-source'/f.name).write_bytes(f.read_bytes())
    (job/'BRIEF.md').write_text('# 投机采样：单worker C/HC v4\n\n'+stamp()+
        '\n\n状态：新job已授权，CPU预检待执行；GPU尚未启动。1 worker、900前向硬限、45分钟含加载清理、0episode/EGL/full policy chunk/训练。共享模式不改。\n\n下一步：新job CPU预检→资源确认→一次GPU分关验证；失败停、不补跑。\n')
    event=dict(job=str(job),event='AUTH',timestamp=stamp(),GPU_started=False,budget=dict(target_reserved=0,target_attempted=0,target_completed=0),
        passed_gates=[],result=dict(status='PASS_AUTHORIZATION_IMPLEMENTATION_PENDING_CPU',
            next_step='按用户持续修复及并行讲解授权：只修私有负例flag隔离/路径断言，增加真实CLI导入前拒绝证明；新job CPU关通过后自动检查空闲并启动一次GPU，累计仍≤900前向/45分钟。共享模式不改，不训练/新prefix kernel/公共模型改动；旧FAIL不重写。'))
    write(job/'events/AUTH.json',event,exclusive=True);print(job)


if __name__=='__main__':main()
