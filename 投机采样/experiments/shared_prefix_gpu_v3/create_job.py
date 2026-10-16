"""CPU-only creation of a new, frozen shared-prefix one-shot job."""
import argparse
import json
from pathlib import Path
from task_contract import ROOT, OWN, BASE, REVISION, PREFIX, STAGES, stamp, write, digest, validate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--diagnostic-authorization',default='');a=ap.parse_args()
    previous=ROOT/'投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028'
    old=json.loads((previous/'plan.json').read_text())
    fail=json.loads((previous/'FAILED.json').read_text())
    if fail['budget']['target_attempted']!=183 or fail['cleanup']['status']!='PASS':raise ValueError('verified old consumption/cleanup required')
    frozen={**old['frozen_sha256'],**old['implementation_sha256']}
    raw=json.loads((ROOT/'.pipeline/research/speculative-sampling/evidence/shared-prefix-implementation-20261012/RAW-BASELINE-FREEZE.json').read_text())
    frozen.update(raw)
    prior_cpu=json.loads((ROOT/'.pipeline/research/speculative-sampling/evidence/shared-prefix-implementation-20261012/V1-CPU-FAILED-FREEZE.json').read_text())
    frozen.update(prior_cpu)
    frozen.update(json.loads((ROOT/'.pipeline/research/speculative-sampling/evidence/shared-prefix-implementation-20261012/V2-CPU-PASS-RAW-FREEZE.json').read_text()))
    frozen.update(json.loads((ROOT/'.pipeline/research/speculative-sampling/evidence/shared-prefix-implementation-20261012/V2-SOURCE-FREEZE.json').read_text()))
    for f in (ROOT/'.pipeline/knowledge/records').glob('*.yaml'):frozen[str(f)]=digest(f)
    report=ROOT/'.pipeline/research/speculative-sampling/SHARED-PREFIX-CFG2-PLAN-20261012.md'
    frozen[str(report)]=digest(report)
    for f,h in frozen.items():
        if digest(f)!=h:raise ValueError('historical evidence/source changed '+f)
    implementation={str(f):digest(f) for f in OWN.iterdir() if f.is_file() and f.suffix in ('.py','.mjs','.md')}
    name=PREFIX+stamp().replace('-','').replace(':','').replace('T','-')[:15]
    job=ROOT/'投机采样/results/runtime'/name;job.mkdir()
    for rel in ('preflight','worker/inbox','worker/outbox','results','events','numerical','traces'):
        (job/rel).mkdir(parents=True,exist_ok=True)
    p={**old, 'job':str(job),'created_at':stamp(),'status':'AUTHORIZED_CPU_PENDING',
       'execution_revision':REVISION,'lane':'speculative-sampling NOT tile',
       'authorization_source':'user explicitly requested CFG retained and seven-node shared KV implementation/performance test',
       'authorization_text':'那cfg暂时先不共享，只共享7份节点的kv，你先把这个做成功，测试下性能好吗；结果不一致你可以分析一下为什么结果不一致吗？',
       'hard_target_forward_cap':715,'GPU_hard_limit_minutes':36,'stage_caps':STAGES,
       'main_orders':[['R','S'],['S','R']]*3,'ports':[20046],
       'frozen_sha256':frozen,'implementation_sha256':implementation,
       'aggregate_prior_target_intents':183,'aggregate_prior_GPU_seconds':507.2913444358856,
       'aggregate_GPU_forward_budget_this_continuation':900,'aggregate_GPU_time_limit_minutes_this_continuation':45,
       'diagnostic_nonbitwise_approved':bool(a.diagnostic_authorization),
       'diagnostic_authorization_text':a.diagnostic_authorization,
       'direct_attention_micro_call_cap':256,'CFG_history_rows':2,'candidate_current_rows':14,
       'S_history14_or_fullKV14_allowed':False,'S_original_pool_zero_copy':True,
       'kernel_diagnostic_ceiling_is_not_target_acceptance':True,
       'success_environment_steps_E2E_NOT_PROVEN':True,'old_failed_job_not_reused':True,
       'CPU_gate_summarization_erratum':'initial CPU-tests-001 was22, not early HW summary24; subsequent new source tests are recorded independently',
       'unallocated130_are_not_retry_permission':True}
    validate(p);write(job/'plan.json',p,exclusive=True)
    (job/'BRIEF.md').write_text('# 七候选共享历史KV：CPU预检待执行\n\n未启动GPU、未训练；CFG保留两路，S无history14/fullKV14。\n')
    write(job/'events/AUTH.json',dict(job=str(job),event='AUTH',timestamp=stamp(),GPU_started=False,
        budget=dict(target_reserved=0,target_attempted=0,target_completed=0),passed_gates=[],
        result=dict(status='AUTHORIZED_CPU_PENDING',diagnostic_nonbitwise_approved=p['diagnostic_nonbitwise_approved'],
            next_step='新job先通过CPU与同worker生产loader/计数预检；之后GPU空闲时一次启动，累计900/45分钟不重置。数学/逐位门分别报告，严格门未过且无诊断授权即停；原始全29输出及同QKV逐层归因保留。')),exclusive=True)
    print(job)


if __name__=='__main__':main()
