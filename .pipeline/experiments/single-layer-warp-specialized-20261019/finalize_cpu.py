"""Close M34 from actual receipts; no GPU imports, jobs, or model hashing."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parent
ROOT = B.parents[2]


def main():
    first = json.loads((B / 'RESULT.json').read_text())
    diagnostic = json.loads((B / 'DIAGNOSTIC-RESULT.json').read_text())
    remote_guard = json.loads((B / 'remote-evidence/FROZEN-REMOTE-FINAL.json').read_text())
    original = json.loads((B / 'FROZEN-CPU.json').read_text())
    files = []
    for item in original['files']:
        p = ROOT / item['path']
        value = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
        files.append(dict(path=item['path'], sha256=value,
                          unchanged=value == item['sha256']))
    guard = dict(at=datetime.now().astimezone().isoformat(),
                 all_unchanged=all(x['unchanged'] for x in files), files=files)
    (B / 'FROZEN-CPU-FINAL.json').write_text(json.dumps(guard, ensure_ascii=False, indent=2) + '\n')
    assert guard['all_unchanged']
    local = [json.loads((B / p / 'OWNED-RESULT.json').read_text())
             for p in ('local-correctness-1', 'local-correctness-2', 'local-gemm-inspect-1')]
    source = B.parents[2] / '算子研究/wam_ops/optimized/csrc/whole_layer.cu'
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source_hash == first['source_CUDA_SHA256']
    result = dict(
        at=datetime.now().astimezone().isoformat(),
        status='CLOSED_BOUNDED_PROTOTYPE_NUMERICAL_AND_PERFORMANCE_NO_GO',
        main_CUDA_SHA256=source_hash,
        complete_block_calls=diagnostic['total_M34_complete_block_calls'],
        complete_block_call_cap=600, further_calls_in_this_subplan=0,
        remote_first_GPU_worker_lifecycle_seconds=first['GPU_job_lifecycle_seconds'],
        remote_diagnostic_reserved_lifecycle_seconds=diagnostic['reservation_seconds'],
        remote_diagnostic_GPU_worker_lifecycle_seconds=diagnostic['GPU_worker_lifecycle_seconds'],
        local_job_lifecycle_seconds=sum(x['lifecycle_seconds'] for x in local),
        local_failed_before_block_lifecycle_seconds=local[0]['lifecycle_seconds'],
        GPU_CUDA_active_seconds='UNKNOWN; lifecycle/event samples are different boundaries',
        CPU_only_wait='separate from actual GPU reservation; source queue timestamps retained',
        unused_parent_hours='not transferred into another job; this 600-call subplan is closed',
        original_precision_admitted=False, production_admitted=False,
        all_local_owned_exited=all(x['owned_exited'] for x in local),
        all_remote_owned_reaped=first['all_GPU_roots_reaped'] and diagnostic['all_owned_reaped'],
        diagnostic_GPU0_empty=not diagnostic['GPU_contexts_after'],
        local_protected88_unchanged=guard['all_unchanged'],
        remote_deployed_protected86_unchanged=remote_guard['deployed_unchanged'],
        remote_original_88_check=False,
        remote_missing_protected_files_retained=first['source_guard']['remote_not_staged'],
        implementation_failures_retained=[
            'CPU build1: missing gemm_tile arguments',
            'CPU build2: private CUDA link.stub absent',
            'local attempt1: wrong Python dependency failure; zero block calls',
            'nine original output comparisons FAILED',
            'nine action legal KV write comparisons FAILED',
        ],
        CPU_post_guard_first_attempt=dict(returncode=1, new_GPU_calls=0,
            reason='read expected manifest from absent remote path; used preserved local remote-evidence manifest instead',
            stderr_not_persisted='UNKNOWN exact traceback; absent path observed independently'),
        timeout_jobs=0, new_robot_episodes=0, full30_calls=0,
        stopped=['padding/register/grid sweep', 'production integration',
                 '30-layer expansion', 'passing old numerical gate by tolerance relaxation'],
        four_quality_permissions=dict(engineering_quality_screen=False,
            independent_test_complete=False, noninferiority_established=False,
            formal_performance_permission=False),
    )
    (B / 'TERMINAL.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    text = [
        '# M34资源与止损', '',
        '终态：有界单层原型和诊断完成；原精度、性能均NO-GO。CUDA主源码固定df8b386…，新增布局仅为独立诊断副本。登记600完整block调用已全部使用，不再启动此轮GPU作业。', '',
        '| 口径 | 实际值 |', '|---|---:|',
        f'|远端首批3个GPU worker生命周期合计|{result["remote_first_GPU_worker_lifecycle_seconds"]:.3f}秒|',
        f'|远端profile/布局阶段GPU预留生命周期（含CPU编译）|{result["remote_diagnostic_reserved_lifecycle_seconds"]:.3f}秒|',
        f'|其中2个GPU worker生命周期合计|{result["remote_diagnostic_GPU_worker_lifecycle_seconds"]:.3f}秒|',
        f'|本机3个作业生命周期合计|{result["local_job_lifecycle_seconds"]:.3f}秒|',
        f'|其中首个依赖失败、0block|{result["local_failed_before_block_lifecycle_seconds"]:.3f}秒|',
        '|总CUDA active秒|UNKNOWN|', '|完整30层/机器人episode|0/0|', '',
        '不同生命周期边界没有相加伪装成CUDA active。CPU等空卡的时间在queue原时间戳中保留；原7小时父上限的未用时间不自动转授新作业。0timeout。两次CPU编译失败、一次本机依赖失败及全部数值失败保留。旧M32/M33及历史22缺失收据没有补造。', '',
        '本机88保护项最终核对不变。远端首批86已部署项不变，原88项检查因state/rules未迁入而为false，原Missing不修造；阶段诊断未改主源码/DSO。所有本轮owned root已退出或reap，01:44:39北京GPU0 context为空；GPU1仍是另一登记的Long计时，不是算子残留。', '',
        '止损方向：不继续布局/寄存器/grid搜索，不接入生产，不扩30层，不放宽旧精度门。下一步先评估设备GEMM/attention与固定Torch数学契约能否一致，再考虑带成熟流水的tile原语；不能把已有prompt缓存作为新增调度贡献。L2/DRAM/bank-conflict/achieved occupancy仍UNKNOWN。',
    ]
    (B / 'RESOURCE-AND-STOP-REPORT.md').write_text('\n'.join(text) + '\n')
    print(result['status'], result['complete_block_calls'], guard['all_unchanged'])


if __name__ == '__main__':
    main()
