"""Write the final first-round evidence packet after the existing Long queue closes.

CPU-only, bounded by the original M16 deadline. Never launches or retries an
episode. The separate 09:15 snapshot remains available if Long finishes later.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import signal
import sys
import time
import traceback

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from baai_approved_retry_v2.morning_report import build, read, write, ARMS, PERMISSIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    p = read(job / 'REGISTRATION.json')
    workspace = (Path.home() / 'cxz').resolve(strict=True)
    assert job.is_relative_to(workspace)
    os.sched_setaffinity(0, {43})
    deadline = datetime.fromisoformat(p['cleanup_at']).timestamp()
    def stop(*_):
        raise TimeoutError('original M16 final-report deadline or owned controller stop')
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGALRM, stop)
    signal.setitimer(signal.ITIMER_REAL, max(.1, deadline-time.time()))
    write(job / 'STARTED.json', dict(pid=os.getpid(), at=time.time(), CPU=[43], GPU=0))
    try:
        queue = Path(p['wait_for_queue'])
        while not (queue / 'TERMINAL.json').exists():
            if (job / 'STOP_REQUESTED').exists():
                stop()
            if time.time()+300 >= deadline:
                break
            time.sleep(20)
        result = build(p, job / 'report')
        report = job / 'report'
        (report / 'QUALITY-RESULT.md').write_text((report / 'LONG-RESULT.md').read_text())
        cases = result['complete_common_cases']
        facts = ['# 事实、未知与建议', '',
            f"OBSERVED：Long首轮共同完整{cases}/500组，{3*cases}/1500局。", 
            'OBSERVED：固定TF_ON同权重对照T/C/P，全臂优化reset；有效自然失败留在分母和800罚分。',
            'OBSERVED：原基础设施PARTIAL未改为成功，指定替代attempt不增加初态样本。',
            'UNKNOWN：独立TEST、非劣性、正式独占E2E加速；当前四项权限均false。',
            'UNKNOWN：三轮均值与SD，后两轮未运行；本轮含训练/DEV暴露。',
            'RECOMMENDATION：按全部质量门和配对风险决定投入；不以总体SR抵消长尾和审计FAILED。', '']
        (report / 'FACTS.md').write_text('\n'.join(facts))
        system = ['# 系统成本：探索性结果', '',
            '每个执行代单列ΣT_round/Σg，含失败局；B2/B14调用数不能按相同成本相加。',
            '并发/硬件不同，以下不是正式speedup；原独占性能入口仍未放行。', '',
            '| 执行源 | 臂 | N | 完整计量局 | ΣT_round秒 | Σg | 秒/有效提交 |',
            '|---|---|---:|---:|---:|---:|---:|']
        for source, methods in result['hardware_execution_strata'].items():
            for arm in ARMS:
                row = methods[arm]
                system.append(f"|{Path(source).name}|{arm}|{row['n']}|{row['measured_round_coverage_episodes']}|{row['sum_T_round_sec']}|{row['sum_g']}|{row['cost_per_valid_commit_exploratory']}|")
        system += ['', '完整episode/reset/policy/cache的逐局值见EPISODES.csv；全attempt与缺项见SOURCE-MANIFEST。',
                   '独占单chunk、单task和E2E正式加速：UNKNOWN。缓存/批处理旧收益不是新MegaTransformer贡献。', '']
        (report / 'SYSTEM-RESULT.md').write_text('\n'.join(system))
        (report / 'CLAIM-EVIDENCE-MATRIX.md').write_text('''# 论文主张与证据

| 主张 | 当前证据 | 缺口/允许程度 |
|---|---|---|
| 本轮固定Long面板的描述性SR/长度/罚分 | RESULT.json、EPISODES.csv及原始完整终态 | 可按实际N和暴露说明报告，不能称独立TEST |
| TF一定优于MLP | 本轮无matched MLP三臂全量 | 不允许 |
| PF2改善全部风险 | 配对得失、长尾和旧NO_GO均保存 | 只能按实际结果陈述，不预设改善 |
| 工程质量/非劣性已成立 | 四项false、旧质量门未改 | 不允许 |
| 减少target calls即E2E提速 | 有ΣT_round/Σg和共享墙钟 | 不允许；独占完整任务仍缺 |
| 超过最强B14的新MegaTransformer收益 | 本作业没有新引擎对照/NCU | 不允许；旧系统证据另列 |
| 三轮稳定性和1500独立初态 | 只一轮，同500初态 | 不允许；三轮也不能改称1500独立初态 |
''')
        attempt_gaps = []
        closures = []
        for source in result['source_roots']:
            source = Path(source)
            closed = (source / 'FINAL-OWNED-CLEANUP.json').exists()
            close = read(source / 'FINAL-OWNED-CLEANUP.json') if closed else {}
            closures.append(dict(source=str(source), owned_cleanup_observed=closed,
                                 all_owned_groups_empty=close.get('all_owned_groups_empty')))
            for folder in source.glob('episodes/case-*/*'):
                if folder.is_dir() and folder.name in ARMS and not (folder / 'EPISODE-TERMINAL.json').exists():
                    attempt_gaps.append(dict(source=str(folder),
                        status='CLOSED_ATTEMPT_WITHOUT_EPISODE_TERMINAL' if closed else 'IN_PROGRESS_OR_UNKNOWN',
                        success='UNKNOWN', missing_receipt_not_fabricated=True))
        write(report / 'INCOMPLETE-ATTEMPT-INDEX.json', dict(
            terminal_based_partial_count=result['partial_attempts'],
            missing_episode_terminals=attempt_gaps, closures=closures,
            approved_infrastructure_case_ids=[219,220,221,222,223,225,226,227],
            these_are_not_new_independent_cases=True))
        attempts = result['all_generation_actual_attempts_at_snapshot']
        resource = ['# 资源与停止报告', '',
            f"OBSERVED：各代LIVE累计实际target尝试{attempts}；原6001000帽剩余{6001000-attempts}。",
            'B2/B14调用成本不同；CUDA active小时UNKNOWN，不用生命周期或CUDA-event区间和替代。',
            f"原停新：{result['original_absolute_work_stop']}；清理：{result['original_cleanup']}，未重置。",
            f"完整共同case：{cases}/500；仅有原始PARTIAL终态的尝试数{result['partial_attempts']}。",
            '缺少终态的旧失效attempt与当前在执行尝试另见INCOMPLETE-ATTEMPT-INDEX，不补造收据。',
            '各旧新Long自有进程清理以原receipt为准；本报告不声称Object或整个服务器GPU已清空。',
            '止损：不新训Long、不搜PF3/4/阈值/架构、不启动后两轮/TEST/正式性能；旧NO_GO保持。',
            '错误冗余产物按已批准的依赖清理规则处理，不删除有效自然失败。', '']
        (report / 'RESOURCE-AND-STOP-REPORT.md').write_text('\n'.join(resource))
        write(job / 'TERMINAL.json', dict(status='FINAL_FIRST_ROUND_EVIDENCE_PACKET_WRITTEN',
            report_status=result['status'], complete_common_cases=cases, at=time.time(),
            permissions=PERMISSIONS, no_additional_GPU_jobs=True))
    except BaseException as exc:
        write(job / 'TERMINAL.json', dict(status='PARTIAL_OR_FAILED_CPU_FINAL_REPORT',
            error=repr(exc), traceback=traceback.format_exc(), at=time.time(), permissions=PERMISSIONS))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        write(job / 'CLEANUP.json', dict(status='CPU_FINAL_REPORTER_EXITS', at=time.time(),
            no_GPU_started=True, no_foreign_processes_signalled=True))


if __name__ == '__main__':
    main()
