"""Read-only aggregation of owned finite jobs; no GPU or model-file hashing."""
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = ROOT / '投机采样/results/runtime'
BASE = RUNTIME / 'p0-wam-shape-audit-v1-20261016-r1'


def main():
    os.sched_setaffinity(0, {14})
    out = Path(sys.argv[1]).resolve()
    assert out.parent == BASE and not out.exists()
    jobs = []
    known = [BASE / name for name in ('long-mechanism', 'full-matrix-r1',
        'full-matrix-r2', 'full-matrix-r3', 'full-matrix-r4', 'graph-matrix-r1',
        'ncu-linear-r1', 'ncu-linear-r2')]
    for prefix in ('p0-wam-quality-q02-replay-v1-20261016-r',
                   'p0-wam-same-seven-sequential-graph-v1-20261016-r',
                   'p0-wam-candidate-tile-build-v1-20261016-r',
                   'p0-wam-candidate-tile-numeric-v1-20261016-r',
                   'p0-wam-candidate-tile-fixture-v1-20261016-r',
                   'p0-wam-candidate-tile-guarded-v2-20261016-r',
                   'p0-wam-candidate-tile-graph-pilot-v1-20261016-r',
                   'p0-wam-candidate-tile-profile-v1-20261016-r',
                   'p0-wam-candidate-tile-chunk-correctness-v1-20261016-r',
                   'p0-wam-candidate-tile-chunk-correctness-v2-20261016-r',
                   'p0-wam-candidate-tile-chunk-correctness-budget-v1-20261016-r',
                   'p0-wam-static-prompt-baseline-v1-20261016-r',
                   'p0-wam-static-prompt-baseline-v2-20261016-r'):
        known += sorted(RUNTIME.glob(prefix + '*'))
    charge = 0.0
    for folder in known:
        if not folder.is_dir(): continue
        terminal = next((folder / n for n in ('SUPERVISOR-RESULT.json', 'RESULT.json', 'STARTUP-FAILED.json', 'CPU-PREPARATION-TERMINAL.json')
                         if (folder / n).is_file()), None)
        result = json.loads(terminal.read_text()) if terminal else {}
        seconds = next((result[k] for k in ('GPU_wall_budget_charge_sec',
            'GPU_wall_sec', 'gpu_wall_seconds', 'gpu_wall_budget_charge_seconds')
            if isinstance(result.get(k), (int, float))), None)
        # Build/CPU preparations have no GPU allocation and are not silently
        # converted into GPU time when a generic elapsed_seconds exists.
        if folder.name == 'p0-wam-quality-q02-replay-v1-20261016-r2':
            assert result.get('NVML_independent_outer_supervisor')
            seconds = result['elapsed_sec']
        if result.get('GPU_used') is False: seconds = 0.0
        if terminal and terminal.name == 'STARTUP-FAILED.json':
            assert not (folder / 'RUN-STARTED.json').exists()
            seconds = 0.0
        if seconds is not None: charge += seconds
        jobs.append(dict(path=str(folder), terminal=str(terminal) if terminal else None,
            status=result.get('status', 'FAILED_BEFORE_GPU_START' if terminal else 'NO_SUPERVISOR_TERMINAL_PRESENT'),
            reason=result.get('failure', result.get('error', result.get('reason'))),
            GPU_job_wall_charge_seconds=seconds,
            owned_clean=result.get('all_owned_live_exited', result.get('all_owned_processes_reaped', result.get('cleanup',
                result.get('own_process_reaped', result.get('owned_root_reaped', result.get('all_roots_reaped')))))),
            plan=str(folder/'PLAN.json') if (folder/'PLAN.json').is_file() else None,
            bytes=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())))
    guards = json.loads((ROOT / '.pipeline/maintenance/hw-upgrade-20261016/BEFORE.json').read_text())['immutable_guards']
    changed = []
    for item in guards:
        p = ROOT / item['path']
        if not p.is_file(): changed.append(dict(path=str(p), reason='missing')); continue
        stat = p.stat()
        if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
            changed.append(dict(path=str(p), reason='size/mtime differ'))
    result = dict(status='COMPLETE_CPU_JOB_INVENTORY',
        at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), jobs=jobs,
        summed_present_GPU_job_wall_charge_seconds=charge,
        timing_scope='Process lifetime charged to local GPU job budgets, not CUDA active time or robot speed; absent charge remains unknown.',
        AutoDL_accounted_separately=True, immutable_guard_count=len(guards),
        unchanged_size_mtime_guard_check=not changed, changed_guards=changed,
        no_large_model_hash=True, no_GPU_or_remote_actions=True)
    with out.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(dict(path=str(out), jobs=len(jobs), charged_seconds=charge,
                         unchanged_guards=not changed), ensure_ascii=False))


if __name__ == '__main__': main()
