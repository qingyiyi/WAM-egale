"""Read immutable completed-episode audits; no tensors/model/robot jobs."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

BASE = Path('/root/cxz/performance/libero-four-suite-20261019/isolated')


def read(p):
    return json.loads(p.read_text())


def main():
    os.sched_setaffinity(0, {4})
    jobs = read(BASE / 'JOBS.json') + read(BASE / 'LONG-JOBS.json')
    rows = []
    for job in jobs:
        for p in sorted((Path(job['job']) / 'episodes').glob('case-*/*/AUDIT.json')):
            terminal = p.with_name('EPISODE-TERMINAL.json')
            if not terminal.exists() or not read(terminal).get('complete_episode'):
                continue
            audit = read(p)
            first = audit.get('first2S', {})
            records = first.get('records', [])
            failed = [r for r in records if r.get('status') == 'FAILED']
            row = dict(suite=job['suite'], repetition=job['repetition'],
                       case_id=audit['case_id'], arm=audit['arm'], source=str(p),
                       component_status=audit['status'], first2_status=first.get('status'),
                       complete_two_S_observed=first.get('complete_two_S_observed'),
                       labels=[dict(label=r['label'], status=r['status']) for r in records])
            if failed:
                r = failed[0]
                comparison = r.get('active_input_comparison', r.get('comparison'))
                row.update(earliest_failed_label=r['label'],
                           earliest_comparison=comparison,
                           source_observation_only=True,
                           missing_raw_tensors_not_reconstructed=True)
            rows.append(row)
    source = Path('/root/cxz/WAM-egale/投机采样/experiments/tf_fullsuite_eval_v2')
    result = dict(at_UTC=datetime.now(timezone.utc).isoformat(), rows=rows,
                  no_new_GPU_calls=True, no_old_audit_changes=True,
                  source_sha256={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                                 for name in ('evidence.py', 'state.py')})
    (BASE / 'PAIRING-DIFFERENCES.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(rows=len(rows), failed=sum('earliest_failed_label' in r for r in rows))))


if __name__ == '__main__':
    main()
