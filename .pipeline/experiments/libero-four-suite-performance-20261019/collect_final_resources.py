"""Read-only closure observation: PID birth identities, guards and GPU contexts."""
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import subprocess

BASE = Path('/root/cxz/performance/libero-four-suite-20261019/isolated')
OP = Path('/root/cxz/ops/whole-layer-warp-20261020')


def read(p):
    return json.loads(p.read_text())


def observe(pin, source):
    if not isinstance(pin, dict) or 'pid' not in pin or 'birth_ticks' not in pin:
        return None
    result = dict(source=source, pid=pin['pid'], expected_birth_ticks=pin['birth_ticks'])
    try:
        p = Path(f'/proc/{pin["pid"]}')
        fields = (p/'stat').read_text().rsplit(')', 1)[1].split()
        same = int(fields[19]) == pin['birth_ticks']
        if 'uid' in pin:
            same = same and p.stat().st_uid == pin['uid']
        result.update(exists=True, same_identity=same, state=fields[0],
                      matching_active_identity=same and fields[0] != 'Z')
    except (FileNotFoundError, ProcessLookupError):
        result.update(exists=False, same_identity=False, matching_active_identity=False)
    return result


def main():
    os.sched_setaffinity(0, {4})
    status = read(BASE/'LONG-QUEUE-STATUS.json')['status']
    assert status in ('COMPLETE_ISOLATED_LONG_ENGINEERING_MEASUREMENT',
                      'STOPPED_OR_PARTIAL_ISOLATED_LONG'), 'Long still active; no closure declaration'
    jobs = read(BASE/'JOBS.json') + read(BASE/'LONG-JOBS.json')
    identities = []
    guards = []
    for item in jobs:
        job = Path(item['job'])
        p = job/'GUARD-FINAL.json'
        guards.append(dict(suite=item['suite'], repetition=item['repetition'],
                           guard_status=read(p).get('status') if p.exists() else 'MISSING'))
        files = list(job.glob('OWNED-*.json')) + [job/'GUARD-PROCESS.json', job/'LAUNCH-PARENT-OWNED.json']
        for p in files:
            if p.exists():
                observed = observe(read(p), str(p))
                if observed:
                    identities.append(observed)
    for name in ('GPU0-QUEUE-OWNED.json', 'GPU1-QUEUE-OWNED.json',
                 'LONG-QUEUE-OWNED.json', 'OBJECT-CONTINUATION-OWNED.json'):
        p = BASE/name
        if p.exists():
            observed = observe(read(p), str(p))
            if observed:
                identities.append(observed)
    for p, key in ((OP/'GPU-QUEUE-STATUS.json', 'owned_queue'),
                   (OP/'GPU-QUEUE-STATUS-before-release-fix.json', 'owned_queue'),
                   (OP/'diagnostic/QUEUE-STATUS.json', 'owned')):
        if p.exists():
            observed = observe(read(p).get(key), str(p))
            if observed:
                identities.append(observed)
    for p in list(OP.glob('*-OWNED.json')) + list((OP/'diagnostic').glob('*-OWNED.json')):
        observed = observe(read(p), str(p))
        if observed:
            identities.append(observed)
    def smi(fields):
        return subprocess.check_output(['nvidia-smi', fields, '--format=csv,noheader,nounits'],
                                       text=True, timeout=15)
    devices = list(csv.reader(io.StringIO(smi('--query-gpu=index,uuid,memory.used,utilization.gpu'))))
    contexts = list(csv.reader(io.StringIO(smi('--query-compute-apps=pid,gpu_uuid,used_gpu_memory'))))
    usage = shutil.disk_usage('/root/cxz')
    storage = {}
    for p in (BASE, OP):
        raw = subprocess.check_output(['du', '-sb', str(p)], text=True, timeout=30)
        storage[str(p)] = int(raw.split()[0])
    result = dict(
        at_UTC=datetime.now(timezone.utc).isoformat(), long_queue_status=status,
        GPU_query_fields=['index', 'uuid', 'memory_used_MiB', 'utilization_percent'],
        GPUs=devices, compute_context_fields=['pid', 'gpu_uuid', 'used_memory_MiB'],
        compute_contexts=contexts, compute_query_scope='NVML compute contexts; not a proof about every graphics workload',
        guards=guards, PID_identity_observations=identities,
        matching_active_owned_identities=[r for r in identities if r['matching_active_identity']],
        all_12_registered_phase_guards_clean=all(g['guard_status'] == 'CLEAN' for g in guards),
        registered_PID_observations=len(identities), storage_bytes=storage,
        cxz_filesystem_total_bytes=usage.total, cxz_filesystem_free_bytes=usage.free,
        processes_signalled=0, model_forwards=0, GPU_contexts_created=0,
        no_old_cleanup_receipt_rewritten=True, instance_shutdown='NOT_REQUESTED_NOT_EXECUTED',
    )
    (BASE/'FINAL-RESOURCE-OBSERVATION.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(active_owned=len(result['matching_active_owned_identities']),
                          compute_contexts=len(contexts), guards_clean=result['all_12_registered_phase_guards_clean'])))


if __name__ == '__main__':
    main()
