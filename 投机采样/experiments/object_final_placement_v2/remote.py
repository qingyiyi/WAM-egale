"""Finite Object300 queue, admitted only after each original GPU chain closes."""
from pathlib import Path
import argparse
import os
import signal
import subprocess
import sys
import time
import traceback

OWN = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(OWN.parent))
from object_final_placement_v1.common import (
    ROOT, PERMISSIONS, read, write, require, stamp, atomic, limits,
    make_plan, controller_pin, phase_clean)
from baai_suite_pipeline_v1.runtime import environment
from tf_suite_pipeline_v1.runner import matches


def primary_closed(job):
    if not (job / 'CAMPAIGN-RESULT.json').is_file():
        return False
    result = read(job / 'CAMPAIGN-RESULT.json')
    groups = result['controller_processes']
    require(groups['scan_complete'] and not any(g['members'] for g in groups['groups']),
            'primary controller child groups not clean')
    for p in (job / 'stages').iterdir():
        if (p / 'LAUNCH-STARTED.json').is_file() and not phase_clean(p):
            return False
    return True



def released_primary_slot(job, slot):
    """Use a naturally freed final slot; original remaining final stays intact."""
    if (job / 'CAMPAIGN-RESULT.json').is_file():
        return set() if primary_closed(job) else None
    finals = [job / 'stages' / ('FINAL_' + str(i)) for i in (0, 1)]
    # Both fixed originals must actually have launched. Therefore there is no
    # undispatched original final that could race our memory admission.
    if not all((p / 'LAUNCH-RESULT.json').is_file() for p in finals):
        return None
    if not phase_clean(finals[slot]):
        return None
    other_models = set()
    for p in finals:
        if phase_clean(p):
            continue
        ready = p / 'server-worker0/READY.json'
        if not ready.is_file():
            return None
        pid = read(ready)['pid']
        pins = [read(x) for x in p.glob('OWNED-*.json')]
        if not any(x['pid'] == pid and matches(x) for x in pins):
            return None
        other_models.add(pid)
    return other_models


def gpu_metadata(index):
    r = subprocess.run(['nvidia-smi', '-i', str(index), '--query-gpu=uuid,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    require(r.returncode == 0, 'actual free-memory metadata unavailable')
    uuid, free = [s.strip() for s in r.stdout.strip().split(',')]
    r = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
    require(r.returncode == 0, 'actual model-process metadata unavailable')
    pids = set()
    for line in r.stdout.splitlines():
        fields = [s.strip() for s in line.split(',')]
        if len(fields) == 2 and fields[0] == uuid:
            pids.add(int(fields[1]))
    return uuid, int(free), pids


def run(job):
    job = job.resolve(strict=True)
    reg = read(job / 'registration.json')
    require(ROOT.is_relative_to((Path.home() / 'cxz').resolve(strict=True))
        and reg['permissions'] == PERMISSIONS and reg['target_forward_attempt_cap'] == 3600000,
        'only original Object900-new-episode remote budget inside ~/cxz')
    limits(reg)
    controller_pin(job, 'Object_final300_resource_dispatcher')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('original M17 stop')))
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(InterruptedError('owned controller stop')))
    signal.alarm(max(1, int(__import__('datetime').datetime.fromisoformat(reg['absolute_work_stop']).timestamp() - time.time())))
    active, finished, plans = {}, {}, []
    error = None
    try:
        while not (job / 'SELECTED-ON3000.json').is_file():
            limits(reg)
            atomic(job / 'status.json', dict(status='WAITING_ACTUAL_FIXED_OBJECT_ON3000',
                at=stamp(), GPU_models_started=0, permissions=PERMISSIONS))
            time.sleep(30)
        selected = read(job / 'SELECTED-ON3000.json')
        require(Path(selected['path']).resolve() == (job / 'checkpoint.pt').resolve()
            and Path(selected['path']).is_file(), 'actual transferred final checkpoint required')
        template = read(job / 'TEMPLATE.json')
        bank = Path(reg['bank'])
        require(bank.is_file(), 'real bound full512 bank required')
        for gpu in (0, 1):
            for slot in (0, 1):
                start = 200 + gpu * 150 + slot * 75
                folder = job / f'gpu{gpu}-slot{slot}'
                first = 16 + gpu * 8 + slot * 4
                aff = dict(server=[first, first + 1], client=[first + 2, first + 3], control=[40 + gpu])
                make_plan(reg, folder, template, selected, list(range(start, start + 75)), gpu, aff,
                    19860 + gpu * 2 + slot, 29860 + gpu * 2 + slot, 900000)
                plans.append(dict(job=folder, gpu=gpu, slot=slot))
        pending = list(plans)
        while pending or active:
            limits(reg)
            for name, item in list(active.items()):
                p = item['job']
                if phase_clean(p):
                    t = read(p / 'SUPERVISOR-RESULT.json')
                    finished[name] = dict(status=t['status'], complete_rows=t['complete_rows'],
                        planned_rows=t['planned_rows'], GPU=item['gpu'])
                    del active[name]
            for gpu in (0, 1):
                candidates = [p for p in pending if p['gpu'] == gpu]
                if not candidates:
                    continue
                primary_models = None
                item = None
                for candidate in candidates:
                    found = released_primary_slot(Path(reg['primary_campaigns'][str(gpu)]), candidate['slot'])
                    if found is not None:
                        item, primary_models = candidate, found
                        break
                if item is None:
                    continue
                same = [p for p in active.values() if p['gpu'] == gpu]
                if len(same) + len(primary_models) >= 2:
                    continue
                if any(not (p['job'] / 'server-worker0/READY.json').is_file() for p in same):
                    continue  # reserve one loading model until its real allocation/READY
                uuid, free, observed_pids = gpu_metadata(gpu)
                require(uuid == reg['GPU_UUIDs'][str(gpu)], 'fixed physical GPU assignment differs')
                allowed = primary_models | {read(p['job'] / 'server-worker0/READY.json')['pid'] for p in same}
                if free < 22000 or not observed_pids.issubset(allowed):
                    continue
                env = environment(reg, dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'))
                r = subprocess.run([reg['va_python'], '-B', str(OWN.parent / 'object_final_placement_v1/launch.py'), '--job', str(item['job'])],
                    cwd=OWN, env=env, capture_output=True, text=True, timeout=180)
                write(item['job'] / 'DISPATCH-LAUNCH-RETURN.json', dict(returncode=r.returncode,
                    stdout=r.stdout[-2000:], stderr=r.stderr[-2000:], at=stamp(), permissions=PERMISSIONS))
                require(r.returncode == 0 and (item['job'] / 'LAUNCH-RESULT.json').is_file(),
                    'new phase launch failed; no automatic retry')
                active[item['job'].name] = item
                pending.remove(item)
            atomic(job / 'status.json', dict(status='RUNNING_OR_WAITING_ORIGINAL_OBJECT300_PARTITIONS',
                at=stamp(), pending=[p['job'].name for p in pending], active=list(active),
                completed=finished, permissions=PERMISSIONS))
            time.sleep(20 if pending or active else 0)
        complete = all(x['status'] == 'COMPLETE_RECORDED_SUITE_PHASE'
            and x['complete_rows'] == x['planned_rows'] == 225 for x in finished.values())
        write(job / 'TERMINAL.json', dict(status='COMPLETE_REMOTE_OBJECT300_DESCRIPTIVE' if complete
            else 'PARTIAL_REMOTE_OBJECT300_NO_RETRY', at=stamp(), partitions=finished,
            no_quality_certificate=True, permissions=PERMISSIONS))
    except BaseException as exc:
        error = dict(error=repr(exc), traceback=traceback.format_exc())
        for item in active.values():
            flag = item['job'] / 'STOP_REQUESTED'
            if not flag.exists():
                flag.write_text('owned resource dispatcher failure; original independent guard cleanup\n')
        if not (job / 'TERMINAL.json').exists():
            write(job / 'TERMINAL.json', dict(status='PARTIAL_RESOURCE_DISPATCH_NO_AUTOMATIC_RETRY',
                at=stamp(), **error, completed=finished, active=list(active), permissions=PERMISSIONS))
        raise
    finally:
        write(job / 'CONTROLLER-EXIT.json', dict(at=stamp(), error=error,
            cleanup='each actual model has original independent phase guard; read phase cleanup receipts',
            no_foreign_processes_signalled=True, instance_shutdown='NOT_REQUESTED', permissions=PERMISSIONS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    run(parser.parse_args().job)
