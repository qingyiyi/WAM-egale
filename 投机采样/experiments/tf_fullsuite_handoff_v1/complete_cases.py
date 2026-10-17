"""One bounded, birth-pinned orchestration handoff after both complete cases; unchanged pending arms finish first.

Pause only the CPU runner when its two clients are already in their registered
registered arm. Clients and servers continue complete natural episodes.
Drain their existing FINISH protocol, then close idle owned servers. No new
model call, environment command, retry, or old scientific receipt is created.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def write(path, row):
    with Path(path).open('x') as f:
        json.dump(row, f, ensure_ascii=False, indent=2)
        f.write('\n')


def proc(pid):
    try:
        base = Path('/proc')/str(pid)
        a = (base/'stat').read_text().rpartition(') ')[2].split()
        return dict(pid=pid, state=a[0], parent=int(a[1]), pgid=int(a[2]),
            birth_ticks=int(a[19]), uid=base.stat().st_uid)
    except (OSError, ValueError, IndexError):
        return None


def matches(pin):
    p = proc(pin['pid'])
    return p is not None and p['birth_ticks'] == pin['birth_ticks'] and p['pgid'] == pin['pgid'] == pin['pid'] and p['uid'] == pin.get('uid', os.getuid())


def arg(argv, flag):
    return argv[argv.index(flag)+1]


def clients(job, parent):
    active = {}
    for f in job.glob('OWNED-*.json'):
        pin = read(f)
        if pin.get('owner_job') != str(job) or not pin.get('role','').startswith('client-case'):
            continue
        if not matches(pin) or proc(pin['pid'])['state'] in ('Z','X'):
            continue
        argv = pin['argv']
        worker = int(arg(argv,'--worker-index'))
        if worker in active:
            return {}
        active[worker] = dict(pin=pin, case_id=int(arg(argv,'--case-id')), arm=arg(argv,'--arm'))
    # Do not suspend between Popen and the immutable lifetime receipt.
    known = {v['pin']['pid'] for v in active.values()}
    for f in Path('/proc').iterdir():
        if not f.name.isdigit():
            continue
        p = proc(int(f.name))
        if p is None or p['parent'] != parent or p['state'] in ('Z','X') or p['pid'] in known:
            continue
        try:
            argv = (f/'cmdline').read_bytes()
        except OSError:
            continue
        if b'client_entry.py' in argv:
            return {}
    return active


def terminal(job, c, a):
    return job/'episodes'/('case-%04d'%c)/a/'EPISODE-TERMINAL.json'


def eligible(job, p, active):
    if set(active) != {0,1}: return False
    for worker,row in active.items():
        spec=p['cases'][row['case_id']];order=spec['within_case_order'].split('_')
        if spec['proposed_GPU_index']!=worker or row['arm'] not in order:return False
        for arm in order[:order.index(row['arm'])]:
            f=terminal(job,row['case_id'],arm)
            if not f.is_file() or read(f).get('complete_episode') is not True:return False
        live=read(job/('server-worker%d'%worker)/'LIVE-COUNTER.json')['episode']
        if not isinstance(live,dict) or live.get('case_id')!=row['case_id'] or live.get('arm')!=row['arm']:return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior-job',required=True,type=Path)
    ap.add_argument('--control-dir',required=True,type=Path)
    a = ap.parse_args()
    job, out = a.prior_job.resolve(strict=True), a.control_dir.resolve()
    out.mkdir(parents=True,exist_ok=False)
    p = read(job/'plan.json')
    launched = read(job/'LAUNCH-RESULT.json')
    pin = launched['runner']
    require = lambda ok,msg: None if ok else (_ for _ in ()).throw(RuntimeError(msg))
    require(pin['owner_job']==str(job) and matches(pin),'prior CPU runner lifetime differs')
    started = time.time()
    deadline = min(started+5400,datetime.fromisoformat(p['absolute_work_stop']).timestamp()-900)
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGALRM):
        signal.signal(sig, lambda received,_frame: (_ for _ in ()).throw(
            TimeoutError('bounded handoff control signal: '+str(received))))
    signal.setitimer(signal.ITIMER_REAL, max(.01,deadline-started))
    write(out/'STARTED.json',dict(pid=os.getpid(),birth_ticks=proc(os.getpid())['birth_ticks'],
        started_at=datetime.now(timezone.utc).isoformat(),deadline_unix=deadline,
        prior_runner=pin,scope='complete admitted cases with unchanged v1 pending arms; no new case or episode retry'))
    paused, handed, selected = False, False, None
    try:
        while time.time()+4200 < deadline:
            require(matches(pin),'prior runner exited; no takeover or retry')
            require(not (job/'HARD-STOP.json').exists(),'prior hard failure; handoff cancelled')
            candidate = clients(job,pin['pid'])
            if not eligible(job,p,candidate):
                time.sleep(.3)
                continue
            os.kill(pin['pid'],signal.SIGSTOP)
            paused=True
            # Observe the stopped runner before trusting the admission barrier.
            until=time.time()+2
            while proc(pin['pid'])['state'] not in ('T','t') and time.time()<until:
                time.sleep(.01)
            actual=clients(job,pin['pid'])
            same = all(k in actual and actual[k]['pin']['pid']==v['pin']['pid'] for k,v in candidate.items())
            if not same or not eligible(job,p,actual):
                os.kill(pin['pid'],signal.SIGCONT);paused=False
                time.sleep(.3)
                continue
            selected=actual
            write(out/'RUNNER-PAUSED-ADMITTED-CASES.json',dict(at_unix=time.time(),clients=selected,
                no_client_or_server_paused=True,whole_episodes_continue=True))
            break
        if selected is None:
            write(out/'RESULT.json',dict(status='NO_COMMON_LAST_ARM_WINDOW_NO_CHANGE',wall_sec=time.time()-started))
            return 2
        while True:
            require(time.time()<deadline,'bounded handoff deadline')
            done=[]
            for worker,row in selected.items():
                f=terminal(job,row['case_id'],row['arm'])
                if f.is_file():
                    require(read(f).get('complete_episode') is True,'actual partial client; no qualification')
                    q=proc(row['pin']['pid'])
                    if q is None or q['state'] in ('Z','X'):
                        done.append(worker)
            if len(done)==2:
                break
            time.sleep(.2)
        for worker,row in selected.items():
            out_ep=terminal(job,row['case_id'],row['arm']).parent
            f=out_ep/'FINISH_REQUESTED'
            if not f.exists():
                f.write_text('actual complete last-arm client exited; controlled full-case handoff\n')
        while True:
            require(time.time()<deadline,'finish drain deadline')
            if all((terminal(job,row['case_id'],row['arm']).parent/'FINISH_READY.json').is_file() for row in selected.values()):
                break
            time.sleep(.2)
        # Continue only the previously registered missing arms of these two
        # admitted cases, using unchanged v1 source, model servers and manifest.
        import sys
        from concurrent.futures import ThreadPoolExecutor
        sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
        from tf_fullsuite_eval_v1.runner import Runner
        original_runner=Runner(job,p)
        v1=Path(__file__).resolve().parent.parent/'tf_fullsuite_eval_v1'
        def finish_pending(worker,row):
            order=p['cases'][row['case_id']]['within_case_order'].split('_')
            for arm in order[order.index(row['arm'])+1:]:
                require(time.time()+1320<deadline,'whole pending episode plus drain reserve')
                e=terminal(job,row['case_id'],arm).parent;e.mkdir(parents=True,exist_ok=False)
                child=original_runner.spawn([p['client_python'],'-B',v1/'client_entry.py','--job',job,
                    '--case-id',row['case_id'],'--arm',arm,'--out',e,'--port',19771+worker,
                    '--worker-index',worker,'--master-port',29771+worker],e/'client.log',
                    original_runner.environment(worker),original_runner.affinity[worker]['client'],
                    'client-case%d-%s'%(row['case_id'],arm))
                limit=time.monotonic()+1200
                while not original_runner.wait_reaped(child):
                    require(time.monotonic()<limit,'original complete pending client1200s cap')
                    require(time.time()<deadline,'orchestration cap')
                    time.sleep(.2)
                require(child['proc'].returncode==0 and read(e/'EPISODE-TERMINAL.json').get('complete_episode') is True,
                    'pending arm is partial; no retry')
                (e/'FINISH_REQUESTED').write_text('actual authorized pending arm naturally finished; handoff drain\n')
                limit=time.time()+120
                while not (e/'FINISH_READY.json').exists():
                    require(time.time()<limit,'original FINISH drain120s cap');time.sleep(.2)
                ready=read(e/'FINISH_READY.json')
                require(ready.get('full_episode_terminal') is True and ready.get('async_drained') is True,'pending arm drain missing')
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fs=[pool.submit(finish_pending,w,row) for w,row in selected.items()]
                for f in fs:f.result()
        finally:
            manual_cleanup=original_runner.cleanup()
            write(out/'MANUAL-PENDING-OWNED-CLEANUP.json',manual_cleanup)
        require(manual_cleanup['all_owned_work_exited'] is True,'manual owned client cleanup did not close')
        for worker,row in selected.items():
            c=row['case_id']
            for arm in ('T','C','P'):
                f=terminal(job,c,arm)
                require(f.exists() and read(f).get('complete_episode') is True,'full three-arm case missing')
            ready=read(terminal(job,c,row['arm']).parent/'FINISH_READY.json')
            require(ready.get('full_episode_terminal') is True and ready.get('async_drained') is True,'complete native drain missing')
        write(out/'ALL_CURRENT_CASES_NATURALLY_COMPLETE.json',dict(cases=[v['case_id'] for v in selected.values()],
            at_unix=time.time(),all_six_terminal=True,no_episode_truncated=True))
        # Both native servers are idle. Signal through their original safe exit
        # protocol rather than signalling a busy model process.
        for worker in (0,1):
            f=job/('server-worker%d'%worker)/'SHUTDOWN_REQUESTED'
            if not f.exists():f.write_text('complete cases drained; next four-worker namespace\n')
        stop=job/'STOP_REQUESTED'
        if not stop.exists():stop.write_text('CONTROLLED_COMPLETE_CASE_HANDOFF_TO_V2; not algorithm failure\n')
        handed=True
        os.kill(pin['pid'],signal.SIGCONT);paused=False
        write(out/'RESULT.json',dict(status='COMPLETE_CASE_HANDOFF_REQUESTED',cases=[v['case_id'] for v in selected.values()],
            wall_sec=time.time()-started,no_episode_truncated=True,prior_runner_resumed_for_original_cleanup=True,
            old_receipts_modified=False,prior_final_status='PENDING_ORIGINAL_RUNNER_CLOSURE'))
        return 0
    except BaseException as exc:
        write(out/'FAILED.json',dict(status='HANDOFF_FAILED_PRESERVE_PRIOR',error=repr(exc),
            traceback=traceback.format_exc(),wall_sec=time.time()-started,handed=handed))
        raise
    finally:
        if paused and matches(pin):
            os.kill(pin['pid'],signal.SIGCONT)


if __name__=='__main__':
    raise SystemExit(main())
