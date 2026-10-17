"""Stream the existing environments and required assets, without hash or probes.

Password is read from the controlling terminal only and never written to disk.
This is real deployment; it never launches a model or an environment test.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import time
import traceback

import paramiko


LOCAL = Path(__file__).resolve().parents[3]
APP = '/root/autodl-tmp/wam-tf-pf2-20261017'
JOB_REL = '投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r1'
REMOTE_JOB = APP + '/' + JOB_REL


def write(path, data):
    path = Path(path)
    tmp = path.with_name(path.name + '.stage-tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def cmd_python(code, *args):
    return shlex.join(['/root/miniconda3/bin/python', '-B', '-c', code, *map(str, args)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', required=True, type=Path)
    parser.add_argument('--port', type=int, default=40416)
    parser.add_argument('--host', default='region-9.autodl.pro')
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    plan = json.loads((job / 'plan.json').read_text())
    started = time.time()
    # Count preparation from this new authorization origin, including failed
    # local entrypoint preparation; interpreter fixes do not reset the cap.
    setup_origin = datetime.fromisoformat(plan['authorization_origin']).timestamp()
    setup_end = min(setup_origin + plan['prepare_wall_cap_sec'],
                    datetime.fromisoformat(plan['absolute_work_stop']).timestamp())
    if (job / 'PREPARATION-STARTED.json').exists():
        raise RuntimeError('one preparation attempt; do not reset the clock')
    prep = dict(status='STARTED_REAL_DEPLOYMENT', at=datetime.now(timezone.utc).isoformat(),
                prepare_deadline_unix=setup_end, no_model_tests=True,
                remote_project_hash_operations=0, credential_storage=False)
    write(job / 'PREPARATION-STARTED.json', prep)
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(str(LOCAL / '投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'))
    ssh.set_missing_host_key_policy(paramiko.RejectPolicy())
    ssh.connect(args.host, port=args.port, username='root',
                password=getpass.getpass('AutoDL SSH password: '), timeout=30,
                auth_timeout=30, look_for_keys=False, allow_agent=False)
    ssh.get_transport().set_keepalive(15)
    sftp = ssh.open_sftp()
    phases = []

    def remote(code, *values):
        _, out, err = ssh.exec_command(cmd_python(code, *values), timeout=60)
        raw, error = out.read(), err.read()
        code = out.channel.recv_exit_status()
        if code:
            raise RuntimeError(f'remote deployment command exit={code}: {error.decode(errors="replace")[:2000]}')
        return raw.decode()

    def mkdir(path):
        remote('from pathlib import Path;import sys;Path(sys.argv[1]).mkdir(parents=True,exist_ok=True)', path)

    def put(local, destination):
        mkdir(str(Path(destination).parent))
        sftp.put(str(local), destination)

    try:
        metadata = remote('''from pathlib import Path
import json,os,shutil,sys
app=Path(sys.argv[1]); job=Path(sys.argv[2]); free=shutil.disk_usage('/root/autodl-tmp').free
if app.exists():raise RuntimeError('fresh deployment directory already exists')
if free < 40*2**30:raise RuntimeError('insufficient space for existing environments + model + evidence reserve')
app.mkdir();job.mkdir(parents=True)
libraries={d:[str(p) for p in Path(d).glob('*EGL*')] for d in ['/usr/lib/x86_64-linux-gnu','/usr/local/nvidia/lib64','/usr/lib64'] if Path(d).exists()}
print(json.dumps(dict(free_bytes=free,EGL_library_files=libraries,affinity=sorted(os.sched_getaffinity(0))),ensure_ascii=False))
''', APP, REMOTE_JOB)
        write(job / 'REMOTE-DEPLOYMENT-METADATA.json', json.loads(metadata))
        remote_plan = dict(plan)
        remote_plan.update(job=REMOTE_JOB, root=APP, local_source_job=str(job),
            deployment_reference=REMOTE_JOB + '/deployment-reference.pt',
            target_checkpoint_path=APP + '/models/' + Path(plan['target_checkpoint_path']).name,
            registry=APP + '/投机采样/configs/libero_checkpoint_registry.json',
            bank=APP + '/' + str(Path(plan['bank']).relative_to(LOCAL)),
            LIBERO_CONFIG_PATH=REMOTE_JOB + '/libero-config')
        remote_plan['selected'] = dict(plan['selected'], original_local_path=plan['selected']['path'],
            path=APP + '/' + str(Path(plan['selected']['path']).relative_to(LOCAL)),
            checkpoint=APP + '/' + str(Path(plan['selected']['path']).relative_to(LOCAL)))
        write(job / 'REMOTE-OPERATING-PLAN.json', remote_plan)
        put(job / 'REMOTE-OPERATING-PLAN.json', REMOTE_JOB + '/plan.json')
        put(job / 'PREREGISTRATION.json', REMOTE_JOB + '/PREREGISTRATION.json')
        put(job / 'PREPARATION-STARTED.json', REMOTE_JOB + '/PREPARATION-STARTED.json')
        put(Path(__file__).with_name('guard.py'), REMOTE_JOB + '/guard.py')
        remote('''import subprocess,sys,os,json,time
from pathlib import Path
job=Path(sys.argv[1]);p=subprocess.Popen(['/root/miniconda3/bin/python','-B',str(job/'guard.py'),'--job',str(job)],stdin=subprocess.DEVNULL,stdout=(job/'guard.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(guard_pid=p.pid,started=True)))
''', REMOTE_JOB)
        print('Deployment admitted; independent absolute watchdog attached.', flush=True)

        def stream(name, base, entries, destination):
            if time.time() >= setup_end:
                raise TimeoutError('registered preparation deadline')
            mkdir(destination)
            code = '''import json,os,signal,subprocess,sys,time
from pathlib import Path
job=Path(sys.argv[1]);name=sys.argv[2];dest=sys.argv[3];end=float(sys.argv[4])
def birth(pid):return int(Path('/proc/%s/stat'%pid).read_text().rpartition(') ')[2].split()[19])
p=subprocess.Popen(['/bin/tar','-xpf','-','-C',dest],stdin=sys.stdin.buffer,start_new_session=True)
pin=dict(pid=p.pid,pgid=os.getpgid(p.pid),birth_ticks=birth(p.pid),owner_job=str(job),role='deployment-'+name,started_unix=time.time())
(job/('OWNED-%s.json'%p.pid)).write_text(json.dumps(pin))
try:
 status=p.wait(timeout=max(1,end-time.time())); print(json.dumps(dict(stage=name,exit_code=status)),flush=True)
 if status:raise SystemExit(status)
finally:
 if p.poll() is None:
  try:
   if birth(p.pid)==pin['birth_ticks']:os.killpg(pin['pgid'],signal.SIGTERM)
  except ProcessLookupError:pass
'''
            ch = ssh.get_transport().open_session(window_size=64 * 2**20)
            ch.settimeout(60)
            ch.exec_command(cmd_python(code, REMOTE_JOB, name, destination, setup_end))
            args_tar = ['/bin/tar', '--format=posix', '--exclude=__pycache__', '--exclude=.git',
                        '--exclude=*.pyc', '-C', str(base), '-cf', '-', *map(str, entries)]
            with (job / f'deploy-{name}.log').open('ab') as log:
                proc = subprocess.Popen(args_tar, stdout=subprocess.PIPE, stderr=log, start_new_session=True)
                pin = dict(pid=proc.pid, pgid=os.getpgid(proc.pid), role='local-stream-' + name,
                           birth_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rpartition(') ')[2].split()[19]))
                write(job / f'LOCAL-OWNED-{proc.pid}.json', pin)
                sent = 0
                begin = report_at = time.time()
                try:
                    while True:
                        if time.time() >= setup_end:
                            raise TimeoutError('registered preparation deadline')
                        chunk = proc.stdout.read(1024 * 1024)
                        if not chunk:
                            break
                        ch.sendall(chunk)
                        sent += len(chunk)
                        if time.time() - report_at >= 30:
                            write(job / 'DEPLOYMENT-STATUS.json', dict(stage=name, bytes_sent=sent,
                                  stage_elapsed_sec=time.time() - begin, phase_completed=phases))
                            print(f'{name}: {sent / 2**30:.2f} GiB streamed', flush=True)
                            report_at = time.time()
                    ch.shutdown_write()
                    exit_local = proc.wait(timeout=30)
                    raw, error = bytearray(), bytearray()
                    while not ch.exit_status_ready():
                        if time.time() >= setup_end:
                            raise TimeoutError('registered preparation deadline')
                        if ch.recv_ready(): raw.extend(ch.recv(65536))
                        if ch.recv_stderr_ready(): error.extend(ch.recv_stderr(65536))
                        time.sleep(.05)
                    while ch.recv_ready(): raw.extend(ch.recv(65536))
                    while ch.recv_stderr_ready(): error.extend(ch.recv_stderr(65536))
                    exit_remote = ch.recv_exit_status()
                    if exit_local or exit_remote:
                        raise RuntimeError(f'{name} archive local={exit_local} remote={exit_remote}: {error.decode(errors="replace")[:2000]}')
                    phase = dict(stage=name, archive_bytes=sent, elapsed_sec=time.time() - begin,
                                 local_exit=exit_local, remote_exit=exit_remote)
                    phases.append(phase)
                    write(job / f'DEPLOYED-{name}.json', phase)
                    print(f'{name}: completed {sent / 2**30:.2f} GiB', flush=True)
                finally:
                    if proc.poll() is None:
                        os.killpg(pin['pgid'], signal.SIGTERM)
                        try: proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(pin['pgid'], signal.SIGKILL)
                            proc.wait()
                    ch.close()

        stream('environments', Path('/rjs/cxz/miniconda3/envs'), ['va', 'libero-va'], APP + '/envs')
        stream('model', LOCAL / 'models', [Path(plan['target_checkpoint_path']).name], APP + '/models')
        frozen = LOCAL / 'review_exports/WAM-egale'
        stream('frozen-source', frozen, ['投机采样/experiments', '投机采样/scripts',
               '投机采样/wam-p0-worktree', '投机采样/configs'], APP)
        stream('libero', LOCAL, ['third_party/LIBERO-official-20260922'], APP)
        for source, dest in [(plan['selected']['path'], remote_plan['selected']['path']),
              (plan['bank'], remote_plan['bank']), (job / 'deployment-reference.pt', remote_plan['deployment_reference']),
              (job / 'libero-config/config.yaml', REMOTE_JOB + '/libero-config/config.yaml')]:
            put(source, dest)
        extension = LOCAL / '投机采样/experiments/shared_prefix_torch_v2/_shared_torch_v1.cpython-312-x86_64-linux-gnu.so'
        put(extension, APP + '/' + str(extension.relative_to(LOCAL)))
        # Preserve existing conda shebangs/editable paths without a reinstall.
        remote('''from pathlib import Path
import json,sys
app=Path(sys.argv[1]);aliases={'/rjs/cxz/论文探索':app,'/rjs/cxz/miniconda3/envs/va':app/'envs/va','/rjs/cxz/miniconda3/envs/libero-va':app/'envs/libero-va'}
for old,new in aliases.items():
 p=Path(old);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() or p.is_symlink():
  if p.resolve()!=new.resolve():raise RuntimeError('unowned alias conflict '+old)
 else:p.symlink_to(new,target_is_directory=True)
print(json.dumps(dict(aliases={k:str(v) for k,v in aliases.items()},no_system_service_change=True)))
''', APP)
        result = dict(status='ASSETS_READY_SOURCE_IMPLEMENTATION_PENDING', at=datetime.now(timezone.utc).isoformat(),
                      elapsed_sec=time.time() - started, phases=phases,
                      remote_project_hash_operations=0, model_forward_calls=0,
                      remote_job=REMOTE_JOB, requires_final_source_freeze=True)
        write(job / 'ASSET-STAGING-RESULT.json', result)
        put(job / 'ASSET-STAGING-RESULT.json', REMOTE_JOB + '/ASSET-STAGING-RESULT.json')
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        write(job / 'DEPLOYMENT-FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc(),
              elapsed_sec=time.time() - started, phases=phases, original_start_not_reset=True))
        raise
    finally:
        sftp.close()
        ssh.close()


if __name__ == '__main__':
    main()
