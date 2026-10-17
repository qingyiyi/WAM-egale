"""Compressed real environment migration plus pinned public model downloads.

Supersedes the stopped slow uncompressed transfer without resetting its clock.
No model/fixture/preflight invocation or project digest runs on AutoDL.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import pickle
import shlex
import signal
import subprocess
import threading
import time
import traceback

import paramiko
from stage_remote import APP, LOCAL, JOB_REL, REMOTE_JOB, cmd_python


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.parallel-tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    job = args.job.resolve(strict=True)
    plan = json.loads((job / 'plan.json').read_text())
    prep = json.loads((job / 'PREPARATION-STARTED.json').read_text())
    deadline = prep['prepare_deadline_unix']
    if time.time() >= deadline or (job / 'DEPLOYED-parallel-environments.json').exists():
        raise RuntimeError('expired preparation or already completed; no reset/overwrite')
    password = getpass.getpass('AutoDL SSH password: ')
    clients, processes, phases = [], [], []
    lock = threading.RLock()
    abort = threading.Event()
    monitor_done = threading.Event()
    def connect():
        c = paramiko.SSHClient()
        c.load_host_keys(str(LOCAL / '投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'))
        c.set_missing_host_key_policy(paramiko.RejectPolicy())
        c.connect('region-9.autodl.pro', port=40416, username='root', password=password,
                  timeout=30, auth_timeout=30, allow_agent=False, look_for_keys=False)
        c.get_transport().set_keepalive(10)
        with lock: clients.append(c)
        return c
    main_ssh = connect()
    sftp = main_ssh.open_sftp()
    def command(code, *values):
        _, out, err = main_ssh.exec_command(cmd_python(code, *values), timeout=60)
        raw, error = out.read(), err.read()
        rc = out.channel.recv_exit_status()
        if rc: raise RuntimeError(f'real deployment command failed {rc}: {error.decode(errors="replace")[:2000]}')
        return raw.decode()
    def put(path, destination):
        command('from pathlib import Path;import sys;Path(sys.argv[1]).parent.mkdir(parents=True,exist_ok=True)',destination)
        sftp.put(str(path), destination)
    try:
        status = command('''from pathlib import Path
import json,os,sys
j=Path(sys.argv[1]);p=json.loads((j/'plan.json').read_text())
assert p['authorization_origin']==sys.argv[2]
assert not (j/'STOP_REQUESTED').exists()
print(json.dumps(dict(glibc=os.confstr('CS_GNU_LIBC_VERSION'),owned_staging_path=str(j),old_transfers='see OWNED pins; no blanket cleanup')))
''', REMOTE_JOB, plan['authorization_origin'])
        write(job / 'PARALLEL-STAGING-METADATA.json', json.loads(status))
        put(Path(__file__).with_name('guard.py'), REMOTE_JOB + '/guard-r2.py')
        command('''from pathlib import Path
import subprocess,os,json,sys,time,signal
j=Path(sys.argv[1]);old=json.loads((j/'GUARD-STARTED.json').read_text());(j/'GUARD-STARTED-r1.json').write_text(json.dumps(old))
p=old['pid'];s=Path('/proc/%s/stat'%p)
if s.exists() and int(s.read_text().rpartition(') ')[2].split()[19])==old['birth_ticks'] and os.getpgid(p)==p:
 os.killpg(p,signal.SIGTERM)
 (j/'GUARD-r1-REPLACEMENT.json').write_text(json.dumps(dict(pid=p,birth_ticks=old['birth_ticks'],signal='TERM',reason='fix preparation escalation timing',at=time.time())))
q=subprocess.Popen(['/root/miniconda3/bin/python','-B',str(j/'guard-r2.py'),'--job',str(j)],stdin=subprocess.DEVNULL,stdout=(j/'guard-r2.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
print(json.dumps(dict(guard_pid=q.pid,replacement=True,absolute_clock_reset=False)))
''', REMOTE_JOB)
        put(Path(__file__).with_name('download_assets.py'), REMOTE_JOB + '/download_assets.py')
        model_root = Path(plan['target_checkpoint_path'])
        recorded = {item['Path']: item['Revision'] for item in pickle.loads((model_root / '.msc').read_bytes())}
        files = [dict(path=str(Path(path).relative_to(model_root)), bytes=stat['bytes'],
                      revision=recorded[str(Path(path).relative_to(model_root))])
                 for path, stat in plan['target_weight_stats'].items()]
        manifest = dict(model_id=plan['target_checkpoint_id'], remote_model_root=APP+'/models/'+model_root.name,
                        files=files, provenance='original local .msc records, not current main',
                        prepare_deadline_unix=deadline, no_file_digest=True)
        write(job / 'PINNED-MODEL-ASSETS.json', manifest)
        put(job / 'PINNED-MODEL-ASSETS.json', REMOTE_JOB + '/PINNED-MODEL-ASSETS.json')
        command('''from pathlib import Path
import subprocess,os,json,sys,time
j=Path(sys.argv[1]);p=subprocess.Popen(['/root/miniconda3/bin/python','-B',str(j/'download_assets.py'),'--manifest',str(j/'PINNED-MODEL-ASSETS.json'),'--job',str(j)],stdin=subprocess.DEVNULL,stdout=(j/'model-download.log').open('ab'),stderr=subprocess.STDOUT,start_new_session=True)
b=int(Path('/proc/%s/stat'%p.pid).read_text().rpartition(') ')[2].split()[19]);pin=dict(pid=p.pid,pgid=os.getpgid(p.pid),birth_ticks=b,owner_job=str(j),role='pinned-public-assets',started_unix=time.time());(j/('OWNED-%s.json'%p.pid)).write_text(json.dumps(pin));print(json.dumps(pin))
''', REMOTE_JOB)
        print('Pinned model download launched; four compressed environment streams starting.', flush=True)
        def monitor_assets():
            while not monitor_done.wait(15):
                try:
                    raw=command('from pathlib import Path;import sys; p=Path(sys.argv[1]); print(p.read_text() if p.exists() else "PENDING")',REMOTE_JOB+'/MODEL-DOWNLOAD-RESULT.json').strip()
                    if raw!='PENDING':
                        value=json.loads(raw);write(job/'MODEL-DOWNLOAD-RESULT.json',value)
                        if value['status']!='COMPLETE_ALL_PINNED_ASSETS':
                            abort.set();print('Pinned asset download failed; stopping own environment streams.',flush=True)
                        return
                except BaseException as exc:
                    write(job/'ASSET-PROGRESS-MONITOR-FAILED.json',dict(error=repr(exc)))
                    abort.set();return
        monitor=threading.Thread(target=monitor_assets,daemon=True);monitor.start()

        def stream(name, base, entries, destination, excludes=()):
            c = connect()
            code = '''import subprocess,os,json,sys,time,signal
from pathlib import Path
j=Path(sys.argv[1]);name=sys.argv[2];dest=Path(sys.argv[3]);end=float(sys.argv[4]);dest.mkdir(parents=True,exist_ok=True)
p=subprocess.Popen(['/bin/tar','-xzpf','-','-C',str(dest)],stdin=sys.stdin.buffer,start_new_session=True)
pin=dict(pid=p.pid,pgid=os.getpgid(p.pid),birth_ticks=int(Path('/proc/%s/stat'%p.pid).read_text().rpartition(') ')[2].split()[19]),owner_job=str(j),role='compressed-deployment-'+name);(j/('OWNED-%s.json'%p.pid)).write_text(json.dumps(pin))
try:
 rc=p.wait(timeout=max(1,end-time.time()));print(json.dumps(dict(stage=name,exit_code=rc)),flush=True)
 if rc:raise SystemExit(rc)
finally:
 if p.poll() is None:
  try:os.killpg(pin['pgid'],signal.SIGTERM)
  except ProcessLookupError:pass
'''
            channel = c.get_transport().open_session(window_size=64*2**20)
            channel.settimeout(60)
            channel.exec_command(cmd_python(code, REMOTE_JOB, name, destination, deadline))
            stop_reader = threading.Event()
            def drain():
                with (job / f'compressed-{name}-remote.log').open('ab') as output:
                    while not stop_reader.wait(.01):
                        try:
                            if channel.recv_ready(): output.write(channel.recv(65536)); output.flush()
                            if channel.recv_stderr_ready(): output.write(channel.recv_stderr(65536)); output.flush()
                        except (EOFError,OSError):return
            reader = threading.Thread(target=drain, daemon=True);reader.start()
            tar_args = ['/bin/tar','--format=posix','-I','gzip -1',
                '--exclude=__pycache__','--exclude=.git','--exclude=*.pyc',
                *['--exclude='+v for v in excludes],'-C',str(base),'-cf','-',*map(str,entries)]
            start = report_at = time.time();sent=0
            with (job / f'compressed-{name}-local.log').open('ab') as output:
                proc = subprocess.Popen(tar_args,stdout=subprocess.PIPE,stderr=output,start_new_session=True)
                with lock:processes.append(proc)
                write(job/f'LOCAL-OWNED-{proc.pid}.json',dict(pid=proc.pid,pgid=os.getpgid(proc.pid),birth_ticks=int(Path(f'/proc/{proc.pid}/stat').read_text().rpartition(') ')[2].split()[19]),role='gzip-stream-'+name))
                try:
                    while True:
                        if abort.is_set():raise RuntimeError('another real staging component failed; no retry')
                        if time.time()>=deadline:raise TimeoutError('original preparation deadline')
                        data=proc.stdout.read(1024*1024)
                        if not data:break
                        channel.sendall(data);sent+=len(data)
                        if time.time()-report_at>=30:
                            write(job/f'COMPRESSED-{name}-STATUS.json',dict(bytes=sent,elapsed_sec=time.time()-start))
                            print(f'{name}: {sent/2**20:.0f} MiB compressed sent',flush=True);report_at=time.time()
                    channel.shutdown_write();rc=proc.wait(timeout=30)
                    while not channel.exit_status_ready():
                        if time.time()>=deadline:raise TimeoutError('original preparation deadline')
                        time.sleep(.1)
                    remote_rc=channel.recv_exit_status()
                    if rc or remote_rc:raise RuntimeError(f'{name} tar local={rc} remote={remote_rc}; see logs')
                    result=dict(stage=name,archive_bytes=sent,elapsed_sec=time.time()-start,local_exit=rc,remote_exit=remote_rc)
                    with lock:phases.append(result)
                    write(job/f'COMPRESSED-{name}-RESULT.json',result)
                    print(f'{name}: complete {sent/2**20:.0f} MiB compressed',flush=True)
                finally:
                    if proc.poll() is None:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    stop_reader.set();reader.join(timeout=2);channel.close();c.close()

        envbase=Path('/rjs/cxz/miniconda3/envs');site='va/lib/python3.12/site-packages'
        parts=[('va-main',envbase,['va'],APP+'/envs',[site+'/nvidia',site+'/torch/lib',site+'/flash_attn_2_cuda*.so']),
               ('va-cuda',envbase,[site+'/nvidia',site+'/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so'],APP+'/envs',[]),
               ('va-torch',envbase,[site+'/torch/lib'],APP+'/envs',[]),
               ('libero-env',envbase,['libero-va'],APP+'/envs',[])]
        with ThreadPoolExecutor(max_workers=4) as pool:
            try:
                for future in as_completed([pool.submit(stream,*values) for values in parts]):future.result()
            except BaseException:
                abort.set();raise
        stream('source',LOCAL/'review_exports/WAM-egale',
               ['投机采样/experiments','投机采样/scripts','投机采样/wam-p0-worktree','投机采样/configs'],APP)
        stream('libero-assets',LOCAL,['third_party/LIBERO-official-20260922'],APP)
        stream('model-metadata',LOCAL/'models',[model_root.name],APP+'/models',
               [model_root.name+'/*.safetensors',model_root.name+'/*/*.safetensors',model_root.name+'/._____temp'])
        remote_plan=json.loads((job/'REMOTE-OPERATING-PLAN.json').read_text())
        for source,dest in [(plan['selected']['path'],remote_plan['selected']['path']),
              (plan['bank'],remote_plan['bank']),(job/'deployment-reference.pt',remote_plan['deployment_reference']),
              (job/'libero-config/config.yaml',REMOTE_JOB+'/libero-config/config.yaml')]:put(source,dest)
        extension=LOCAL/'投机采样/experiments/shared_prefix_torch_v2/_shared_torch_v1.cpython-312-x86_64-linux-gnu.so'
        put(extension,APP+'/'+str(extension.relative_to(LOCAL)))
        glibc=json.loads(status)['glibc'].split()[-1]
        if tuple(map(int,glibc.split('.')[:2]))<(2,34):raise RuntimeError('user-space dispatcher requires glibc2.34')
        for soname in ['libEGL.so.1','libGLdispatch.so.0','libOpenGL.so.0']:
            put(Path('/usr/lib/x86_64-linux-gnu')/soname,APP+'/userlib/'+soname)
        command('''from pathlib import Path
import sys,json
app=Path(sys.argv[1]); aliases={'/rjs/cxz/论文探索':app,'/rjs/cxz/miniconda3/envs/va':app/'envs/va','/rjs/cxz/miniconda3/envs/libero-va':app/'envs/libero-va'}
for old,new in aliases.items():
 p=Path(old);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() or p.is_symlink():
  if p.resolve()!=new.resolve():raise RuntimeError('unowned alias conflict')
 else:p.symlink_to(new,target_is_directory=True)
print(json.dumps(dict(aliases=True,no_service_change=True)))
''',APP)
        while True:
            if time.time()>=deadline:raise TimeoutError('original preparation deadline waiting assets')
            value=command('from pathlib import Path;import sys; p=Path(sys.argv[1]); print(p.read_text() if p.exists() else "PENDING")',REMOTE_JOB+'/MODEL-DOWNLOAD-RESULT.json').strip()
            if value!='PENDING':
                result=json.loads(value);write(job/'MODEL-DOWNLOAD-RESULT.json',result)
                if result['status']!='COMPLETE_ALL_PINNED_ASSETS':raise RuntimeError('pinned model staging failed; see result')
                break
            time.sleep(5)
        receipt=dict(status='ASSETS_READY_SOURCE_FREEZE_PENDING',phases=phases,
                     original_prepare_deadline_unix=deadline,at=datetime.now(timezone.utc).isoformat(),
                     no_model_forward=True,no_project_digest=True,old_slow_attempt_preserved=True)
        write(job/'DEPLOYED-parallel-environments.json',receipt)
        put(job/'DEPLOYED-parallel-environments.json',REMOTE_JOB+'/DEPLOYED-parallel-environments.json')
        print(json.dumps(receipt),flush=True)
    except BaseException as exc:
        write(job/'PARALLEL-STAGING-FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),phases=phases,clock_not_reset=True))
        raise
    finally:
        monitor_done.set()
        for proc in processes:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        sftp.close()
        for client in clients:client.close()


if __name__=='__main__':
    main()
