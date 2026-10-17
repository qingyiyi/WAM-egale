"""Small-file task SSH control; credential is read from TTY and kept in RAM."""
from __future__ import annotations
import getpass
import json
from pathlib import Path
import shlex
import sys
import termios
import paramiko

ROOT=Path(__file__).resolve().parents[3]

def main():
    c=paramiko.SSHClient();c.load_host_keys(str(ROOT/'投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'))
    c.set_missing_host_key_policy(paramiko.RejectPolicy())
    secret=getpass.getpass('AutoDL SSH password: ')
    c.connect('region-9.autodl.pro',port=40416,username='root',password=secret,look_for_keys=False,
        allow_agent=False,timeout=30,auth_timeout=30);secret=None;c.get_transport().set_keepalive(10)
    original=None
    if sys.stdin.isatty():
        original=termios.tcgetattr(sys.stdin.fileno());quiet=termios.tcgetattr(sys.stdin.fileno())
        quiet[3]&=~termios.ECHO;termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,quiet)
    print('READY_SMALL_TASK_CONTROL',flush=True)
    try:
        for line in sys.stdin:
            q=json.loads(line)
            if q.get('close'):break
            result=dict(id=q.get('id'),transfers=[])
            try:
                if q.get('get') or q.get('put'):
                    with c.open_sftp() as s:
                        for local,remote in q.get('put',[]):
                            require_size=Path(local).stat().st_size
                            if require_size>8*2**20:raise ValueError('small source/receipt transfer only')
                            s.put(local,remote);result['transfers'].append(dict(direction='put',bytes=require_size))
                        for remote,local in q.get('get',[]):
                            size=s.stat(remote).st_size
                            if size>8*2**20:raise ValueError('small source/receipt transfer only')
                            Path(local).parent.mkdir(parents=True,exist_ok=True);s.get(remote,local)
                            result['transfers'].append(dict(direction='get',bytes=size))
                if q.get('code'):
                    argv=['/root/miniconda3/bin/python','-B','-c',q['code'],*map(str,q.get('args',[]))]
                    _,out,err=c.exec_command(shlex.join(argv),timeout=60)
                    result.update(stdout=out.read().decode(errors='replace'),stderr=err.read().decode(errors='replace'),
                        exit_code=out.channel.recv_exit_status())
            except BaseException as exc:result['error']=repr(exc)
            print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:
        if original is not None:termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,original)
        c.close()

if __name__=='__main__':main()
