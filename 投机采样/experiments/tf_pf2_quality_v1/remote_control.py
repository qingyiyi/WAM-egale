"""Interactive task-owned file transfer/control; SSH secret remains in memory."""
from __future__ import annotations
import getpass
import json
from pathlib import Path
import sys
import termios
import paramiko
from stage_remote import LOCAL, cmd_python


def main():
    client=paramiko.SSHClient()
    client.load_host_keys(str(LOCAL/'投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'))
    client.set_missing_host_key_policy(paramiko.RejectPolicy())
    password=getpass.getpass('AutoDL SSH password: ')
    client.connect('region-9.autodl.pro',port=40416,username='root',password=password,
        timeout=30,auth_timeout=30,look_for_keys=False,allow_agent=False)
    password=None
    client.get_transport().set_keepalive(10)
    sftp=client.open_sftp()
    original=None
    if sys.stdin.isatty():
        original=termios.tcgetattr(sys.stdin.fileno())
        hidden=termios.tcgetattr(sys.stdin.fileno());hidden[3]&=~termios.ECHO
        termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,hidden)
    print('READY_TASK_REMOTE_CONTROL',flush=True)
    try:
        for line in sys.stdin:
            request=json.loads(line)
            if request.get('close'):break
            try:
                transfers=[]
                for local,remote in request.get('put',[]):
                    sftp.put(local,remote);transfers.append(dict(direction='put',bytes=Path(local).stat().st_size))
                for remote,local in request.get('get',[]):
                    Path(local).parent.mkdir(parents=True,exist_ok=True)
                    sftp.get(remote,local);transfers.append(dict(direction='get',bytes=Path(local).stat().st_size))
                result=dict(id=request.get('id'),transfers=transfers)
                if request.get('code'):
                    _,out,err=client.exec_command(cmd_python(request['code'],*request.get('args',[])),timeout=60)
                    result.update(stdout=out.read().decode(errors='replace'),stderr=err.read().decode(errors='replace'),exit_code=out.channel.recv_exit_status())
                print(json.dumps(result,ensure_ascii=False),flush=True)
            except BaseException as exc:
                print(json.dumps(dict(id=request.get('id'),error=repr(exc)),ensure_ascii=False),flush=True)
    finally:
        if original is not None:termios.tcsetattr(sys.stdin.fileno(),termios.TCSANOW,original)
        sftp.close();client.close()


if __name__=='__main__':main()
