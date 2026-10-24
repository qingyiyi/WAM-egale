"""Birth-pinned process ownership for this temporary benchmark only."""
from pathlib import Path
import json,os,signal,subprocess,time
P=Path('/root/autodl-tmp/cxz/spec-speed-20261025')
def identity(pid):
    try:
        text=Path(f'/proc/{pid}/stat').read_text();fields=text[text.rfind(')')+2:].split()
        return dict(pid=int(pid),birth=int(fields[19]),ppid=int(fields[1]),session=int(fields[3]),state=fields[0])
    except (FileNotFoundError,ProcessLookupError):return None
def launch(name,command,*,env=None):
    target=P/'results/processes';target.mkdir(parents=True,exist_ok=True)
    log=(target/(name+'.log')).open('ab',buffering=0)
    proc=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
    row=dict(name=name,command=command,start=time.time(),identity=identity(proc.pid),status='LAUNCHED')
    (target/(name+'.json')).write_text(json.dumps(row,indent=2)+'\n');return proc,row
def stop(row):
    origin=row['identity'];live=identity(origin['pid']);killed=[]
    # A process group is used only while its birth-pinned leader still exists.
    if live and live['birth']==origin['birth']:
        try:os.killpg(origin['pid'],signal.SIGTERM);killed.append(origin)
        except ProcessLookupError:pass
    for _ in range(50):
        check=identity(origin['pid'])
        if check is None or check['state']=='Z':break
        time.sleep(.1)
    live=identity(origin['pid'])
    if live and live['birth']==origin['birth'] and live['state']!='Z':
        try:os.killpg(origin['pid'],signal.SIGKILL)
        except ProcessLookupError:pass
    return dict(role=row['name'],requested=time.time(),birth_pinned_group_leader=origin,signals=killed,final_identity=identity(origin['pid']))
