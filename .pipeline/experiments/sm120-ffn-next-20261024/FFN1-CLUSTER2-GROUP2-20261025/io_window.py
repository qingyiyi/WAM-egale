from pathlib import Path
import os,json,signal,time
P=Path('/root/autodl-tmp/cxz/acceptance-m42-20261025');O=Path('/root/autodl-tmp/cxz/operator-sm120-20261023/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025')
rows=[]
def same(d):
 try:return int(Path(f'/proc/{d["pid"]}/stat').read_text().rsplit(')',1)[1].split()[19])==int(d['birth'])
 except FileNotFoundError:return False
try:
 for n in ['bootstrap/assets','curobo-setup/build-curobo','server-setup-r2/server-packages']:
  f=P/'results'/(n+'-OWNED.json');end=P/'results'/(n+'-RESULT.json')
  if f.exists() and not end.exists():
   d=json.loads(f.read_text())['identity']
   if same(d):os.killpg(d['pgid'],signal.SIGSTOP);rows.append(dict(tag=n,**d))
 (O/'IO-PAUSE.json').write_text(json.dumps(dict(start=time.time(),owned=rows),indent=2))
 (O/'MEASURE-READY').touch(exist_ok=False)
 deadline=time.time()+180
 while time.time()<deadline:
  f=O/'RUN-STATUS.json'
  if f.exists() and json.loads(f.read_text()).get('status')!='RUNNING':break
  time.sleep(1)
finally:
 for d in rows:
  if same(d):os.killpg(d['pgid'],signal.SIGCONT)
 (O/'IO-RESUME.json').write_text(json.dumps(dict(end=time.time(),resumed=rows),indent=2))
