from pathlib import Path
import os,time,json,tarfile,urllib.request,traceback,shutil
R=Path('/root/cxz/robotwin');B=R/'runs/oidn-compatibility-1';B.mkdir(exist_ok=False);os.sched_setaffinity(0,{18});t=time.time();state=dict(status='RUNNING_PRIVATE_ENVIRONMENT_REPAIR',start=t,expected_bytes=64809031,source='official OIDN v2.3.3 release',reason='first scene OIDN2.0.1 illegal memory access',policy_episodes=0,teacher_forwards=0)
def save():(B/'STATUS.json').write_text(json.dumps(state,indent=2)+'\n')
try:
 p=R/'cache/oidn-transfer-private.json';j=json.loads(p.read_text());p.unlink();dest=R/'deps/oidn-2.3.3.tar.gz';save()
 with urllib.request.urlopen(j['url'],timeout=30) as resp,dest.open('xb') as f:
  n=0
  while True:
   if time.time()-t>300:raise TimeoutError('environment transfer cap')
   b=resp.read(4*2**20)
   if not b:break
   f.write(b);n+=len(b)
 if n!=j['size']:raise RuntimeError('release length mismatch')
 with tarfile.open(dest) as z:
  for a in z.getmembers():
   q=Path(a.name)
   if q.is_absolute() or '..' in q.parts:raise RuntimeError('unsafe archive')
  z.extractall(R/'deps')
 lib=R/'deps/oidn-2.3.3.x86_64.linux/lib';site=R/'envs/client/lib/python3.10/site-packages/sapien';loader=site/'_oidn_tricks.py';original=loader.read_text()
 if '2.0.1' not in original:raise RuntimeError('unexpected original loader')
 (B/'original_oidn_tricks.py').write_text(original)
 copied=[]
 for p in lib.iterdir():
  if p.name.startswith('libOpenImageDenoise') and p.is_file():
   target=site/'oidn_library'/p.name
   if not target.exists():shutil.copyfile(p,target);copied.append(dict(name=p.name,bytes=target.stat().st_size))
 loader.write_text(original.replace('2.0.1','2.3.3'))
 state.update(status='PRIVATE_OIDN_COMPATIBILITY_READY',bytes=n,libraries=copied,old_libraries_retained=True,render_settings_unchanged=True,denoiser='oidn',numeric_equivalence_to_old_unavailable=True)
except BaseException as e:state.update(status='FAILED_PRIVATE_ENVIRONMENT_REPAIR',error=repr(e),traceback=traceback.format_exc())
finally:state.update(end=time.time(),seconds=time.time()-t);save()
