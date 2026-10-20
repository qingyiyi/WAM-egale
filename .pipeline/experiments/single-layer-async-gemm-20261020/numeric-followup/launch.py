from pathlib import Path
import importlib.util,json
s=importlib.util.spec_from_file_location('remote_existing',Path(__file__).resolve().parents[2]/'persistent-tile-remote-20261019/remote.py')
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
worker=Path(__file__).with_name('worker.py').read_text()
guard='''import subprocess,os,json,time,signal
from pathlib import Path
B=Path(__file__).resolve().parent
T=time.time();work=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究')
env=dict(os.environ,PYTHONPATH=str(work),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726',TMPDIR=str(B),CUBLASLT_LOG_LEVEL='5',CUBLASLT_LOG_FILE=str(B/'cublaslt.log'))
def identity(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
with (B/'worker.log').open('x') as f:
 p=subprocess.Popen(['/root/cxz/envs/va/bin/python','-B',str(B/'worker.py')],cwd=work,env=env,stdout=f,stderr=f,start_new_session=True)
 rec=dict(owned=identity(p.pid),T0=T);B.joinpath('OWNED.json').write_text(json.dumps(rec))
 try:rc=p.wait(timeout=560)
 except subprocess.TimeoutExpired:
  rec['timeout']=True
  if identity(p.pid)==rec['owned']:os.killpg(p.pid,signal.SIGTERM)
  try:rc=p.wait(timeout=15)
  except subprocess.TimeoutExpired:
   if identity(p.pid)==rec['owned']:os.killpg(p.pid,signal.SIGKILL)
   rc=p.wait(timeout=15)
 rec.update(returncode=rc,reaped=True,lifecycle_seconds=time.time()-T)
 B.joinpath('CLEANUP.json').write_text(json.dumps(rec,indent=2))
'''
code="from pathlib import Path\nimport subprocess,os,json,time\nB=Path('/root/cxz/ops/persistent-tile-20261019/evidence/async-gemm-20261020/numeric-followup')\nB.mkdir(parents=True,exist_ok=False)\nB.joinpath('worker.py').write_text("+repr(worker)+")\nB.joinpath('guard.py').write_text("+repr(guard)+")\nB.joinpath('BUDGET.json').write_text(json.dumps(dict(T0=time.time(),GPU='GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726',lifecycle_cap=600,full_block_cap=6,primitive_cap=24,source_authorization='2026-10-20 user approved ordered plan; existing M39 evidence',new_robot_work=False)))\nf=open(B/'guard.log','x');p=subprocess.Popen(['python3',str(B/'guard.py')],stdout=f,stderr=f,start_new_session=True,cwd=B)\nprint(dict(guard_pid=p.pid,birth_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rsplit(')',1)[1].split()[19])))\n"
r=m.run(code,30);assert r.returncode==0
