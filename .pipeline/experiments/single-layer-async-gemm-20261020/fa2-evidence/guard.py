import os,sys,time,json,subprocess,signal
from pathlib import Path
B=Path(__file__).resolve().parent
OPS=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究')
START=json.loads((B/'START.json').read_text());os.sched_setaffinity(0,{4,5})
def ident(pid):
 return dict(pid=pid,birth=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid),uid=Path(f'/proc/{pid}').stat().st_uid)
name=sys.argv[1];mode=sys.argv[2];cap=900 if mode=='build' else 600
assert time.time()+cap+60<START['deadline']
assert not (B/(name+'.json')).exists()
for x in ('tmp','cache','cuda-cache','torch-cache','build'): (B/x).mkdir(exist_ok=True)
env=dict(os.environ,PYTHONPATH=str(OPS),PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MAX_JOBS='2',CUDA_VISIBLE_DEVICES='' if mode=='build' else START['GPU'],CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',TORCH_CUDA_ARCH_LIST='8.0',TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'),CUDA_CACHE_PATH=str(B/'cuda-cache'),TORCH_EXTENSIONS_DIR=str(B/'torch-cache'),WAM_FA2_BUILD_ROOT=str(B))
py='/root/cxz/envs/va/bin/python'
cmd=[py,str(OPS/'wam_ops/optimized/build_whole_layer.py'),'build_ext','--build-lib',str(OPS/'wam_ops/optimized'),'--build-temp',str(B/'build')]
if mode!='build':cmd=[py,'-B',str(B/'worker.py'),'--mode',mode,'--out',str(B/name)]
t0=time.time();rec=dict(status='RUNNING',mode=mode,parent=ident(os.getpid()),T0=t0,source=json.loads((B/'SOURCE.json').read_text()))
with (B/(name+'.log')).open('xb') as log:
 child=subprocess.Popen(cmd,cwd=OPS,env=env,stdout=log,stderr=log,start_new_session=True);rec['child']=ident(child.pid);(B/(name+'.json')).write_text(json.dumps(rec,indent=2))
 try:rc=child.wait(timeout=cap);rec['timeout']=False
 except subprocess.TimeoutExpired:
  rec['timeout']=True
  if ident(child.pid)==rec['child']:os.killpg(child.pid,signal.SIGTERM)
  try:rc=child.wait(timeout=15)
  except subprocess.TimeoutExpired:
   if ident(child.pid)==rec['child']:os.killpg(child.pid,signal.SIGKILL)
   rc=child.wait(timeout=15)
rec.update(status='COMPLETE' if rc==0 else 'FAILED',exit=rc,reaped=True,lifecycle_seconds=time.time()-t0)
rec['gpu_contexts_after']=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True)
(B/(name+'.json')).write_text(json.dumps(rec,indent=2))
