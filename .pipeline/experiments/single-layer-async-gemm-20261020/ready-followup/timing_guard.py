import subprocess,os,json,time,signal
from pathlib import Path
B=Path(__file__).resolve().parent;R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized');T=time.time();jobs=[]
N=B;B=N.parent/'ready-followup'
os.sched_setaffinity(0,{4,5})
for sub in ('norm-build','tmp','cache'):(B/sub).mkdir(exist_ok=True)
env=dict(os.environ,CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',WAM_FA2_BUILD_ROOT=str(B/'norm-build'),TORCH_CUDA_ARCH_LIST='8.0',TMPDIR=str(B/'tmp'),XDG_CACHE_HOME=str(B/'cache'),MAX_JOBS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=str(R.parents[1]))
def identity(pid):return dict(pid=pid,birth_ticks=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]),pgid=os.getpgid(pid))
deadline=time.time()+900
while not (B/'FULL-GATE-2-RESULT.json').exists():
 if time.time()>deadline:raise TimeoutError('bounded finite-gate dependency wait')
 time.sleep(2)
gate=json.loads((B/'FULL-GATE-2-RESULT.json').read_text())
while gate['status']=='RUNNING':
 if time.time()>deadline:raise TimeoutError('bounded finite-gate dependency wait')
 time.sleep(2);gate=json.loads((B/'FULL-GATE-2-RESULT.json').read_text())
assert gate['status']=='COMPLETE_FINITE_GATE_NOT_PRODUCTION_ADMISSION' and gate['output_pass_count']==9
for row in gate['rows']:
 assert row['new_vs_Torch']['bitwise_pass'] and row['cache_vs_Torch']['pass_all']
 assert row['repeat_exact'] and row['inputs_unchanged'] and row['RNG_unchanged'] and row['exact_once']
 assert all(x['bitwise_pass'] for x in row['stages'].values())
 assert all(x['output']['bitwise_pass'] and x['cache']['pass_all'] for x in row['legal_state_sequences'])
for name,cmd,cap,cpu in [('ready-timing-1',['/root/cxz/envs/va/bin/python','-B',str(B/'timing.py'),'1'],600,False),('ready-timing-2',['/root/cxz/envs/va/bin/python','-B',str(B/'timing.py'),'2'],600,False)]:
 precheck=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
 if any(line.startswith('GPU-c3fa8d6f') for line in precheck.splitlines()):raise RuntimeError('GPU0 not empty before worker')
 env['GPU_PRELAUNCH_EMPTY']='1'
 with (B/(name+'.log')).open('x') as out:
  p=subprocess.Popen(cmd,cwd=R,env=dict(env,CUDA_VISIBLE_DEVICES='' if cpu else 'GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726'),stdout=out,stderr=out,start_new_session=True);rec=dict(name=name,owned=identity(p.pid),start=time.time());jobs.append(rec)
  try:rc=p.wait(timeout=cap)
  except subprocess.TimeoutExpired:
   rec['timeout']=True
   if identity(p.pid)==rec['owned']:os.killpg(p.pid,signal.SIGTERM)
   try:rc=p.wait(timeout=15)
   except subprocess.TimeoutExpired:
    if identity(p.pid)==rec['owned']:os.killpg(p.pid,signal.SIGKILL)
    rc=p.wait(timeout=15)
  rec.update(rc=rc,reaped=True,end=time.time());B.joinpath('O3-TIMING-JOBS.json').write_text(json.dumps(dict(T0=T,jobs=jobs,lifecycle_seconds=time.time()-T),indent=2))
  if rc:break

