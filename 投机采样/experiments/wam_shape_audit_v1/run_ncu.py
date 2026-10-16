"""Finite NCU mechanism counters, clock-control none, owned process tree only."""
import json,os,signal,subprocess,sys,time,traceback,csv,io
from pathlib import Path
from run_matrix import bounded,NVML,Idle,ownership,pin,bootstrap_key,OWN,ROOT
NCU='/opt/nvidia/nsight-compute/2026.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu'
def lifetime(info):
 return (info.pid,info.pgid,info.start_ticks,info.uid)
def parse_profile_csv(contents):
 lines=contents.splitlines();begin=next(i for i,x in enumerate(lines) if x.startswith('"ID",'))
 parsed=list(csv.DictReader(io.StringIO('\n'.join(lines[begin:]))))
 units=[row for row in parsed if row.get('ID','').strip()=='']
 assert len(units)==1 and units[0].get('Kernel Name','').strip()=='','one actual CSV units row required'
 assert units[0].get('gpu__time_duration.sum')=='us','actual duration unit must be us'
 assert units[0].get('sm__throughput.avg.pct_of_peak_sustained_elapsed')=='%','actual SM unit must be percent'
 assert units[0].get('dram__bytes.sum.per_second')=='Gbyte/s','actual DRAM bandwidth unit required'
 table=[row for row in parsed if row.get('ID','').strip()!='']
 assert len(table)==3 and sorted(row['ID'] for row in table)==['0','1','2'],'three actual launches 0,1,2 required'
 assert all('ampere_bf16' in row.get('Kernel Name','') for row in table),'unexpected actual profiled kernel'
 return table,units[0]
def main():
 out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
 os.sched_setaffinity(0,{1,2})
 cmd=[NCU,'--target-processes','all','--clock-control','none','--cache-control','none','--nvtx','--nvtx-include','SHAPE_PROFILE/','--kernel-name','regex:.*ampere_bf16.*','--launch-count','3','--section','SpeedOfLight','--section','MemoryWorkloadAnalysis','--section','LaunchStats','--section','Occupancy','--section','SchedulerStats','--export',str(out/'real-linear'),'/rjs/cxz/miniconda3/envs/va/bin/python','-B',str(OWN/'ncu_linear.py'),str(out/'INPUTS.json')]
 (out/'PLAN.json').write_text(json.dumps(dict(command=cmd,GPU_seconds_cap=180,cleanup_reserved=60,CPU=2,storage_cap_bytes=64*2**20,profile_kernel_count=3,clock_changes=False,formal_timing=False),indent=2)+'\n')
 nv=None;child=None;pins={};rows=[];error=None;start=None;cleanup=[]
 def refresh():
  for p in Path('/proc').iterdir():
   if not p.name.isdigit():continue
   try:
    inf=ownership.read_process(int(p.name))
    if ownership.is_owned(inf,pins):pins[inf.pid]=inf
   except (FileNotFoundError,ProcessLookupError,PermissionError,RuntimeError):continue
 def stop(sig):
  refresh()
  for pid,known in list(pins.items()):
   try:
    inf=ownership.read_process(pid)
    if lifetime(inf)==lifetime(known):os.kill(pid,sig);cleanup.append(dict(pid=pid,signal=int(sig)))
   except (FileNotFoundError,ProcessLookupError):pass
 def live():
  alive=[]
  for pid,known in pins.items():
   try:
    inf=ownership.read_process(pid)
    if lifetime(inf)==lifetime(known):
     raw=Path('/proc',str(pid),'stat').read_text()
     if raw[raw.rfind(')')+2:].split()[0]!='Z':alive.append(pid)
   except (FileNotFoundError,ProcessLookupError):pass
   except RuntimeError:alive.append(pid)
  return alive
 try:
  nv=bounded(NVML);idle=Idle()
  for _ in range(30):
   row=bounded(nv.sample);rows.append(row)
   if idle.add(row,time.monotonic()):break
   time.sleep(1)
  if len(idle.samples)<15:raise RuntimeError('clean idle unavailable')
  env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
  with (out/'ncu.log').open('x') as log:
   child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   start=time.monotonic();identity=pin(child,cmd,on_boot=lambda x:pins.update({x.pid:x}));pins[child.pid]=identity
   while child.poll() is None:
    refresh();row=bounded(nv.sample);rows.append(row)
    foreign=[p['pid'] for k in ('compute','graphics') for p in row[k] if p['pid'] not in pins]
    if foreign:raise RuntimeError('foreign GPU PID: '+repr(foreign))
    if time.monotonic()-start>120:raise TimeoutError('profile time cap, reserve60s cleanup')
    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())>64*2**20:raise RuntimeError('profile storage cap')
    time.sleep(1)
  assert child.returncode==0,'ncu nonzero exit'
  assert (out/'INPUTS.json').is_file() and (out/'real-linear.ncu-rep').is_file()
  exported=subprocess.run([NCU,'--import',str(out/'real-linear.ncu-rep'),'--csv','--page','raw'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=20)
  assert exported.returncode==0,exported.stderr
  (out/'metrics.csv').write_text(exported.stdout)
  table,units=parse_profile_csv(exported.stdout);launches=sorted({x['ID'] for x in table})
  assert len(launches)==3,'expected three actual profiler launches, got '+repr(launches)
  assert all('ampere_bf16' in x.get('Kernel Name','') for x in table),'unexpected profiled kernel'
  (out/'PROFILE-COUNT.json').write_text(json.dumps(dict(actual_launch_ids=launches,count=3,order_n_v=[1,3,7],kernel_filter='ampere_bf16',raw_csv='metrics.csv'),indent=2)+'\n')
  assert sum(p.stat().st_size for p in out.iterdir() if p.is_file())<=64*2**20,'post-export profile storage cap'
 except BaseException as exc:error=repr(exc)
 finally:
  if child is not None:
   try:
    stop(signal.SIGTERM)
    if child.poll() is None:
     try:child.wait(timeout=5)
     except subprocess.TimeoutExpired:stop(signal.SIGKILL);child.wait(timeout=5)
    limit=time.monotonic()+5
    while live() and time.monotonic()<limit:time.sleep(.2)
    if live():stop(signal.SIGKILL)
    limit=time.monotonic()+5
    while live() and time.monotonic()<limit:time.sleep(.2)
    if live():raise RuntimeError('owned processes still live: '+repr(live()))
   except BaseException as exc:error=error or repr(exc)
  final=None
  if nv:
   try:final=bounded(nv.sample);bounded(nv.close)
   except BaseException as exc:error=error or repr(exc)
  result=dict(status='COMPLETE_PROFILE_ONLY' if error is None else 'FAILED_OR_BLOCKED',error=error,GPU_wall_sec=0 if start is None else time.monotonic()-start,own_root_reaped=child is None or child.poll() is not None,all_owned_live_exited=not live(),cleanup=cleanup,owned=[vars(v) for v in pins.values()],resources=final,samples=rows)
  (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('samples','owned')}))
if __name__=='__main__':main()
