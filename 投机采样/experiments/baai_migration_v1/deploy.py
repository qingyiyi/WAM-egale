"""Upload the small new namespace/HW, then continue only authorized full episodes."""
from pathlib import Path
import argparse,io,json,os,shlex,subprocess,tarfile,time

ROOT=Path(__file__).resolve().parents[3]
RUNTIME=ROOT/'投机采样/results/runtime'
REMOTE=r'''
from pathlib import Path
import json,os,subprocess,sys,tarfile,time,traceback
os.sched_setaffinity(0,{43});base=(Path.home()/'cxz').resolve(strict=True);assert Path.cwd().resolve()==base
r=json.loads(sys.argv[1]);meta=base/'jobs/baai-deploy-continuation-20261018-r1';meta.mkdir(parents=True,exist_ok=False)
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
 with Path(p).open('x') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def pin(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,birth_ticks=int(s[19]),pgid=int(s[2]),uid=p.stat().st_uid)
write(meta/'REGISTRATION.json',r);write(meta/'STARTED.json',dict(**pin(os.getpid()),at=time.time(),CPU=[43],GPU_preparation=0))
overlay=base/'code-overlay-20261018-r1';overlay.mkdir(exist_ok=False);count=0
try:
 with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as t:
  for m in t:
   rel=Path(m.name);assert not rel.is_absolute() and '..' not in rel.parts and m.isfile()
   dest=overlay/rel;dest.parent.mkdir(parents=True,exist_ok=True)
   with t.extractfile(m) as source,dest.open('xb') as out:
    while True:
     block=source.read(2**20)
     if not block:break
     out.write(block)
   count+=1
 write(meta/'OVERLAY-RESULT.json',dict(files=count,source_commit=r['source_commit'],no_model_or_GPU_test=True,no_SHA=True))
 requirements=[base/'jobs/baai-direct-migration-20261017-r1/RESULT.json',base/'jobs/baai-evidence-migration-20261018-r1/previous-RESULT.json',base/'jobs/baai-evidence-migration-20261018-r1/new-shards-RESULT.json']
 last=0
 while not all(p.is_file() for p in requirements):
  assert time.time()+180<r['deadline_epoch'],'original migration deadline/reserve'
  failed=list((base/'jobs/baai-direct-migration-20261017-r1').glob('*FAILED.json'))+list((base/'jobs/baai-evidence-migration-20261018-r1').glob('*FAILED.json'))
  assert not failed, 'existing transfer failure, no automatic retry'
  if time.time()-last>30:
   print(json.dumps(dict(status='WAITING_ALREADY_RUNNING_STATIC_AND_ORIGINAL_COPY',missing=[str(p) for p in requirements if not p.is_file()])),flush=True);last=time.time()
  time.sleep(2)
 env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',TMPDIR=str(base/'tmp'),XDG_CACHE_HOME=str(base/'cache'))
 finalizer=overlay/'投机采样/experiments/baai_migration_v1/finalize.py'
 with (meta/'finalize.log').open('xb') as log:
  f=subprocess.run(['/usr/bin/python3','-B',str(finalizer),'--source-commit',r['source_commit']],cwd=base,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=min(600,max(1,r['deadline_epoch']-time.time()-180)))
 assert f.returncode==0,'finalization failed; preserve failure, no automatic retry'
 app=base/'WAM-egale';job=app/'投机采样/results/runtime/p0-tf-long-baai-resume-v1-20261018-r1'
 p=read(job/'plan.json');assert time.time()+1100<r['deadline_epoch'],'preserve model900s + actual first episode/readiness reserve'
 # This is resource admission for actual full episodes, not a model test.
 q=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],capture_output=True,text=True,timeout=15)
 write(meta/'GPU-ADMISSION.json',dict(returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,formal_performance=False))
 assert q.returncode==0 and not q.stdout.strip(),'new GPU externally occupied; no foreign signals'
 env['LD_LIBRARY_PATH']=':'.join([str(app/'userlib'),str(base/'envs/va/lib'),str(base/'envs/libero-va/lib'),str(base/'envs/va/lib/python3.12/site-packages/torch/lib'),'/usr/local/cuda/compat/lib.real','/usr/local/cuda/compat/lib','/usr/local/nvidia/lib','/usr/local/nvidia/lib64'])
 env['PATH']=str(base/'envs/va/bin')+':'+env['PATH']
 os.sched_setaffinity(0,{44})
 with (job/'supervisor.log').open('xb') as log:
  child=subprocess.Popen([p['va_python'],'-B',str(app/'投机采样/experiments/baai_resume_v1/launch.py'),'--job',str(job)],cwd=app,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True,start_new_session=True)
 owned=pin(child.pid);write(job/'SUPERVISOR-PROCESS.json',dict(**owned,owner_job=str(job),role='independent_original-budget-supervisor'));write(meta/'OWNED-RESEARCH-LAUNCH.json',dict(**owned,owner_job=str(job),original_work_stop=p['absolute_work_stop'],original_cleanup=p['absolute_cleanup_deadline']))
 print(json.dumps(dict(status='ACTUAL_FULL_EPISODE_CONTINUATION_LAUNCHED',job=str(job),supervisor=owned,source_commit=r['source_commit'],prior_complete_cases=219,pending_cases=281,no_smoke=True,no_SHA=True)),flush=True)
 while True:
  assert time.time()+30<r['deadline_epoch'],'migration verification deadline; research retains own original guard'
  ready=[i for i in range(4) if (job/f'server-worker{i}/READY.json').is_file()]
  cases=[]
  for x in job.glob('CASE-*-RESULT.json'):
   item=read(x)
   if item['status']=='COMPLETE_RECORDED_CASE':cases.append(item['case_id'])
  if len(ready)==4 and cases:
   result=dict(status='NEW_FULL_CASES_RECORDED_AND_ALL_FOUR_SERVICES_READY',job=str(job),ready_workers=ready,complete_new_case_ids=sorted(cases),source_commit=r['source_commit'],old_cases_reused_only=True,original_M16_budget_preserved=True,permissions=p['permissions'],formal_performance=False)
   write(meta/'RESULT.json',result);print(json.dumps(result),flush=True);break
  if child.poll() is not None:raise RuntimeError('actual research launcher exited before full-case handoff; inspect original terminal, no automatic retry')
  if time.time()-last>30:
   terminals=list((job/'episodes').glob('case-*/*/EPISODE-TERMINAL.json')) if (job/'episodes').exists() else []
   print(json.dumps(dict(status='OBSERVING_ACTUAL_COMPLETE_EPISODES_NO_TEST',ready_workers=ready,complete_episode_terminal_files=len(terminals),complete_cases=cases)),flush=True);last=time.time()
  time.sleep(3)
except BaseException as e:
 write(meta/'FAILED.json',dict(error=repr(e),traceback=traceback.format_exc(),no_retry=True,research_budget_not_reset=True));raise
finally:
 write(meta/'CLEANUP.json',dict(status='MIGRATION_OBSERVER_EXITS_RESEARCH_KEEPS_ITS_ORIGINAL_REGISTERED_GUARD',source_deleted=False,no_secret_files=True,does_not_claim_research_has_exited=True))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);a=p.parse_args();job=a.job.resolve(strict=True)
    r=json.loads((job/'REGISTRATION.json').read_text());os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    files=[]
    for root in (ROOT/'投机采样/experiments/baai_resume_v1',ROOT/'投机采样/experiments/baai_migration_v1'):
        files.extend(q for q in root.rglob('*') if q.is_file() and not q.is_symlink() and '__pycache__' not in q.parts and q.suffix!='.pyc')
    # Navigation, human records and structured rules are small. Large original
    # local tensor/JSON evidence stays on the original server, untouched.
    files.extend(q for q in (ROOT/'.pipeline').rglob('*') if q.is_file() and not q.is_symlink()
        and q.suffix in ('.md','.yaml','.yml') and 'evidence' not in q.relative_to(ROOT/'.pipeline').parts)
    files.append(RUNTIME/'p0-tf-pf2-readonly-plan-v1-20261017-r1/SOURCE-AND-CHECKPOINT-MANIFEST.json')
    files.append(ROOT/'AGENTS.md')
    assert sum(q.stat().st_size for q in files)<=32*2**20,'bounded source/HW upload only'
    blob=io.BytesIO()
    with tarfile.open(fileobj=blob,mode='w') as tar:
        for q in sorted(set(files)):tar.add(q,arcname=str(q.relative_to(ROOT)),recursive=False)
    ssh=['ssh','-T','-x','-a','-i','/rjs/cxz/.ssh/id_ed25519','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=3','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(RUNTIME/'p0-baai-server-inventory-readonly-local-cpu1-v1-20261017-r1/HOST-KEYS'),'-p','2222','flagfft-dev.liulibo.tech-platform_research.dx-calc1.ws@ssh.platform-multi.baai.ac.cn']
    command='cd "$HOME/cxz" && PYTHONDONTWRITEBYTECODE=1 python3 -B -c '+shlex.quote(REMOTE)+' '+shlex.quote(json.dumps(r))
    child=subprocess.Popen(ssh+[command],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
    birth=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]);(job/'OWNED-SSH.json').write_text(json.dumps(dict(pid=child.pid,birth_ticks=birth,pgid=child.pid,uid=os.getuid())))
    child.stdin.write(blob.getvalue());child.stdin.close()
    with (job/'remote.log').open('xb') as log:
        for line in child.stdout:
            log.write(line);log.flush();print(line.decode(errors='replace').rstrip(),flush=True)
    rc=child.wait();(job/'TERMINAL.json').write_text(json.dumps(dict(SSH_returncode=rc,files=len(set(files)),source_deleted=False,no_SHA=True),indent=2));raise SystemExit(rc)

if __name__=='__main__':main()
