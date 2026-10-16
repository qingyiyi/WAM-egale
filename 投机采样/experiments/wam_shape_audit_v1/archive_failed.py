"""Read-only archive of the declared FAILED old100; never grants admission."""
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

OWN = Path(__file__).resolve().parent
REMOTE = r'''
import io,json,os,sys,tarfile
from pathlib import Path
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
job=Path('/root/autodl-tmp/libero-scale/heldout50pair-job')
read=lambda p:json.loads(p.read_text())
plan=read(job/'plan.json'); runs=sorted(job.glob('RUN-*-RESULT.json')); run=read(runs[-1])
assert run['status']=='FAILED' and run['completed_episodes']==100
assert run['cleanup']['all_roots_reaped'] is True
assert 'paired initial state differs task2/init25/initial_obs_hashes' in run['error']
assert plan['config']['arms']==['T','A_OFF']
assert {(r['task'],r['init'],r['arm']) for r in plan['assignments']}=={(t,i,a) for t in range(10) for i in range(25,30) for a in ('T','A_OFF')}
live=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:args=(p/'cmdline').read_bytes().split(b'\0'); stat=(p/'stat').read_text()
 except OSError:continue
 if stat[stat.rfind(')')+2:].split()[0]=='Z':continue
 if any(x.decode(errors='replace')==str(job) for x in args):live.append(int(p.name))
assert not live,repr(live)
files={}
for p in job.glob('*.json'):files['job/'+p.name]=p
required=('QUALIFIED.json','READY.json','CLEANUP.json','COUNTER-CLOSED.json','FINISH_READY.json','ACCEPTANCE-TREES.json','FORWARD-RESERVATION.json','client/episodes.jsonl','client/repro_metadata.json','client/run_metadata.json','client/chunk_events.jsonl','client/commands.npy','client/initial_state.npz','client/final_state.npz')
for r in plan['assignments']:
 folder=job/'episodes'/r['id']; q=read(folder/'QUALIFIED.json')
 assert all(q[k]==r[k] for k in ('id','task','init','arm'))
 assert read(folder/'CLEANUP.json')['all_roots_reaped'] is True
 for name in required:
  p=folder/name; assert p.is_file() and not p.is_symlink(),str(p)
  files['job/'+str(p.relative_to(job))]=p
 for name in ('client/initial_observations.npz','RESET.json'):
  p=folder/name
  if p.is_file():files['job/'+str(p.relative_to(job))]=p
total=sum(p.stat().st_size for p in files.values());assert total<=256*2**20,total
meta=dict(status='ARCHIVED_ORIGINAL_FAILED_NOT_PASS',original_status=run['status'],episodes=100,source=str(job),remote_writes=False,GPU_used=False,live_original_roots=live,original_quality_exists=(job/'QUALITY-ANALYSIS.json').exists(),total_bytes=total,files=[dict(name=n,bytes=p.stat().st_size) for n,p in sorted(files.items())])
raw=json.dumps(meta,ensure_ascii=False,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 info=tarfile.TarInfo('ARCHIVE-MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 for name,p in sorted(files.items()):tar.add(p,arcname=name,recursive=False)
'''

def main():
    out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=False)
    config=json.loads((OWN.parent/'autodl_research_quality_v1/config.json').read_text())
    s=config['ssh'];r=config['remote'];start=time.monotonic()
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ControlPath='+s['control_path'],'-p',str(s['port']),s['host'],shlex.join([r['server_python'],'-B','-'])]
    partial=out/'evidence.partial.tar'
    try:
        with partial.open('xb') as stream:
            result=subprocess.run(command,input=REMOTE.encode(),stdout=stream,stderr=subprocess.PIPE,timeout=600)
        if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
        assert partial.stat().st_size<270*2**20
        archive=out/'evidence.tar';partial.rename(archive)
        with tarfile.open(archive) as tar:
            for member in tar:
                p=Path(member.name)
                if p.is_absolute() or '..' in p.parts or not member.isfile():raise ValueError('unsafe member')
                dest=out/p;dest.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as src,dest.open('xb') as dst:
                    while block:=src.read(2**20):dst.write(block)
        status=dict(status='ARCHIVED_ORIGINAL_FAILED_NOT_PASS',elapsed_sec=time.monotonic()-start,archive_bytes=archive.stat().st_size,remote_writes=False,GPU_used=False,cleanup='SSH child exited and reaped; no remote worker created')
    except BaseException as e:
        status=dict(status='FAILED_ARCHIVE',error=repr(e),elapsed_sec=time.monotonic()-start)
        (out/'ARCHIVE-RESULT.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');raise
    (out/'ARCHIVE-RESULT.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n');print(json.dumps(status))

if __name__=='__main__':main()
