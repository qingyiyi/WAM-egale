"""Losslessly archive only finished, prefix-audited evidence; never alter old experiments."""
from pathlib import Path
import datetime,gzip,json,os,shutil,time,traceback
J=Path(__file__).resolve().parent;os.sched_setaffinity(0,{18})
reg=json.loads((J/'PLAN.json').read_text());deadline=min(datetime.datetime.fromisoformat(reg['deadline']).timestamp(),datetime.datetime.fromisoformat(reg['T0']).timestamp()+reg['GPU_lifecycle_cap_seconds'])+900
deadline=min(deadline,datetime.datetime.fromisoformat(reg['job_absolute_deadline']).timestamp()+300)
assert not (J/'ARCHIVER-STATUS.json').exists()
s=dict(status='RUNNING_CPU_ARCHIVER',pid=os.getpid(),birth_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),start=time.time(),archived=0,raw_bytes=0,compressed_bytes=0,GPU_used=False)
def save():
 f=J/'ARCHIVER-STATUS.json';t=f.with_suffix('.tmp');t.write_text(json.dumps(s,indent=2)+'\n');t.replace(f)
def compress(p):
 dst=p.with_suffix(p.suffix+'.gz');tmp=dst.with_suffix('.tmp')
 assert not dst.exists(),str(dst)
 with p.open('rb') as src,gzip.open(tmp,'wb',compresslevel=1) as out:shutil.copyfileobj(src,out,2**20)
 with p.open('rb') as src,gzip.open(tmp,'rb') as out:
  while True:
   a=src.read(2**20);b=out.read(2**20);assert a==b,'lossless round trip failed'
   if not a:break
 row=dict(original=str(p),archive=str(dst),original_bytes=p.stat().st_size,compressed_bytes=tmp.stat().st_size,roundtrip_byte_exact=True,time=time.time())
 tmp.replace(dst)
 # Manifest precedes removing the redundant uncompressed representation.
 with (J/'LOSSLESS-ARCHIVE.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
 p.unlink();s['archived']+=1;s['raw_bytes']+=row['original_bytes'];s['compressed_bytes']+=row['compressed_bytes'];save()
try:
 save()
 while time.time()<deadline:
  st=json.loads((J/'STATUS.json').read_text());audited={x['pair_id'] for x in st['pair_audits'] if x['pass_all']}
  for e in st['episodes']:
   if e['pair_id'] not in audited or not e['status'].startswith('COMPLETE_'):continue
   for p in [J/e['episode_id']/'INITIAL-ACTUAL.pt',J/'server'/e['episode_id']/'FIRST-TWO-POLICY.pt']:
    if p.exists():compress(p)
  if 'end' in st:s.update(status='COMPLETE_ARCHIVE_AFTER_JOB_TERMINAL');break
  time.sleep(30)
 else:s.update(status='STOPPED_ARCHIVER_DEADLINE')
except BaseException as e:s.update(status='FAILED_ARCHIVER',error=repr(e),traceback=traceback.format_exc())
finally:s.update(end=time.time());save()
