"""Read actual closed AutoDL receipt identities before recommending shutdown."""
from pathlib import Path
import argparse,getpass,json,os,shlex,time

REMOTE=r'''
from pathlib import Path
import json,os,subprocess
os.sched_setaffinity(0,{18});app=Path('/root/autodl-tmp/wam-tf-pf2-20261017');runtime=app/'投机采样/results/runtime'
names=['p0-baai-static-export-20261017-r1','p0-baai-closed-resume-export-20261017-r1'];receipts={};remaining=[]
def process(pid):
 try:
  p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(')',1)[1].split();return dict(pid=pid,birth_ticks=int(v[19]),pgid=int(v[2]),uid=p.stat().st_uid,state=v[0])
 except (OSError,ValueError,IndexError):return None
def inspect_pin(pin,source):
 if not isinstance(pin,dict) or not {'pid','birth_ticks'}<=set(pin):return
 q=process(pin['pid'])
 if q and q['birth_ticks']==pin['birth_ticks'] and q['state']!='Z':remaining.append(dict(source=source,registered=pin,actual=q))
for name in names:
 root=runtime/name
 for p in root.glob('*.json'):
  raw=p.read_text();receipts[str(p)]=raw;value=json.loads(raw)
  if 'STARTED' in p.name:inspect_pin(value.get('pin',value),str(p))
  if 'OWNED-TAR' in p.name:inspect_pin(value,str(p))
root=runtime/'p0-tf-long-shard-resume-v1-20261017-r1'
for name in ['RESUME-RESULT.json','GPU-OWNED-CLEANUP.json','FINAL-OWNED-CLEANUP.json','GUARD-FINAL.json','LAUNCH-TERMINAL.json','GUARD-PROCESS.json','SUPERVISOR-PROCESS.json']:
 p=root/name
 if p.is_file():
  raw=p.read_text();receipts[str(p)]=raw
  if name.endswith('PROCESS.json'):inspect_pin(json.loads(raw),str(p))
for p in root.glob('OWNED-*.json'):inspect_pin(json.loads(p.read_text()),str(p))
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name','--format=csv,noheader'],capture_output=True,text=True,timeout=20)
status=json.loads((root/'RESUME-RESULT.json').read_text())
static=[json.loads((runtime/names[0]/(phase+'-RESULT.json')).read_text()) for phase in ['envs','assets']]
cleanup=[json.loads((runtime/names[0]/(phase+'-CLEANUP.json')).read_text()) for phase in ['envs','assets']]
ready=(not remaining and gpu.returncode==0 and not gpu.stdout.strip() and status['all_owned_groups_empty'] and all(x['status']=='STATIC_SOURCE_TAR_EXIT0' for x in static) and all(x['tar_exit']==0 for x in cleanup))
print(json.dumps(dict(status='OLD_OWNED_RESEARCH_AND_STATIC_EXPORT_CLOSED' if ready else 'BLOCKED_OR_UNKNOWN',remaining_birth_matched_owned=remaining,GPU_query=dict(returncode=gpu.returncode,stdout=gpu.stdout,stderr=gpu.stderr),complete_cases=status['complete_cases'],complete_episodes=status['complete_episodes'],unstarted_cases=len(status['unstarted_case_ids']),original_receipts=receipts,signals_sent=[],source_deleted=False,no_SHA=True,instance_shutdown='NOT_EXECUTED_USER_OPERATION'),ensure_ascii=False))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);args=p.parse_args();job=args.job.resolve(strict=True)
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});r=json.loads((job/'REGISTRATION.json').read_text());assert time.time()<r['deadline_epoch']
 password=getpass.getpass('AutoDL password (memory only): ')
 import paramiko
 known=Path(__file__).resolve().parents[3]/'投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'
 client=paramiko.SSHClient();client.load_host_keys(str(known));client.set_missing_host_key_policy(paramiko.RejectPolicy())
 try:
  client.connect('region-9.autodl.pro',port=40416,username='root',password=password,look_for_keys=False,allow_agent=False,timeout=20,auth_timeout=20,banner_timeout=20)
  password=None
  command='cd /root/autodl-tmp/wam-tf-pf2-20261017 && /root/miniconda3/bin/python -B -c '+shlex.quote(REMOTE)
  stdin,stdout,stderr=client.exec_command(command,timeout=120);stdin.close();raw=stdout.read(4*2**20);err=stderr.read(2**16);rc=stdout.channel.recv_exit_status()
  (job/'REMOTE-STDERR.txt').write_bytes(err);assert rc==0
  data=json.loads(raw);(job/'RESULT.json').write_bytes(raw)
  print(json.dumps({k:v for k,v in data.items() if k!='original_receipts'},ensure_ascii=False),flush=True)
 finally:
  password=None;client.close();(job/'CLEANUP.json').write_text(json.dumps(dict(SSH_closed=True,source_deleted=False,signals_sent=[],secret_written=False)))

if __name__=='__main__':main()
