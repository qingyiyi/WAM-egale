"""Atomic complete-pair claims and the original shared 160 GPU-hour ledger."""
from pathlib import Path
from contextlib import contextmanager
import datetime,fcntl,json,math,os,time
ROOT=Path(__file__).resolve().parent

def atomic(path,value):
 tmp=path.with_name(path.name+f'.{os.getpid()}.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)

class PairQueue:
 def __init__(self,root,manifest,owner,gpu,namespace,soft_target=None):
  self.root=Path(root);self.manifest=manifest;self.owner=owner;self.gpu=gpu;self.namespace=str(namespace);self.soft=soft_target
  self.groups={}
  for c in manifest:self.groups.setdefault(c['pair_id'],[]).append(c)
 @contextmanager
 def locked(self):
  with (self.root/'PAIR-CLAIMS.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);path=self.root/'PAIR-CLAIMS.json';d=json.loads(path.read_text());yield d;atomic(path,d)
 def forecast(self,d,rows):
  # Conservative observed task/arm P90, with a 15-minute fallback for an unseen arm.
  values=[]
  for c in rows:
   samples=[e['lifecycle_seconds'] for p in d['pairs'].values() for e in p.get('episodes',[]) if e['task']==c['task'] and e['arm']==c['arm'] and e['status'].startswith('COMPLETE_')]
   samples.sort();values.append(samples[max(0,math.ceil(.9*len(samples))-1)] if samples else 900.)
  return max(60.,sum(values)*1.10+15.)
 def take(self):
  with self.locked() as d:
   if d.get('halt'):return None,'GLOBAL_HALT_NO_NEW_PAIR'
   remaining=[i for i in sorted(self.groups) if str(i) not in d['pairs']]
   if not remaining:return None,'NO_UNCLAIMED_PAIRS'
   i=remaining[0];rows=self.groups[i];estimate=self.forecast(d,rows);now=time.time()
   if self.soft is not None and (now>=self.soft or (now>=self.soft-2700 and now+estimate>self.soft+900)):
    d.setdefault('soft_stops',{})[self.owner]=dict(at=now,target=self.soft,next_pair=i,predicted_seconds=estimate,reason='estimated next full pair would finish beyond morning window')
    return None,'SOFT_MORNING_WINDOW_COMPLETE'
   d['pairs'][str(i)]=dict(status='CLAIMED',owner=self.owner,GPU_uuid=self.gpu,namespace=self.namespace,started_at=now,predicted_seconds=estimate,predicted_finish=now+estimate,episodes=[])
   return rows,'CLAIMED'
 def episode(self,pair,terminal):
  with self.locked() as d:
   p=d['pairs'][str(pair)];assert p['owner']==self.owner and p['status']=='CLAIMED';assert terminal['episode_id'] not in {x['episode_id'] for x in p['episodes']};p['episodes'].append(terminal)
 def finish(self,pair,audit):
  with self.locked() as d:
   p=d['pairs'][str(pair)];assert p['owner']==self.owner and len(p['episodes'])==3 and {e['arm'] for e in p['episodes']}=={'T','C','P'} and audit['pass_all'];p.update(status='COMPLETE_PAIR',finished_at=time.time(),prefix_audit=audit)
 def failed(self,pair,error):
  with self.locked() as d:
   if pair is not None and str(pair) in d['pairs'] and d['pairs'][str(pair)]['owner']==self.owner:d['pairs'][str(pair)].update(status='FAILED_OR_PARTIAL_NO_RETRY',error=error)
   d['halt']=dict(owner=self.owner,pair=pair,error=error,time=time.time())
 def halted(self):
  with self.locked() as d:return bool(d.get('halt'))

class BudgetLease:
 def __init__(self,parent,owner,gpu,namespace,cap=576000):
  self.parent=Path(parent);self.owner=owner;self.gpu=gpu;self.namespace=str(namespace);self.cap=cap;self.start=None
 @contextmanager
 def locked(self):
  with (self.parent/'GPU-LEDGER.lock').open('a') as f:
   fcntl.flock(f,fcntl.LOCK_EX);path=self.parent/'GPU-LEDGER.json';d=json.loads(path.read_text());d.setdefault('active_jobs',{});yield d;atomic(path,d)
 def spent(self,d):return d['consumed_seconds']+sum(time.time()-x['start'] for x in d['active_jobs'].values())
 def open(self):
  with self.locked() as d:
   assert self.owner not in d['active_jobs'];assert self.spent(d)+sum(x['reserved_seconds'] for x in d['active_jobs'].values())+2700<=self.cap
   self.start=time.time();d['active_jobs'][self.owner]=dict(start=self.start,GPU_uuid=self.gpu,namespace=self.namespace,reserved_seconds=2700);return d['consumed_seconds']
 def reserve_pair(self):
  with self.locked() as d:
   others=sum(x['reserved_seconds'] for k,x in d['active_jobs'].items() if k!=self.owner)
   if self.spent(d)+others+6300>self.cap:return False
   d['active_jobs'][self.owner]['reserved_seconds']=6300;return True
 def reserve_after_episode(self,arms_left):
  with self.locked() as d:d['active_jobs'][self.owner]['reserved_seconds']=arms_left*1800+900
 def within_cap(self):
  with self.locked() as d:return self.spent(d)<self.cap-900
 def close(self,state):
  if self.start is None:return
  with self.locked() as d:
   assert self.owner in d['active_jobs'];elapsed=time.time()-self.start;d['consumed_seconds']+=elapsed;d['jobs'].append(dict(namespace=self.namespace,**state));del d['active_jobs'][self.owner]
