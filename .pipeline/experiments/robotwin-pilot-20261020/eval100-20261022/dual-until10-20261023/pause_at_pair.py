"""Gracefully drain only the current old worker's complete three-arm pair."""
from pathlib import Path
import json,time,datetime,os,traceback
J=Path('/root/cxz/robotwin/runs/eval100-20261022/gpu1');O=J.parent/'dual-until10-20261023';os.sched_setaffinity(0,{26})
q=json.loads((J/'INPUT-MANIFEST.json').read_text());last={c['pair_id']:c['episode_id'] for c in q};start=time.time();deadline=start+6000
state=dict(status='WAITING_LAST_ARM_OF_CURRENT_PAIR',start=start,deadline=deadline,GPU_used=False,pid=os.getpid(),birth_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]))
def save():
 p=O/'DRAIN-STATUS.json';p.with_suffix('.tmp').write_text(json.dumps(state,indent=2)+'\n');p.with_suffix('.tmp').replace(p)
try:
 assert not (O/'DRAIN-STATUS.json').exists();save()
 while time.time()<deadline:
  d=json.loads((J/'STATUS.json').read_text());state.update(old_status=d['status'],active_case=d.get('active_case'),complete=len(d['episodes']))
  if 'end' in d:
   assert d['status']=='PAUSED_BY_REQUEST_AFTER_COMPLETE_EPISODE',d.get('error',d['status'])
   assert len(d['episodes'])%3==0 and len(d['pair_audits'])==len(d['episodes'])//3
   state.update(status='COMPLETE_OLD_WORKER_DRAINED_AFTER_FULL_PAIR',old_terminal=d);break
  if d.get('active_case')==last.get(d.get('active_pair')) and not (J/'STOP-AFTER-EPISODE').exists():
   (J/'STOP-AFTER-EPISODE').write_text('User-approved temporary dual-GPU handoff; finish current complete pair, preserve all terminal records.\n');state.update(status='STOP_REQUESTED_ON_LAST_ARM',requested_at=time.time())
  save();time.sleep(5)
 else:raise TimeoutError('Finite handoff wait cap; no forced episode interruption')
except BaseException as e:state.update(status='FAILED_HANDOFF_CONTROLLER',error=repr(e),traceback=traceback.format_exc())
finally:state['end']=time.time();save()
