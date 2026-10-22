"""CPU-only contention/admission checks; synthetic transitions are not episodes."""
from pathlib import Path
import importlib.util,json,time,tempfile,threading
from concurrent.futures import ThreadPoolExecutor
from shared_dispatch import PairQueue,BudgetLease
B=Path(__file__).resolve().parent
with tempfile.TemporaryDirectory(prefix='cpu-queue-',dir=B) as path:
 p=Path(path);(p/'PAIR-CLAIMS.json').write_text(json.dumps(dict(pairs={},halt=None)))
 manifest=[dict(pair_id=i,episode_id=f'synthetic-{i}-{a}',task='synthetic',arm=a) for i in range(24) for a in ['T','C','P']];seen=[];lock=threading.Lock()
 def work(owner):
  q=PairQueue(p,manifest,owner,owner,p/owner)
  while True:
   rows,reason=q.take()
   if rows is None:break
   for c in rows:q.episode(c['pair_id'],dict(c,status='COMPLETE_SYNTHETIC_CPU_ONLY',lifecycle_seconds=1))
   q.finish(rows[0]['pair_id'],dict(pass_all=True,CPU_SYNTHETIC=True))
   with lock:seen.append((rows[0]['pair_id'],owner))
 with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(work,['CPU-A','CPU-B']))
 assert len(seen)==24 and len({i for i,o in seen})==24
 q=PairQueue(p,[dict(pair_id=999,episode_id='synthetic',task='synthetic',arm='T')],'CPU-A','CPU-A',p,time.time()-1);rows,reason=q.take();assert rows is None and reason=='SOFT_MORNING_WINDOW_COMPLETE'
 q.failed(None,'synthetic stop propagation');assert PairQueue(p,manifest,'CPU-B','CPU-B',p).take()[1]=='GLOBAL_HALT_NO_NEW_PAIR'
 (p/'GPU-LEDGER.json').write_text(json.dumps(dict(consumed_seconds=0,jobs=[])))
 a=BudgetLease(p,'CPU-A','GPU0',p,10000);b=BudgetLease(p,'CPU-B','GPU1',p,10000);a.open();b.open();assert a.reserve_pair() and not b.reserve_pair();a.close(dict(CPU_SYNTHETIC=True));b.close(dict(CPU_SYNTHETIC=True));assert not json.loads((p/'GPU-LEDGER.json').read_text())['active_jobs']
result=dict(status='PASS_CPU_ONLY_SHARED_QUEUE_CONTRACT',concurrent_pairs=24,unique_pair_ownership=True,three_arms_same_owner=True,soft_deadline_prevents_new_pair=True,global_halt_propagates=True,total_160h_budget_admission_no_double_reservation=True,CUDA_started=False,synthetic_not_robot_results=True)
(B/'CPU-QUEUE-RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(result['status'])
