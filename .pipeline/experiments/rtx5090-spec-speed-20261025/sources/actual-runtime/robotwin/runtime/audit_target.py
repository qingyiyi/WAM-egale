"""CPU audit of all 48 natural terminal Target episodes; split fixed before collection."""
from pathlib import Path
import os,json,time,traceback
os.sched_setaffinity(0,{18,19})
import torch
B=Path(__file__).resolve().parent
R=Path('/root/cxz/robotwin/runs')
torch.set_num_threads(2)
TASKS=['place_object_basket','stack_blocks_two','handover_block','lift_pot','press_stapler','open_microwave']
IDS=[0,1,2,3,4,5,6,28,7,8,9,10,11,12,13,29]
state=dict(status='RUNNING_TARGET48_AUDIT',start=time.time(),episodes=[],extra_teacher_forwards=0,
           full_conditions_for_teacher_requery=False)
def save():(B/'DATA-AUDIT.json').write_text(json.dumps(state,indent=2)+'\n')
try:
    assert not torch.cuda.is_initialized()
    collections=[R/'target-first6-3',R/'target-remaining42-20261020']
    expected_status=['COMPLETE_FIRST6_TARGET_ONLY','COMPLETE_REMAINING42_TARGET_COLLECTION']
    records={}
    for root,status in zip(collections,expected_status):
        st=json.loads((root/'STATUS.json').read_text());assert st['status']==status and not st.get('remaining_owned')
        for record in st['episodes']:
            assert record['case'] not in records
            records[record['case']]=(root,record)
    assert len(records)==48
    cols={s:{k:[] for k in ['state','previous','target','scalars','task','episode','chunk','step']} for s in ['FIT','CAL']}
    references=[]
    for taskid,task in enumerate(TASKS):
        for valid in range(8):
            key=f'{task}-valid{valid:02d}';root,rec=records[key]
            ep=json.loads((root/key/'STATUS.json').read_text())
            assert ep['status'] in ['COMPLETE_SUCCESS','COMPLETE_ALGORITHMIC_FAILURE']
            assert all(ep['initial_checks'].values()) and rec['policy_chunks']==ep['policy_chunks']
            split='FIT' if valid<6 else 'CAL';chunks=sorted((root/'server'/key).glob('chunk-*.pt'))
            assert len(chunks)==ep['policy_chunks']>0
            seen=0
            for i,path in enumerate(chunks):
                stat=path.stat();d=torch.load(path,map_location='cpu',weights_only=False)
                assert d['case']==key and d['chunk']==i and len(d['rows'])==50
                assert d['source_kind']=='actual_target_visited_query'
                if 'case_manifest' in d:assert d['case_manifest']['role']==dict(stage='TARGET',split=split)
                previous=None
                for index,row in enumerate(d['rows']):
                    assert row['effective_channel_ids']==IDS
                    for k in ['state_full','sample_before','sample_after','teacher_direction_full','previous_target_velocity']:
                        assert row[k].shape==(1,30,2,16,1) and row[k].dtype==torch.bfloat16 and torch.isfinite(row[k]).all()
                    assert torch.equal(row['state_full'],row['sample_before'])
                    raw=row['teacher_raw_CFG2'].permute(0,2,1).reshape(2,30,2,16,1)
                    assert torch.equal(raw[:1],row['teacher_direction_full'])
                    if previous is not None: assert torch.equal(row['previous_target_velocity'],previous)
                    previous=row['teacher_direction_full']
                    sigma=float(row['sigma']);nxt=float(row['next_sigma']);t=float(row['timestep'])
                    assert 0<=nxt<=sigma<=1 and abs(t-1000*sigma)<.002
                    if index==0: continue # Production starts with canonical teacher; train successor queries.
                    for dst,src in [('state','state_full'),('previous','previous_target_velocity'),('target','teacher_direction_full')]:
                        cols[split][dst].append(row[src][0,IDS].reshape(512).float())
                    cols[split]['scalars'].append(torch.tensor([sigma,nxt,t/1000.]))
                    for k,v in [('task',taskid),('episode',taskid*100+valid),('chunk',i),('step',index)]:cols[split][k].append(v)
                    seen+=1
                references.append(dict(path=str(path),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,case=key,split=split,rows=49))
            state['episodes'].append(dict(case=key,task=task,valid=valid,split=split,rows=seen,
                success=ep['success'],commands=ep['environment_commands'],failure_penalty=ep['failure_penalty'],
                horizon=ep['task_step_limit'],policy_chunks=ep['policy_chunks']))
            save()
    data={s:{k:torch.stack(v) if k in ['state','previous','target','scalars'] else torch.tensor(v,dtype=torch.int64)
             for k,v in c.items()} for s,c in cols.items()}
    assert len(data['FIT']['episode'].unique())==36 and len(data['CAL']['episode'].unique())==12
    assert not set(data['FIT']['episode'].tolist())&set(data['CAL']['episode'].tolist())
    torch.save(dict(schema='robotwin-actual-target48/v1',data=data,channels=IDS,episodes=state['episodes']),B/'target48.pt')
    (B/'SHARD-REFERENCES.json').write_text(json.dumps(references,indent=2))
    state.update(status='PASS_TARGET48_COMPLETE_SUPERVISION',FIT_rows=len(data['FIT']['state']),CAL_rows=len(data['CAL']['state']),
                 all_failed_complete_episodes_retained=True,independent_test_complete=False)
except BaseException as e:state.update(status='FAILED_TARGET48_DATA_AUDIT',error=repr(e),traceback=traceback.format_exc())
finally:state['elapsed_seconds']=time.time()-state['start'];save()
if state['status']!='PASS_TARGET48_COMPLETE_SUPERVISION':raise SystemExit(1)
