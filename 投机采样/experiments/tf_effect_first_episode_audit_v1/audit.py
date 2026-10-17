"""Audit one already terminal real episode during collection, not full30 GO."""
import json,os,sys,time,hashlib,signal,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];RT=ROOT/'投机采样/results/runtime'
SOURCE=RT/'p0-tf-effect-collect30-v1-20261016-r1'
OUT=RT/'p0-tf-effect-first-complete-episode-audit-cpu-v1-20261016-r1'
def write(p,x):
    with p.open('x') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
def main():
    assert os.getenv('CUDA_VISIBLE_DEVICES')=='' and not OUT.exists()
    sys.path.insert(0,str(ROOT/'投机采样/experiments'))
    from tf_effect_collector_audit_v1 import pack
    plan=pack.read(SOURCE/'plan.json');case=plan['cases'][0];out=SOURCE/'episodes'/case['key']
    assert (out/'FINISH_READY.json').exists()
    counter=pack.read(out/'COUNTER.json');refs=counter['teacher']['shards']
    files=[out/'COUNTER.json',out/'FINISH_READY.json',out/'FINAL-SERVER-STATE.json',out/'RESET.json',
        out/'client/episodes.jsonl',out/'client/COMMAND-MAPPING.json',out/'client/repro_metadata.json']
    files += [Path(r['path']) for r in refs]
    OUT.mkdir();start=time.monotonic()
    write(OUT/'PREREGISTRATION.json',dict(goal='Complete first real terminal episode data-path audit while remaining collection runs; not a truncated or full30 quality claim',
        CPU_affinity=sorted(os.sched_getaffinity(0)),wall_cap_sec=300,storage_cap_bytes=64*2**20,GPU=0,model_forward=0,teacher_forward=0,
        case=case,source_commit=plan['commit'],inputs={str(f):dict(bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in files},
        source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (Path(__file__),Path(pack.__file__))},
        full30_admission=False,training=False,terminal_status='REGISTERED_CPU_ONLY'))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('first full episode audit cap')));signal.alarm(300)
    try:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        qual=pack.qualifications(plan)
        records=[json.loads(x) for x in (out/'client/episodes.jsonl').read_text().splitlines() if x.strip()]
        assert len(records)==1 and records[0]['status']=='completed'
        record=records[0];mapping=pack.read(out/'client/COMMAND-MAPPING.json')
        pack.commands(torch,mapping,record,counter['requests'])
        pack.physical_artifacts(torch,out,record,mapping,pack.read(SOURCE/'TF_CONT-READY.json'),pack.read(out/'RESET.json'))
        max_error=0.;trees=rows=reached=0
        for ref in refs:
            shard=torch.load(ref['path'],map_location='cpu',weights_only=True);cols,tr=shard['columns'],shard['trees']
            n=shard['tree_count'];assert n==ref['trees'] and len(cols['state'])==n*3
            pred=cols['proposals'].float();target=cols['target'].float()
            costs=torch.linalg.vector_norm(pred-target[:,None],dim=2)*cols['delta'][:,None]
            saved=torch.stack((tr['first_costs'],tr['second_costs'][:,0],tr['second_costs'][:,1]),1).reshape(n*3,2)
            max_error=max(max_error,float((costs-saved).abs().max()));assert max_error<=1e-6
            node=cols['node'];depth=cols['accepted_depth'];parent=tr['selected_parent'].repeat_interleave(3)
            selected=(node==0)|(node-1==parent);actual_reached=(node==0)|(selected&(depth>=1))
            assert torch.equal(cols['state_reached'],actual_reached)
            assert torch.equal(cols['draft_edge_committed'],((node==0)&(depth>=1))|((node>0)&selected&(depth==2)))
            for i in range(n):
                route=pack.route(tr['first_costs'][i],tr['second_costs'][i],float(tr['threshold'][i]))
                assert route==(int(tr['selected_parent'][i]),int(tr['selected_child'][i]),int(tr['accepted_depth'][i]))
            trees+=n;rows+=len(node);reached+=int(actual_reached.sum())
        assert trees==counter['phase_counters']['tree_count']==len(counter['acceptance'])
        assert counter['phase_counters']['committed_scheduler_update_count']==record['infer_chunks']*50
        assert not torch.cuda.is_initialized()
        result=dict(status='PASS_COMPLETE_FIRST_EPISODE_DATA_PATH_AUDIT_ONLY',case=case,complete_episode=True,
            complete_policy_chunks=record['infer_chunks'],trees=trees,rows=rows,reached_rows=reached,teacher_score_CPU_max_abs=max_error,
            production_command_and_physical_artifacts_exact=True,qualification=qual,full30_admission=False,new_model_quality='UNMEASURED',
            GPU=0,model_forward=0,teacher_forward=0,elapsed_sec=time.monotonic()-start)
        write(OUT/'RESULT.json',result);print(json.dumps(result,ensure_ascii=False))
    except BaseException as exc:
        write(OUT/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),elapsed_sec=time.monotonic()-start));raise
    finally:signal.alarm(0)
if __name__=='__main__':main()
