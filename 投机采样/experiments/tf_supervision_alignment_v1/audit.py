"""Read-only TF64 supervision audit. CPU predictions are diagnostic only."""
import json, os, signal, sys, time, traceback, hashlib
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[3]
EXP = ROOT/'投机采样/experiments'
RT = ROOT/'投机采样/results/runtime'
OUT = RT/'p0-tf-supervision-alignment-cpu-v1-20261016-r1'
SOURCE = RT/'p0-task4-direction-diagnostic-completion-v1-20261016-r2'
COMMIT = 'b0dfac3a02f65de33a1217246b83003615166640'

def read(p): return json.loads(Path(p).read_text())
def write(p,x):
    with Path(p).open('x') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False); f.write('\n')
def require(x,s):
    if not x: raise ValueError(s)

def main():
    require(os.getenv('CUDA_VISIBLE_DEVICES')=='','CPU only')
    require(not OUT.exists(),'fresh namespace required')
    old = read(RT/'p0-tf-usable-train-v1-20261016-r1/PLAN.json')
    prior = read(SOURCE/'plan.json')
    cp = Path(prior['selected']['TF_CONT']['checkpoint'])
    files = sorted((SOURCE/'episodes/TF_CONT-task4-init26-repeat0/direction_trace').glob('chunk-*.pt'))
    require(len(files)==50,'complete episode required')
    inputs = [cp,Path(old['data']),Path(old['same_query_data'])]+files
    OUT.mkdir()
    write(OUT/'PREREGISTRATION.json',dict(goal='Diagnose fixed TF64 input/teacher alignment and conditional training coverage before the one allowed new data experiment',
        commit=COMMIT,created_at=datetime.now(timezone.utc).isoformat(),CPU_affinity=sorted(os.sched_getaffinity(0)),
        threads=2,wall_cap_sec=900,storage_cap_bytes=64*2**20,GPU=0,new_teacher=0,training=0,
        inputs={str(f):dict(bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in inputs},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        metrics=['same-task and scheduler-step matched TF error; current logged GPU scores kept separately',
                 'joint standardized state/previous nearest FIT distances versus original held-out CAL',
                 'previous direction provenance and reachable-parent teacher semantics'],
        sampling='all saved current queries; same-task historical FIT, mixed FIT and original CAL; no independent TEST reads',
        NN_reference='up to first 256 FIT task4 rows per scheduler step, no current or CAL references fitted',
        NN_interpretation='CAL distance p95 is descriptive OOD reference, not a new acceptance/quality gate',
        training_admission='separate evidence review; descriptive effects do not establish episode-failure causality',
        no_DEV_rows_for_training=True,permissions=prior['permissions'],terminal_status='REGISTERED_CPU_ONLY'))
    start=time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('CPU 900s cap')));signal.alarm(900)
    try:
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        sys.path.insert(0,str(EXP))
        from tf_usable_train_v1.adapter import load_candidate
        predictor=load_candidate(cp,'transformer64',device='cpu',expected_channels=range(7),expected_sha256=prior['selected']['TF_CONT']['checkpoint_sha256'])
        payload=torch.load(cp,map_location='cpu',weights_only=True);norm=payload['normalization']
        cur=[]
        for f in files:
            ch=torch.load(f,map_location='cpu',weights_only=False)
            for tr in ch['rounds']:
                if tr['kind']!='speculative_tree':continue
                for node in range(3):
                    level=int(node!=0)
                    cur.append(dict(state=tr['candidate_states'][node].float(),
                        previous=(tr['root_previous_direction'] if node==0 else tr['first_directions'][node-1]).float(),
                        target=tr['teacher_target_directions'][node].float(),
                        scalars=torch.tensor([tr['sigma_path'][level],tr['sigma_path'][level+1],tr['target_timesteps'][level]/1000.]),
                        node=node,step=tr['start_step']+level,
                        reached=node==0 or (tr['accepted_depth']>=1 and node-1==tr['selected_parent']),
                        logged=min(tr['first_costs'] if node==0 else tr['second_costs'][node-1])))
        current={k:torch.stack([r[k] for r in cur]) for k in ('state','previous','target','scalars')}
        current.update(step=torch.tensor([r['step'] for r in cur]),node=torch.tensor([r['node'] for r in cur]),
            reached=torch.tensor([r['reached'] for r in cur]),logged=torch.tensor([r['logged'] for r in cur]))
        hp=torch.load(old['data'],map_location='cpu',weights_only=False)
        def view(ids):
            idx=ids[hp['columns']['task'][ids]==4]
            return {k:hp['columns'][k][idx] for k in ('state','previous','target','scalars','step')}
        fit,cal=view(hp['views']['fit174']),view(hp['views']['cal20'])
        sqpack=torch.load(old['same_query_data'],map_location='cpu',weights_only=False)
        rows=[r for r in sqpack['current']['train_rows']+sqpack['aux_rows'] if r['task_index']==4]
        sq={k:torch.stack([r[k] for r in rows]) for k in ('state','previous','target')}
        sq['scalars']=torch.tensor([[r['sigma'],r['next_sigma'],r['timestep']/1000.] for r in rows])
        times={int(s):float(fit['scalars'][fit['step']==s,2][0]) for s in torch.unique(fit['step'])}
        sq['step']=torch.tensor([min(times,key=lambda s:abs(times[s]-float(t))) for t in sq['scalars'][:,2]])
        @torch.no_grad()
        def error(d):
            vals=[]
            for i in range(0,len(d['state']),128):
                # CPU, batched prediction is solely an offline error diagnostic.
                st=d['state'][i:i+128].bfloat16().float();p=d['previous'][i:i+128].bfloat16().float()
                sc=d['scalars'][i:i+128]
                x=(torch.cat((st,p,sc),1)-norm['feature_mean'])/norm['feature_std']
                y=(p[:,None]+predictor.model(x)*norm['residual_std'][None,None]+norm['residual_mean'][None,None]).bfloat16().float()
                delta=(sc[:,0]-sc[:,1]).abs()
                vals.append((y-d['target'][i:i+128,None]).square().sum(2).sqrt().min(1).values*delta)
            return torch.cat(vals)
        for d in (current,fit,cal,sq):d['err']=error(d)
        def features(d):return ((torch.cat((d['state'],d['previous']),1)-norm['feature_mean'][:224])/norm['feature_std'][:224])
        ff=features(fit)
        for d in (current,cal):
            x=features(d);dist=torch.empty(len(x))
            for s in torch.unique(d['step']):
                mask=d['step']==s;ref=ff[fit['step']==s][:256]
                require(len(ref)>0,'matching reference step missing')
                dist[mask]=torch.cdist(x[mask],ref).min(1).values/(224**.5)
            d['nn']=dist
        def stats(v):
            return dict(n=len(v),mean=float(v.mean()),p50=float(v.median()),p90=float(torch.quantile(v,.9)),p95=float(torch.quantile(v,.95)),max=float(v.max())) if len(v) else dict(n=0)
        groups={}
        for name,mask in [('root',current['node']==0),('reached_parent',(current['node']!=0)&current['reached']),('counterfactual_parent',(current['node']!=0)&~current['reached'])]:
            for band,bm in [('all',torch.ones(len(cur),dtype=torch.bool)),('high',current['scalars'][:,0]>=.6),('mid',(current['scalars'][:,0]>=.1)&(current['scalars'][:,0]<.6)),('low',current['scalars'][:,0]<.1)]:
                chosen=mask&bm
                if not chosen.any():continue
                idx=chosen.nonzero().flatten(); ref95=[]; matched={}
                for i in idx.tolist():
                    v=cal['nn'][cal['step']==current['step'][i]]
                    ref95.append(torch.quantile(v,.95))
                ref95=torch.stack(ref95)
                for label,d in [('target_FIT',fit),('original_CAL',cal),('old_mixed_FIT',sq)]:
                    weights=[];means=[];passes=[];covered=0
                    for s in torch.unique(current['step'][chosen]):
                        n=int((current['step'][chosen]==s).sum());v=d['err'][d['step']==s]
                        if len(v):weights.append(n);means.append(float(v.mean()));passes.append(float((v<=prior['threshold']).float().mean()));covered+=n
                    matched[label]=dict(coverage=covered/len(idx),mean_error=sum(n*v for n,v in zip(weights,means))/covered,
                        pass_rate=sum(n*v for n,v in zip(weights,passes))/covered)
                groups[name+'/'+band]=dict(rows=len(idx),logged_GPU_best_error=stats(current['logged'][chosen]),
                    diagnostic_CPU_best_error=stats(current['err'][chosen]),
                    diagnostic_CPU_pass_rate=float((current['err'][chosen]<=prior['threshold']).float().mean()),
                    nearest_target_FIT_joint_distance=stats(current['nn'][chosen]),
                    outside_matched_original_CAL_p95_fraction=float((current['nn'][chosen]>ref95).float().mean()),
                    matched=matched)
        write(OUT/'RESULT.json',dict(status='COMPLETE_TF_SUPERVISION_ALIGNMENT_CPU_DIAGNOSTIC',groups=groups,
            current_queries=len(cur),current_full_DEV_episodes=1,
            CPU_predictions_are_not_production_PARITY=True,production_decisions_unchanged=True,
            numerical_teacher_contract='Historical target-only FIT labels are sequential B2; current actual verifier labels are B14. M12 proves difference for one exact same query, not its full population effect.',
            recursive_previous_semantics='current parent previous = actual draft first direction; original target-only FIT previous = sequential teacher direction; historical mixed includes older generators',
            causal_task_failure='UNKNOWN; selected failed DEV episode cannot establish cause',
            no_training_or_new_data_started=True,permissions=prior['permissions'],elapsed_sec=time.monotonic()-start,GPU_initialized=torch.cuda.is_initialized()))
        print(json.dumps(dict(status='COMPLETE',seconds=time.monotonic()-start,groups=groups),ensure_ascii=False))
    except BaseException as exc:
        write(OUT/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),elapsed_sec=time.monotonic()-start));raise
    finally:signal.alarm(0)

if __name__=='__main__':main()
