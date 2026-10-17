"""Explicit relocated-provenance contract, without changing old bytes or numerical checks."""
from __future__ import annotations
import ast,copy,json
from pathlib import Path
import sys

OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from tf_fullsuite_eval_v2 import protocol as base


def basic_contract():
    # Reuse every original current-job mathematical/resource gate up to the
    # predecessor filesystem gate. The latter is explicitly replaced below,
    # because old absolute paths are immutable provenance on a different host.
    source=Path(base.__file__);tree=ast.parse(source.read_bytes(),filename=str(source))
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load')
    at=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(v,ast.Name) and v.id=='prior' for v in n.targets))
    node=copy.deepcopy(fn);node.name='_basic_relocated_plan_contract';node.body=node.body[:at]+[ast.Return(value=ast.Name(id='p',ctx=ast.Load()))]
    module=ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[]));ns=dict(vars(base))
    exec(compile(module,str(source),'exec'),ns)
    return ns[node.name]

_BASIC=basic_contract()


def load(job):
    job=Path(job).resolve(strict=True);p=_BASIC(job);require=base.require;read=base.read
    workspace=(Path.home()/'cxz').resolve()
    require(p['migration_schema']=='baai-unmodified-historical-provenance/v1','explicit migration contract required')
    require(job.is_relative_to(workspace) and Path(p['root']).resolve()==base.ROOT
        and Path(p['va_python']).is_relative_to(workspace) and Path(p['client_python']).is_relative_to(workspace),'new job/assets must stay in cxz')
    require(all(Path(p[n]).is_relative_to(workspace) for n in ('bank','deployment_reference','target_checkpoint_path','registry','LIBERO_CONFIG_PATH')),'all operating paths must stay in cxz')
    require(p['no_remote_root_aliases'] is True and p['no_historical_receipt_rewrite'] is True,'relocation must not edit historical records')
    complete=p['prior_complete_source_cases'];ids=[v['case_id'] for v in complete];blocked=p.get('blocked_prior_partial_case_ids',[])
    require(ids==sorted(set(ids))==p['prior_completed_case_ids'] and not set(ids).intersection(blocked),'explicit global source map required')
    roots=p['historical_source_roots'];physical={v['logical_source_job']:Path(v['physical_source_job']).resolve(strict=True) for v in roots}
    require(len(physical)==len(roots) and all(q.is_relative_to(workspace) for q in physical.values()),'historical roots must be unique and inside cxz')
    observed=set();debit=0;historical_plans={}
    for logical,path in physical.items():
        old=read(path/'plan.json');historical_plans[logical]=old
        require(old.get('job',logical)==logical and old['scientific_commit']==p['scientific_commit']
            and old['suite']==p['suite'] and old['evaluation_seed']==p['evaluation_seed']
            and old['threshold']==p['threshold'] and old['mode']==p['mode'] and old['selector']==p['selector']
            and old['selected']['metadata']==p['selected']['metadata']
            and old['target_checkpoint_id']==p['target_checkpoint_id']
            and old['authorization_origin']==p['authorization_origin']
            and old['absolute_work_stop']==p['absolute_work_stop'] and old['absolute_cleanup_deadline']==p['absolute_cleanup_deadline'],
            'historical algorithm, selected identity or original budget differs')
        close=read(path/'GPU-OWNED-CLEANUP.json');final=read(path/'FINAL-OWNED-CLEANUP.json')
        require(close['all_owned_groups_empty'] and close['all_owned_work_exited'] and final['all_owned_groups_empty'],'real original owned cleanup required')
        require(read(path/'GUARD-FINAL.json')['status']=='CLEAN','old guard must have closed')
        for i in range(old['max_GPU_workers']):
            v=read(path/f'server-worker{i}'/'LIVE-COUNTER.json')
            require(type(v['target_forward_attempts']) is int and v['target_forward_attempts']>=0 and v['active_target_forwards']==0,'closed old actual attempts required')
            debit+=v['target_forward_attempts']
        for folder in (path/'episodes').glob('case-*'):
            if any((folder/a).exists() for a in base.ARMS):
                cid=int(folder.name[5:]);require(cid not in observed,'one logical case cannot have multiple attempted sources');observed.add(cid)
    require(observed==set(ids)|set(blocked),'no started/partial case can become new pending work')
    for item in complete:
        cid=item['case_id'];logical=item['source_job'];path=physical[logical]
        for arm in base.ARMS:
            oldout=Path(logical)/'episodes'/f'case-{cid:04d}'/arm;newout=path/'episodes'/f'case-{cid:04d}'/arm
            t,c,a,f=[read(newout/n) for n in ('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json')]
            require(all(v['case_id']==cid and v['arm']==arm and v['permissions']==base.PERMISSIONS for v in (t,c,a,f)),'original raw identity differs')
            require(t['context']==dict(case_id=cid,arm=arm,out=str(oldout)),'original logical context must stay untouched')
            require(t['status']=='COMPLETE_RECORDED_FULL_EPISODE' and t['complete_episode'] and t['quality_terminal']
                and t['horizon']==800 and type(t['success']) is bool and 0<t['env_steps']<=800
                and (t['success'] or t['horizon_reached']) and t['failure800_penalty']==(t['env_steps'] if t['success'] else 800)
                and f['full_episode_terminal'] and f['async_drained'] and f['status'] in ('PASS','FAILED_COMPONENT_AUDIT'),
                'natural whole episode/drain required; component FAILED is preserved, not promoted')
    require(debit==p['prior_target_forward_debit'],'every generation actual forward debit must be carried')
    pending=[s for s in p['cases'] if s['case_id'] not in set(ids)|set(blocked)]
    require(p['unstarted_case_ids']==[s['case_id'] for s in pending],'new execution restricted to actual unstarted cases')
    for i in range(4):require(p['worker_forward_caps'][str(i)]==sum(s['execution_worker_index']==i for s in pending)*12000+250,'remaining worker reservations differ')
    require(sum(p['worker_forward_caps'].values())+debit<=p['job_forward_attempt_cap'],'original series forward cap exceeded')
    shards=p['migration_shards'];flat=[cid for shard in shards for cid in shard]
    require(sorted(flat)==p['unstarted_case_ids'] and len(set(flat))==len(flat) and all(0<len(s)<=10 for s in shards),'predeclared complete case shards differ')
    return p


def install():
    base.load=load
    return base
