"""Prepare a new-host namespace while preserving every old receipt verbatim."""
from pathlib import Path
import argparse,copy,json,os,time


def main():
    cli=argparse.ArgumentParser();cli.add_argument('--source-commit',required=True);a=cli.parse_args()
    os.sched_setaffinity(0,{43});base=(Path.home()/'cxz').resolve(strict=True)
    assert Path.cwd().resolve()==base and time.time()<1792272260.1204956
    meta=base/'jobs/baai-finalize-20261018-r1';meta.mkdir(parents=True,exist_ok=False)
    def read(p):return json.loads(Path(p).read_text())
    def write(p,obj):
        Path(p).parent.mkdir(parents=True,exist_ok=True)
        with Path(p).open('x') as f:json.dump(obj,f,ensure_ascii=False,indent=2)
    write(meta/'REGISTRATION.json',dict(goal='Relocate copied assets and create bounded unstarted-only original Long continuation',source_commit=a.source_commit,original_migration_origin=1792250660.1204956,deadline_epoch=1792272260.1204956,CPU=1,GPU=0,storage_cap_bytes=150*2**30,no_smoke=True,no_SHA=True,no_model_test=True))
    try:
        assert read(base/'jobs/baai-direct-migration-20261017-r1/RESULT.json')['status']=='DIRECT_STATIC_COPY_COMPLETE_NO_MODEL_EXECUTION'
        assert read(base/'jobs/baai-env-relocation-20261018-r1/RESULT.json')['status']=='COPIED_ENVIRONMENTS_PATH_RELOCATED_AND_METADATA_READ'
        for label in ('previous','new-shards'):
            assert read(base/'jobs/baai-evidence-migration-20261018-r1'/f'{label}-RESULT.json')['status']=='COPIED_ARCHIVE_AND_ALL_REGULAR_MEMBERS'
        app=base/'WAM-egale';assert not app.exists()
        (base/'direct-r1/WAM-egale').rename(app)
        assets=base/'assets';assert not assets.exists();(base/'direct-r1/assets').rename(assets)
        overlay=base/'code-overlay-20261018-r1'
        def apply_overlay():
            added=[]
            for p in overlay.rglob('*'):
                if not p.is_file():continue
                assert not p.is_symlink()
                dest=app/p.relative_to(overlay);dest.parent.mkdir(parents=True,exist_ok=True)
                if dest.exists():assert dest.read_bytes()==p.read_bytes(), 'new overlay cannot rewrite frozen source/receipt'
                else:
                    with dest.open('xb') as f:f.write(p.read_bytes())
                added.append(str(dest))
            return added
        mapping=[('/root/autodl-tmp/wam-tf-pf2-20261017/envs',str(base/'envs')),('/rjs/cxz/miniconda3/envs',str(base/'envs')),('/root/wam-suite-assets-20261017-v1',str(assets/'community')),('/root/autodl-tmp/wam-tf-pf2-20261017',str(app)),('/rjs/cxz/论文探索',str(app))]
        def translate(v):
            if not isinstance(v,str):return v
            for old,new in mapping:
                if v==old or v.startswith(old+'/'):return new+v[len(old):]
            return v
        links=[]
        for root in (app,assets):
            for parent,dirs,files in os.walk(root,followlinks=False):
                for n in dirs+files:
                    p=Path(parent)/n
                    if not p.is_symlink():continue
                    before=os.readlink(p)
                    absolute=os.path.normpath(str(p.parent/before)) if not os.path.isabs(before) else before
                    after=translate(absolute)
                    if after==absolute and Path(after).is_relative_to(base):continue
                    if after!=absolute:
                        assert Path(after).is_relative_to(base);relative=os.path.relpath(after,p.parent)
                        p.unlink();p.symlink_to(relative);links.append(dict(path=str(p),before=before,after=relative))
        runtime=app/'投机采样/results/runtime';runtime.mkdir(parents=True,exist_ok=True)
        moved=[]
        for p in (base/'evidence-stage/runtime').iterdir():
            dest=runtime/p.name;assert not dest.exists(), 'archive collision requires explicit identity review'
            p.rename(dest);moved.append(str(dest))
        overlay_files=apply_overlay()
        oldapp='/root/autodl-tmp/wam-tf-pf2-20261017'
        names=['p0-tf-fullsuite-long-v1-20261017-r1','p0-tf-fullsuite-long-v2-20261017-r2','p0-tf-long-shard-resume-v1-20261017-r1']
        roots=[dict(logical_source_job=oldapp+'/投机采样/results/runtime/'+n,physical_source_job=str(runtime/n)) for n in names]
        old=read(runtime/names[-1]/'plan.json');end=read(runtime/names[-1]/'RESUME-RESULT.json')
        assert end['status']=='PAUSED_AFTER_COMPLETE_SHARDS' and end['complete_cases']==219 and end['complete_episodes']==657 and end['all_owned_groups_empty'] and not end['partial_case_ids_no_retry']
        job=runtime/'p0-tf-long-baai-resume-v1-20261018-r1';job.mkdir(exist_ok=False)
        plan=copy.deepcopy(old)
        for k in ('root','remote_root','code_namespace','evidence_path','input_manifest','deployment_reference','bank','target_checkpoint_path','registry','va_python','client_python','LIBERO_CONFIG_PATH','target_source_reference','local_evidence_mirror','HW_namespace'):
            if k in plan:plan[k]=translate(plan[k])
        for k in ('path','checkpoint','remote_path'):
            if k in plan['selected']:plan['selected'][k]=translate(plan['selected'][k])
        plan['target_weight_stats']={translate(k):v for k,v in plan['target_weight_stats'].items()}
        registry=read(app/'投机采样/configs/libero_checkpoint_registry.json')
        for suite,row in registry['suites'].items():
            if row.get('local_path'):row['local_path']=translate(row['local_path'])
        registry['suites']['libero_10']['local_path']=plan['target_checkpoint_path']
        registry_path=job/'relocated-libero-checkpoint-registry.json';write(registry_path,registry);plan['registry']=str(registry_path)
        config=job/'libero-config';config.mkdir()
        benchmark=app/'third_party/LIBERO-official-20260922/libero/libero'
        y=dict(benchmark_root=str(benchmark),bddl_files=str(benchmark/'bddl_files'),init_states=str(benchmark/'init_files'),datasets=str(benchmark.parent/'datasets'),assets=str(benchmark/'assets'))
        (config/'config.yaml').write_text(''.join(k+': '+json.dumps(v,ensure_ascii=False)+'\n' for k,v in y.items()));plan['LIBERO_CONFIG_PATH']=str(config)
        done=end['complete_source_cases'];done_ids=[v['case_id'] for v in done];remaining=end['unstarted_case_ids'];assert done_ids==list(range(219)) and remaining==list(range(219,500))
        debit=0
        for row in roots:
            path=Path(row['physical_source_job']);p=read(path/'plan.json')
            for i in range(p['max_GPU_workers']):debit+=read(path/f'server-worker{i}/LIVE-COUNTER.json')['target_forward_attempts']
        queues={i:[c for c in remaining if plan['cases'][c]['execution_worker_index']==i] for i in range(4)};shards=[]
        while any(queues.values()):
            shard=[]
            for j in range(10):
                start=(len(shards)*2+j)%4
                for k in range(4):
                    i=(start+k)%4
                    if queues[i]:shard.append(queues[i].pop(0));break
            shards.append(sorted(shard))
        plan.update(job=str(job),root=str(app),remote_root=str(app),code_namespace=str(app/'投机采样/experiments/baai_resume_v1'),source_code_commit=a.source_commit,
            GPU='user authorized new BAAI 2x A100-SXM4-40GB; distinct from old AutoDL PCIe and local formal timing',prior_complete_source_cases=done,prior_completed_case_ids=done_ids,
            blocked_prior_partial_case_ids=[],historical_source_roots=roots,prior_target_forward_debit=debit,unstarted_case_ids=remaining,migration_shards=shards,
            worker_forward_caps={str(i):sum(plan['cases'][cid]['execution_worker_index']==i for cid in remaining)*12000+250 for i in range(4)},
            migration_schema='baai-unmodified-historical-provenance/v1',no_remote_root_aliases=True,no_historical_receipt_rewrite=True,
            migration_original_budget_clock_preserved=True,original_hardware_observations_preserved=True,
            physical_GPU_UUIDs=['GPU-c3fa8d6f-a1fe-2cfe-1dc9-5cf0a0779726','GPU-4acf84ff-31ce-dead-c2a4-2a122b123b8c'],
            evidence_path=str(job),local_evidence_mirror='NOT_STARTED_NEW_HARDWARE_EVIDENCE',deployment_source_path_mapping='All operating paths relocated under ~/cxz; old raw receipt paths remain immutable logical provenance')
        assert sum(plan['worker_forward_caps'].values())+debit<=plan['job_forward_attempt_cap']
        for k in ('selected','normalizer'):
            assert plan[k]['metadata']==old[k]['metadata'] if k=='selected' else plan[k]==old[k]
        mandatory=[Path(plan[k]) for k in ('bank','deployment_reference','target_source_reference','va_python','client_python','target_checkpoint_path','input_manifest')]+[Path(plan['selected']['path'])]
        assert all(p.exists() and p.resolve().is_relative_to(base) for p in mandatory),[(str(p),p.exists()) for p in mandatory]
        assert Path(plan['selected']['path']).stat().st_size==plan['selected']['bytes']
        for p,record in plan['target_weight_stats'].items():assert Path(p).stat().st_size==record['bytes'],p
        write(job/'plan.json',plan)
        write(job/'SOURCE-FREEZE.json',dict(source_code_commit=a.source_commit,scientific_commit=plan['scientific_commit'],unchanged_mathematical_source='tf_fullsuite_eval_v2 server/client/controller/evidence; target source checked inline against original bytes',new_namespace='baai_resume_v1',historical_sources=roots,source_SHA_executed=False))
        write(job/'DEPLOYMENT-COMPLETE.json',dict(status='OPERATING_ASSETS_AND_ORIGINAL_RECEIPTS_PRESENT',source_code_commit=a.source_commit,bank=str(plan['bank']),old_raw_receipts_rewritten=False,model_GPU_tests_executed=False,source_math_admission='PENDING_ORIGINAL_INLINE_CHECKS',path_binding='legacy ROOT only in an independent module namespace',permissions=plan['permissions']))
        ag=base/'AGENTS.md'
        if not ag.exists():ag.write_text('# 本服务器永久工作范围\n\n用户明确要求：所有项目代码、环境、缓存、下载、日志、数据、结果与临时文件只能位于本账号的 `~/cxz`。只读系统核查允许；不得在该目录外写项目文件、安装系统依赖、修改系统服务、复制SSH私钥或建立/rjs兼容alias。\n\nWAM科研原始失败、账本、四项false、固定模型数学与审计保持。迁移不重置授权。\n')
        write(meta/'RESULT.json',dict(status='NEW_HOST_ASSETS_AND_UNSTARTED_ONLY_PLAN_PREPARED_NO_GPU_START',job=str(job),relocated_symlinks=links,moved_original_roots=moved,overlay_files=overlay_files,prior_target_forward_debit=debit,complete_cases=219,pending_cases=len(remaining),original_authorization_origin=plan['authorization_origin'],original_work_stop=plan['absolute_work_stop'],original_cleanup=plan['absolute_cleanup_deadline'],no_SHA=True))
        print(json.dumps(dict(status='PREPARED_NO_GPU_START',job=str(job),prior_actual_forward_debit=debit,pending_cases=len(remaining)),ensure_ascii=False))
    except BaseException as e:
        write(meta/'FAILED.json',dict(error=repr(e),no_source_deletion=True));raise
    finally:write(meta/'CLEANUP.json',dict(status='PREPARATION_PROCESS_EXITS_NO_CHILDREN',GPU_started=False))

if __name__=='__main__':main()
