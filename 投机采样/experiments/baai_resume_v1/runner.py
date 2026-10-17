"""Original whole-case numerical execution; independent new-host paths and source map."""
from __future__ import annotations
import argparse,json,os
from pathlib import Path
import sys,time,traceback
from concurrent.futures import ThreadPoolExecutor
OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from baai_resume_v1.contract import install
protocol=install()
from tf_fullsuite_eval_v2.runner import Runner
read,write,require,PERMISSIONS=protocol.read,protocol.write,protocol.require,protocol.PERMISSIONS


class MigratedRunner(Runner):
    def spawn(self,argv,log,env,affinity,role):
        if role.startswith('server-') or role.startswith('client-'):
            argv=[argv[0],'-B',OWN/'entry.py','server' if role.startswith('server-') else 'client',*argv[3:]]
        return super().spawn(argv,log,env,affinity,role)

    def environment(self,index,*,report=False):
        env=super().environment(index,report=report);root=(Path.home()/'cxz').resolve();cache=root/'cache';temp=root/'tmp'/self.job.name
        temp.mkdir(parents=True,exist_ok=True)
        names=dict(TMPDIR=temp,TMP=temp,TEMP=temp,XDG_CACHE_HOME=cache,XDG_CONFIG_HOME=root/'config',XDG_DATA_HOME=root/'data',
            HF_HOME=cache/'huggingface',HF_HUB_CACHE=cache/'huggingface/hub',HUGGINGFACE_HUB_CACHE=cache/'huggingface/hub',
            HF_DATASETS_CACHE=cache/'huggingface/datasets',TRANSFORMERS_CACHE=cache/'huggingface/transformers',TORCH_HOME=cache/'torch',
            TORCH_EXTENSIONS_DIR=cache/'torch_extensions',TRITON_CACHE_DIR=cache/'triton',CUDA_CACHE_PATH=cache/'cuda',
            TORCHINDUCTOR_CACHE_DIR=cache/'torchinductor',MPLCONFIGDIR=cache/'matplotlib',NUMBA_CACHE_DIR=cache/'numba',PIP_CACHE_DIR=cache/'pip',CONDA_PKGS_DIRS=cache/'conda',
            WANDB_DIR=cache/'wandb',WANDB_CACHE_DIR=cache/'wandb/cache',WANDB_CONFIG_DIR=cache/'wandb/config',
            __GL_SHADER_DISK_CACHE_PATH=cache/'nvidia_shader',MESA_SHADER_CACHE_DIR=cache/'mesa_shader')
        env.update({k:str(v) for k,v in names.items()});env['PYTHONNOUSERSITE']='1';env['PYTHONDONTWRITEBYTECODE']='1'
        # Do not inherit NGC's unrelated torch2.11 library paths.
        va=Path(self.p['va_python']).resolve().parents[1];client=Path(self.p['client_python']).resolve().parents[1]
        env['LD_LIBRARY_PATH']=':'.join(map(str,[Path(self.p['root'])/'userlib',va/'lib',client/'lib',va/'lib/python3.12/site-packages/torch/lib',Path('/usr/local/cuda/compat/lib.real'),Path('/usr/local/cuda/compat/lib'),Path('/usr/local/nvidia/lib'),Path('/usr/local/nvidia/lib64')]))
        return env

    def guard(self,stage):
        super().guard(stage)
        if not hasattr(self,'series_check') or time.monotonic()-self.series_check>=10:
            self.series_check=time.monotonic();total=0
            for root in [self.job,*[Path(v['physical_source_job']) for v in self.p['historical_source_roots']]]:
                for f in root.rglob('*'):
                    if f.is_file():
                        try:total+=f.stat().st_size
                        except FileNotFoundError:continue
            require(total<=self.p['new_evidence_cap_bytes'],'all hardware/generation evidence cap')
            attempts=self.p['prior_target_forward_debit']
            for i in range(4):
                f=self.job/f'server-worker{i}'/'LIVE-COUNTER.json'
                if f.is_file():attempts+=read(f)['target_forward_attempts']
            require(attempts<=self.p['job_forward_attempt_cap'],'all generation attempt cap')

    def waves(self):
        for number,ids in enumerate(self.p['migration_shards']):
            self.guard('whole30_shard_admission')
            if (self.job/'STOP_AFTER_SHARD').exists():break
            counts=[sum(self.p['cases'][cid]['execution_worker_index']==i for cid in ids) for i in range(4)];reserve=max(counts)*3*(1200+120)
            if time.time()+reserve>self.work or time.time()+reserve+900>self.hard:break
            self.atomic_new(self.job/f'SHARD-{number:03d}-STARTED.json',dict(case_ids=ids,at=time.time()))
            def worker(i):
                for cid in ids:
                    if self.p['cases'][cid]['execution_worker_index']!=i:continue
                    if self.hard_event.is_set():return
                    if self.run_case(cid)['status']!='COMPLETE_RECORDED_CASE':return
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(worker,i) for i in range(4)]
                while not all(f.done() for f in futures):
                    if self.hard_event.is_set():self.cleanup(gpu_only=True)
                    time.sleep(.2)
                for f in futures:f.result()
            complete=[cid for cid in ids if self.results.get(cid,{}).get('status')=='COMPLETE_RECORDED_CASE']
            self.atomic_new(self.job/f'SHARD-{number:03d}-RESULT.json',dict(status='COMPLETE_FULL_CASE_SHARD' if complete==ids else 'PARTIAL_NO_RETRY',case_ids=ids,complete_case_ids=complete,complete_episodes=3*len(complete),permissions=PERMISSIONS))
            if complete!=ids or self.hard_event.is_set():break

    def run_report(self):
        # Old raw contexts remain old. This is a multi-host inventory, not an
        # attempt to pass old row() path checks by rewriting old receipts.
        sources={v['case_id']:dict(v) for v in self.p['prior_complete_source_cases']}
        for cid,r in self.results.items():
            if r['status']=='COMPLETE_RECORDED_CASE':sources[cid]=dict(case_id=cid,source_job=str(self.job))
        physical={v['logical_source_job']:v['physical_source_job'] for v in self.p['historical_source_roots']};rows=[]
        for cid,v in sorted(sources.items()):
            logical=v['source_job'];actual=Path(physical.get(logical,logical))
            for arm in ('T','C','P'):
                out=actual/'episodes'/f'case-{cid:04d}'/arm;t=read(out/'EPISODE-TERMINAL.json')
                require(t['complete_episode'] and t['quality_terminal'],'natural complete raw coverage required')
                rows.append(dict(case_id=cid,arm=arm,logical_source=logical,physical_source=str(out),success=t['success'],env_steps=t['env_steps'],failure800_penalty=t['failure800_penalty']))
        self.report_result=dict(status='RECORDED_MULTIHOST_COMPLETE_CASE_COVERAGE',returncode=0,complete_cases=len(sources),complete_episodes=len(rows),episodes=rows,permissions=PERMISSIONS,strict_old_pair_requalification=False,formal_performance=False)
        write(self.job/'REPORT-TERMINAL.json',self.report_result)


def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);x=a.parse_args();job=x.job.resolve(strict=True)
    try:
        p=protocol.load(job);require(not (job/'RUN-STARTED.json').exists(),'one actual lifecycle/no retry')
        runner=MigratedRunner(job,p);rc=runner.execute();close=read(job/'FINAL-OWNED-CLEANUP.json')
        write(job/'MIGRATION-RUN-RESULT.json',dict(status='COMPLETE_RECORDED_LONG500_NO_QUALITY_CERTIFICATE' if rc==0 else 'PARTIAL_OR_FAILED_OR_STAGE_PAUSED',native_returncode=rc,complete_new_cases=[cid for cid,r in runner.results.items() if r['status']=='COMPLETE_RECORDED_CASE'],all_owned_groups_empty=close['all_owned_groups_empty'],permissions=PERMISSIONS))
        return rc
    except BaseException as e:
        f=job/'MIGRATION-RUNNER-ENTRY-FAILED.json'
        if not f.exists():write(f,dict(error=repr(e),traceback=traceback.format_exc(),permissions=PERMISSIONS))
        raise

if __name__=='__main__':raise SystemExit(main())
