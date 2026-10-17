"""Wait for recorded Long500 and strict cleanup, retire only registered cache shards,
then run one fresh original Spatial asset producer. No model/GPU/test/digest.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import errno
import json
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import time
import traceback

PERMISSIONS=dict(engineering_quality_screen=False,independent_test_complete=False,
    noninferiority_established=False,formal_performance_permission=False)
SHARDS={'diffusion_pytorch_model-00001-of-00003.safetensors':4821987820,
    'diffusion_pytorch_model-00002-of-00003.safetensors':4821655760,
    'diffusion_pytorch_model-00003-of-00003.safetensors':535373816}
SPATIAL_BYTES=10177831668
SPATIAL_REVISION='933ab666f8cd21ff71d8765dea9fdd110512034f'
STAT_KEYS=('st_dev','st_ino','st_size','st_mtime_ns','st_uid','st_nlink')

class Blocked(RuntimeError):pass

def require(ok,message):
    if not ok:raise Blocked(message)

def read(path):
    path=Path(path);require(path.stat().st_size<=64*2**20,'oversized registered metadata')
    return json.loads(path.read_text(),parse_constant=lambda value:(_ for _ in ()).throw(ValueError(value)))

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    require(len(raw)<=2**20,'bounded individual rotation metadata receipt')
    with path.open('xb') as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno())

def stamp():return datetime.now(timezone.utc).isoformat()

def process(pid):
    try:
        base=Path('/proc')/str(pid);items=(base/'stat').read_text().rpartition(') ')[2].split()
        return dict(pid=pid,birth_ticks=int(items[19]),pgid=int(items[2]),uid=base.stat().st_uid,state=items[0])
    except (FileNotFoundError,ProcessLookupError):return None

def matches(pin):
    actual=process(pin['pid'])
    return bool(actual and actual['birth_ticks']==pin['birth_ticks'] and actual['uid']==pin['uid']
        and actual['pgid']==pin['pgid']==pin['pid'])

def tree_bytes(root):
    total=0
    if not root.exists():return total
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in files:
            try:
                row=os.lstat(Path(directory)/name)
                if stat.S_ISREG(row.st_mode):total+=row.st_size
            except FileNotFoundError:continue
    return total

def observed_stat(path):
    row=os.lstat(path)
    return dict(**{key:getattr(row,key) for key in STAT_KEYS},regular=stat.S_ISREG(row.st_mode),
        is_symlink=stat.S_ISLNK(row.st_mode))

def group_members(pins):
    wanted={p['pgid'] for p in pins};members=[]
    for base in Path('/proc').iterdir():
        if not base.name.isdigit():continue
        try:
            actual=process(int(base.name))
            if actual and actual['pgid'] in wanted:members.append(actual)
        except (PermissionError,OSError,ValueError,IndexError) as exc:
            raise Blocked('unknown global process-group scan: '+repr(exc))
    return members


class Rotation:
    def __init__(self,job,registration):
        self.job=job.resolve();self.registration=registration.resolve(strict=True);self.p=read(self.registration)
        p=self.p
        require(p['schema']=='tf-suite-asset-rotation/v2' and p['permissions']==PERMISSIONS
            and p['retirement_authorized'] is True,
            'fresh explicit asset rotation registration required')
        require(os.sched_getaffinity(0)=={16} and os.getenv('CUDA_VISIBLE_DEVICES')=='',
            'rotation must use CPU16 only and explicit CUDA_VISIBLE_DEVICES empty')
        require(p['cpu_core']==16 and p['metadata_cap_bytes']==8*2**20
            and p['asset_cap_bytes']==11*2**30 and p['data_floor_bytes']==10*2**30
            and 600<=p['stage_wall_cap_sec']<=7200,'fixed metadata/asset/storage/wall budget required')
        self.root=Path(p['root_path']).resolve(strict=True)
        self.work=datetime.fromisoformat(p['absolute_work_stop']).timestamp()
        self.hard=datetime.fromisoformat(p['absolute_cleanup_deadline']).timestamp()
        origin=datetime.fromisoformat(p['authorization_origin']).timestamp()
        require(origin<self.work<self.hard and self.hard-origin<=96*3600
            and self.hard-self.work>=900,'original finite master lifetime and cleanup reserve required')
        require(self.work>time.time(),'original master work window already expired')
        self.asset=Path(p['asset_root']).resolve();self.stage=Path(p['stage_job']).resolve()
        self.runtime=Path(p['canonical_spatial_runtime']).resolve(strict=True)
        self.ready=Path(p['canonical_spatial_ready'])
        self.shared=Path(p['shared_components']).resolve(strict=True)
        self.child=None;self.child_pin=None;self.deleted=[];self.finished=False
        require(not self.job.exists() and not self.asset.exists() and not self.stage.exists(),
            'fresh rotation/stage/cache only; prior partial assets never resumed')
        require(self.stage.is_relative_to(self.job),'fresh original stage receipts must remain within rotation metadata budget')
        require(not self.asset.is_relative_to(self.job),'large asset cache must not enter metadata-only job')
        require(not self.asset.is_relative_to(self.shared) and not self.shared.is_relative_to(self.asset),
            'fresh Spatial cache must not alter original shared component tree')
        require(self.asset.parent.is_dir(),'registered data cache parent must already exist')
        require(not os.path.lexists(self.runtime/'transformer') and not os.path.lexists(self.ready),
            'canonical Spatial transformer/ready must be absent; never replace an existing artifact')
        canonical=Path(p['canonical_asset_root']).resolve(strict=True)
        require(canonical==Path('/root/wam-suite-assets-20261017-v1')
            and self.runtime==canonical/'spatial/runtime'
            and self.ready==canonical/'spatial/ASSET-READY.json',
            'canonical Spatial scaffold/ready must be the registered owned OS-cache paths')
        require(all((self.runtime/name).resolve(strict=True)==(self.shared/name).resolve(strict=True)
            for name in ('vae','text_encoder','tokenizer')),'existing shared component scaffold differs')
        self.job.mkdir(parents=True,exist_ok=False)
        write(self.job/'REGISTRATION.json',p)
        write(self.job/'STARTED.json',dict(at=stamp(),pid=os.getpid(),birth_ticks=process(os.getpid())['birth_ticks'],
            original_authorization_origin=p['authorization_origin'],waiting_counts_against_original_lifetime=True,
            input_identity='UNKNOWN_NOT_HASHED',CPU=1,GPU=0,permissions=PERMISSIONS))

    def budget(self):
        require(time.time()<self.work and not (self.job/'STOP_REQUESTED').exists(),'original work stop or explicit stop')
        require(tree_bytes(self.job)<=self.p['metadata_cap_bytes'],'rotation metadata8MiB cap')
        require(tree_bytes(self.asset)<=self.p['asset_cap_bytes'],'fresh Spatial asset11GiB cap')

    def current(self,status,**fields):
        value=dict(status=status,at=stamp(),GPU=0,CPU=1,
            authorization_origin=self.p['authorization_origin'],
            absolute_work_stop=self.p['absolute_work_stop'],
            deleted_cache_files=self.deleted,permissions=PERMISSIONS,**fields)
        path=self.job/'status.json';temp=self.job/'status.pending.json'
        with temp.open('x') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n')
        os.replace(temp,path)

    def owned_cleanup(self,job):
        job=Path(job).resolve(strict=True)
        closed=read(job/'ALL-OWNED-CLEANUP.json');guard=read(job/'GUARD-FINAL.json')
        require(closed['all_owned_groups_empty'] is True and guard['status']=='CLEAN',
            'Long job is not fully clean: '+str(job))
        require(closed.get('permissions',PERMISSIONS)==PERMISSIONS,'old permissions cannot change')
        pins=[]
        for file in job.glob('OWNED-*.json'):
            pin=read(file)
            require(pin['owner_job']==str(job) and pin['uid']==os.getuid() and pin['pgid']==pin['pid'],
                'Long owned pin namespace/UID differs')
            actual=process(pin['pid'])
            if actual and actual['birth_ticks']==pin['birth_ticks'] and actual['uid']==pin['uid']:
                return None  # guard/runner exit may follow their final receipt shortly
            pins.append(pin)
        guard_pin_file=job/'GUARD-PROCESS.json'
        require(guard_pin_file.is_file(),'Long independent guard lifetime pin is required')
        guard_pin=read(guard_pin_file)
        require(guard_pin['owner_job']==str(job) and guard_pin['pgid']==guard_pin['pid'],
            'Long guard namespace/PGID differs')
        actual=process(guard_pin['pid'])
        if actual and actual['birth_ticks']==guard_pin['birth_ticks']:
            require(actual['uid']==os.getuid(),'Long guard current UID differs')
            return None
        pins.append(dict(guard_pin,uid=os.getuid()))
        members=group_members(pins)
        if members:return None
        return dict(job=str(job),guard_status=guard['status'],actual_owned_groups_empty=True,pins_checked=len(pins))

    def long_complete(self):
        job=Path(self.p['long_job']).resolve(strict=True);plan=read(job/'plan.json')
        require(plan['schema']=='tf-fullsuite-four-worker-eval/v2' and plan['suite']=='libero_10'
            and len(plan['cases'])==500 and plan['frozen_permissions']==PERMISSIONS,'real Long v2 manifest required')
        supfile=job/'SUPERVISOR-RESULT.json'
        if not supfile.exists():return None
        sup=read(supfile)
        require(sup['schema']=='tf-fullsuite-dual-supervisor/v1' and sup['status'] in
            ('COMPLETE_RECORDED_LONG500_THREE_ARM_COVERAGE','RECORDED_LONG500_WITH_CLOSURE_GAP')
            and sup['failure'] is None and sup['logical_complete_episodes']==1500
            and sup['all_owned_groups_empty'] is True,'Long ended PARTIAL or did not close all500 three-arm cases; no retry')
        prior=Path(plan['prior_job']).resolve(strict=True)
        reused=plan['prior_completed_case_ids'];blocked=plan.get('blocked_prior_partial_case_ids',[])
        require(reused==sorted(set(reused)) and not blocked and all(type(c) is int and 0<=c<500 for c in reused),
            'Long full coverage cannot contain blocked prior partial cases')
        if not all((base/'ALL-OWNED-CLEANUP.json').is_file() and (base/'GUARD-FINAL.json').is_file()
            for base in (job,prior)):return None
        cleanup=[self.owned_cleanup(base) for base in (job,prior)]
        for extra in self.p.get('additional_cleanup_jobs',[]):cleanup.append(self.owned_cleanup(extra))
        if any(value is None for value in cleanup):return None
        require(plan['target_checkpoint_id']==self.p['long_checkpoint_id'],
            'registered retired cache checkpoint differs from actual Long producer')
        self.long_transformer=(Path(plan['target_checkpoint_path']).resolve(strict=True)/'transformer').resolve(strict=True)
        coverage=[]
        for cid,spec in enumerate(plan['cases']):
            self.budget()
            require((spec['case_id'],spec['task'],spec['init'])==(cid,cid%10,cid//10),'Long case manifest identity differs')
            origin=prior if cid in reused else job
            for arm in ('T','C','P'):
                out=origin/'episodes'/f'case-{cid:04d}'/arm
                t,c,a,f=[read(out/name) for name in ('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json')]
                require(t['schema']=='tf-fullsuite-full-episode-terminal/v1' and t['status']=='COMPLETE_RECORDED_FULL_EPISODE'
                    and t['complete_episode'] is True and t['quality_terminal'] is True and t['horizon']==800
                    and type(t['success']) is bool and (t['success'] or t['horizon_reached'] is True)
                    and type(t['env_steps']) is int and 0<t['env_steps']<=800
                    and t['failure800_penalty']==(t['env_steps'] if t['success'] else 800)
                    and t['context']==dict(case_id=cid,arm=arm,out=str(out))
                    and c['schema']=='tf-fullsuite-episode-counter/v1'
                    and c['policy_chunks']==t['policy_chunks'] and a['schema']=='tf-fullsuite-episode-audit/v1'
                    and f['status'] in ('PASS','FAILED_COMPONENT_AUDIT') and f['full_episode_terminal'] is True
                    and f['async_drained'] is True,'Long complete natural/drained terminal missing; failures are not excluded')
                require(all(x['case_id']==cid and x['arm']==arm and x['permissions']==PERMISSIONS for x in (t,c,a,f))
                    and t['task']==spec['task'] and t['init']==spec['init'],'Long original source identity differs')
            coverage.append(dict(case_id=cid,source_job=str(origin),all_three_complete=True))
        return dict(schema='long500-complete-cache-retirement-evidence/v1',cases=coverage,
            complete_cases=500,complete_episodes=1500,reused_cases=reused,cleanup=cleanup,
            raw_component_FAILED_and_algorithm_failures_retained=True,not_a_quality_certificate=True,permissions=PERMISSIONS)

    def retire_checks(self):
        p=self.p;files=p['retire_files'];proof=read(p['backup_proof'])
        require(proof['schema']=='tf-long-local-backup-proof/v1' and proof['root_verified_present'] is True
            and proof['checkpoint_id']==p['long_checkpoint_id'] and proof['input_identity']=='UNKNOWN_NOT_HASHED',
            'root preregistered local backup/source stat proof required')
        require(datetime.fromisoformat(proof['captured_at']).timestamp()<=os.stat(self.registration).st_mtime,
            'backup proof must precede this explicit rotation registration')
        entries={e['entry_id']:e for e in proof['files']}
        require(len(files)==3 and {Path(e['path']).name for e in files}==set(SHARDS),'exact three registered Long shards only')
        snapshots=[]
        parents=set()
        for item in files:
            path=Path(item['path']);expected=item['expected_stat'];backup=entries[item['backup_entry_id']]
            require(path.is_absolute() and path.resolve(strict=True)==path and path.name in SHARDS,
                'retired cache shard must be exact non-symlink canonical path')
            actual=observed_stat(path)
            require(set(expected)==set(STAT_KEYS) and all(type(expected[k]) is int and actual[k]==expected[k] for k in STAT_KEYS)
                and actual['regular'] and not actual['is_symlink'] and actual['st_nlink']==1
                and actual['st_uid']==os.getuid() and actual['st_size']==SHARDS[path.name],
                'Long cache stat/UID/nlink/size no longer matches root registration')
            require(backup['root_verified_present'] is True and backup['publisher_source']
                and Path(backup['local_path']).name==path.name and backup['stat']['st_size']==actual['st_size']
                and backup['stat']['st_nlink']>=1 and backup['stat']['regular'] is True,
                'same-source complete local original backup stat proof absent')
            require(path.parent==self.long_transformer,'cache shard is not from actual registered Long transformer')
            require(all(k in backup['stat'] and type(backup['stat'][k]) is int for k in STAT_KEYS),
                'complete local backup inode/dev/size/mtime/UID/nlink witness required')
            parents.add(str(path.parent));snapshots.append(dict(path=str(path),stat=actual,backup_entry=backup))
        require(len(parents)==1,'all cache shards must share exactly one transformer directory')
        directory=Path(next(iter(parents)))
        require((directory/'config.json').is_file()
            and (directory/'diffusion_pytorch_model.safetensors.index.json').is_file(),
            'retained original config and shard index must exist')
        return snapshots

    def consumers(self,snapshots):
        targets={(x['stat']['st_dev'],x['stat']['st_ino']):x['path'] for x in snapshots}
        hits=[];unknown=[];scanned=0;fds=0;maps=0
        for base in Path('/proc').iterdir():
            if not base.name.isdigit():continue
            pid=int(base.name)
            self.budget()
            try:
                before=process(pid)
                if before is None:continue
                for fd in (base/'fd').iterdir():
                    try:
                        row=fd.stat();fds+=1
                        if (row.st_dev,row.st_ino) in targets:
                            hits.append(dict(pid=pid,birth_ticks=before['birth_ticks'],kind='FD',fd=fd.name,
                                file=targets[row.st_dev,row.st_ino]))
                    except OSError as exc:
                        if exc.errno in (errno.ENOENT,errno.ESRCH):continue
                        raise
                with (base/'maps').open() as stream:
                    for line in stream:
                        if not line.strip():continue
                        fields=line.split(None,5);require(len(fields)>=5,'unknown proc maps row')
                        major,minor=(int(v,16) for v in fields[3].split(':'));inode=int(fields[4]);maps+=1
                        for (device,target_inode),name in targets.items():
                            if inode==target_inode and (major,minor)==(os.major(device),os.minor(device)):
                                hits.append(dict(pid=pid,birth_ticks=before['birth_ticks'],kind='MAP',file=name))
                after=process(pid)
                if after is not None and after['birth_ticks']!=before['birth_ticks']:
                    unknown.append(dict(pid=pid,reason='PID lifetime changed during proc scan'))
                scanned+=1
            except OSError as exc:
                if exc.errno in (errno.ENOENT,errno.ESRCH):continue
                unknown.append(dict(pid=pid,error=repr(exc)))
            except (Blocked,ValueError,IndexError) as exc:unknown.append(dict(pid=pid,error=repr(exc)))
            require(len(hits)+len(unknown)<=1000,'proc scan metadata cap; unsafe to continue')
        receipt=dict(at=stamp(),scanned_visible_processes=scanned,FDs_checked=fds,maps_rows_checked=maps,
            consumers=hits,unknowns=unknown,scope='all visible /proc FD and maps; future opens cannot be excluded',
            safe_to_retire=not hits and not unknown)
        if not receipt['safe_to_retire']:
            blocked=self.job/f'BLOCKED-PROC-SCAN-{time.time_ns()}.json'
            write(blocked,receipt)
            raise Blocked('consumer or permission/identity UNKNOWN blocks retirement; see '+str(blocked))
        return receipt

    def cleanup_child(self):
        if self.child is None:return dict(status='NO_ASSET_CHILD',all_groups_empty=True)
        signals=[]
        for sig,wait in ((signal.SIGTERM,15),(signal.SIGKILL,10)):
            if self.child.poll() is not None:break
            require(self.child_pin is not None and matches(self.child_pin),'asset child lifetime no longer proven; no signal')
            try:os.killpg(self.child_pin['pgid'],sig);signals.append(sig.name)
            except ProcessLookupError:pass
            try:self.child.wait(timeout=max(0,min(wait,self.hard-time.time())))
            except subprocess.TimeoutExpired:continue
        members=group_members([self.child_pin]) if self.child_pin is not None else ['UNKNOWN']
        return dict(status='CLEAN' if self.child.poll() is not None and not members else 'REMAIN_OR_UNKNOWN',
            returncode=self.child.poll(),signals=signals,group_members=members,all_groups_empty=not members)

    def produce(self):
        self.budget();require(time.time()+self.p['stage_wall_cap_sec']+120<self.work,
            'full bounded fresh stage cannot fit remaining original work window')
        require(shutil.disk_usage(self.asset.parent).free>=self.p['data_floor_bytes']+SPATIAL_BYTES+16*2**20,
            'retirement did not create enough actual data-disk space for full fresh Spatial cache plus10GiB floor')
        script=self.root/'投机采样/experiments/tf_suite_assets_v1/stage.py'
        argv=[self.p['python'],'-B',str(script),'--job',str(self.stage),'--asset-root',str(self.asset),
            '--shared-components',str(self.shared),'--source-commit',self.p['source_code_commit'],
            '--suites','spatial','--transport-host',self.p.get('transport_host','huggingface.co'),
            '--wall-cap',str(self.p['stage_wall_cap_sec'])]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',
            OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
        self.current('PRODUCING_PINNED_SPATIAL_ASSET',expected_weight_bytes=SPATIAL_BYTES)
        with (self.job/'stage.log').open('xb') as stream:
            self.child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,
                env=env,cwd=self.root,close_fds=True,start_new_session=True)
        self.child_pin=process(self.child.pid)
        require(self.child_pin is not None and self.child_pin['uid']==os.getuid()
            and self.child_pin['pgid']==self.child.pid,'fresh asset child owned lifetime required')
        write(self.job/f'OWNED-{self.child.pid}.json',dict(**self.child_pin,owner_job=str(self.job),role='spatial-stage',argv=argv))
        until=min(self.work,time.time()+self.p['stage_wall_cap_sec'])
        while self.child.poll() is None:
            self.budget();require(time.time()<until,'fresh asset stage7200s cap')
            require(shutil.disk_usage(self.asset.parent).free>=self.p['data_floor_bytes'],'actual data free floor10GiB')
            time.sleep(1)
        self.child.wait();require(self.child.returncode==0,'original fresh stage FAILED/PARTIAL; never retry')
        result=read(self.stage/'RESULT.json')
        require(result['status']=='COMPLETE_REGISTERED_SUITE_ASSETS' and result['model_loads']==result['GPU']==result['tests']==result['digests']==0
            and len(result['suites'])==1 and result['suites'][0]['suite']=='libero_spatial','only original complete fresh stage may bind canonical Spatial')
        info=read(self.asset/'spatial/ASSET-READY.json')
        require(info==result['suites'][0] and info['publisher_revision']==SPATIAL_REVISION
            and info['checkpoint_id']=='Wjjjh/lingbot-va-libero-spatial'
            and Path(info['model_path']).resolve()==(self.asset/'spatial/runtime').resolve(),
            'fresh Spatial source identity differs')
        transformer=self.asset/'spatial/runtime/transformer'
        require((transformer/'diffusion_pytorch_model.safetensors').stat().st_size==SPATIAL_BYTES,
            'fresh published Spatial weight incomplete')
        cleanup=self.cleanup_child();write(self.job/'STAGE-OWNED-CLEANUP.json',cleanup)
        require(cleanup['all_groups_empty'],'asset child process group must be empty before canonical publication')
        self.budget();require(not os.path.lexists(self.runtime/'transformer') and not os.path.lexists(self.ready),
            'canonical destination changed; no overwrite')
        (self.runtime/'transformer').symlink_to(transformer,target_is_directory=True)
        derived=dict(info,model_path=str(self.runtime),original_fresh_stage_result=str(self.stage/'RESULT.json'),
            original_fresh_asset_ready=str(self.asset/'spatial/ASSET-READY.json'),
            canonical_binding_scope='only new Spatial transformer link; shared components and Long records untouched',
            identity='UNKNOWN_NOT_HASHED',permissions=PERMISSIONS)
        write(self.ready,derived)
        write(self.job/'RESULT.json',dict(status='COMPLETE_REGISTERED_CACHE_ROTATION_AND_SPATIAL_ASSET',
            deleted_cache_files=self.deleted,derived_asset_ready=str(self.ready),asset_bytes=tree_bytes(self.asset),
            original_master_lifetime_not_reset=True,permissions=PERMISSIONS))
        self.finished=True
        self.current('COMPLETE_REGISTERED_CACHE_ROTATION_AND_SPATIAL_ASSET')

    def run(self):
        old={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGALRM)}
        for sig in old:signal.signal(sig,lambda *_:(_ for _ in ()).throw(InterruptedError('owned asset rotation stop/deadline')))
        signal.setitimer(signal.ITIMER_REAL,max(.001,self.work-time.time()))
        failure=None
        try:
            while True:
                self.budget();coverage=self.long_complete()
                if coverage is not None:break
                self.current('WAITING_LONG_FULL_COVERAGE_AND_ACTUAL_CLEANUP')
                time.sleep(10)
            write(self.job/'LONG-COMPLETE-COVERAGE.json',coverage)
            self.current('CHECKING_REGISTERED_CACHE_AND_REAL_CONSUMERS')
            snapshots=self.retire_checks();scan=self.consumers(snapshots)
            write(self.job/'PRE-RETIRE-PROC-SCAN.json',scan)
            write(self.job/'INTENT.json',dict(status='REGISTERED_THREE_CACHE_SHARDS_ONLY',at=stamp(),files=snapshots,
                backup_proof=self.p['backup_proof'],no_shared_component_or_result_deletion=True,
                local_backup_remote_content_identity='UNKNOWN_NOT_HASHED',permissions=PERMISSIONS))
            self.budget();require(self.retire_checks()==snapshots,'cache identity changed after intent')
            scan=self.consumers(snapshots);write(self.job/'IMMEDIATE-PRE-UNLINK-PROC-SCAN.json',scan)
            for item in snapshots:
                self.budget();path=Path(item['path'])
                require(observed_stat(path)==item['stat'],'one cache shard identity changed before unlink')
                path.unlink();self.deleted.append(str(path))
                write(self.job/f'UNLINK-{len(self.deleted)}.json',dict(file=str(path),at=stamp(),
                    path_absent=not os.path.lexists(path),actual_statfs=dict(
                        data_free_bytes=shutil.disk_usage(self.asset.parent).free,
                        retired_volume_free_bytes=shutil.disk_usage(path.parent).free),permissions=PERMISSIONS))
            self.produce();return 0
        except BaseException as exc:
            failure=dict(status='BLOCKED' if isinstance(exc,Blocked) else 'FAILED_OR_PARTIAL_NO_RETRY',
                error=repr(exc),traceback=traceback.format_exc(),deleted_cache_files=self.deleted,
                prior_partial_not_retried=True,identity='UNKNOWN_NOT_HASHED',permissions=PERMISSIONS)
            write(self.job/'STOP-REPORT.json',failure)
            self.current(failure['status'],error=repr(exc))
            return 2
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            try:cleanup=self.cleanup_child()
            except BaseException as exc:cleanup=dict(status='REMAIN_OR_UNKNOWN',error=repr(exc),all_groups_empty=False)
            if not (self.job/'FINAL-OWNED-CLEANUP.json').exists():write(self.job/'FINAL-OWNED-CLEANUP.json',cleanup)
            for sig,handler in old.items():signal.signal(sig,handler)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    fresh=not args.job.exists()
    try:return Rotation(args.job,args.registration).run()
    except BaseException as exc:
        if fresh:
            args.job.mkdir(parents=True,exist_ok=True)
            if not (args.job/'ENTRY-BLOCKED.json').exists():
                write(args.job/'ENTRY-BLOCKED.json',dict(status='BLOCKED_ENTRY_NO_RETRY',error=repr(exc),
                    traceback=traceback.format_exc(),permissions=PERMISSIONS,no_asset_action_claim=True))
        raise


if __name__=='__main__':raise SystemExit(main())
