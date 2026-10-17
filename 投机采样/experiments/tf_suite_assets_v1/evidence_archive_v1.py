"""Bounded production archive of closed Long evidence, without hashes or retries.

Only ``main`` connects to SSH.  The remote controller is this file's independent
stdlib program; it never imports project/model modules or changes old receipts.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import json
import os
from pathlib import Path
import shlex
import signal
import stat
import sys
import time
import traceback

SCHEMA = 'tf-long-evidence-archive-registration/v1'
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
HOST, PORT, USER = 'region-9.autodl.pro', 40416, 'root'
ROOT = Path(__file__).resolve().parents[3]
HOST_KEYS = ROOT/'投机采样/results/runtime/p0-tf-pf2-readonly-plan-v1-20261017-r1/REMOTE-HOST-KEYS'
TOTAL_CAP, FILE_CAP, META_CAP = 12*2**30, 64*2**20, 128*2**20
BULK_SUFFIXES = ('.pt', '.npz', '.npy', '.png', '.jsonl')


class Blocked(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise Blocked(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    p = Path(path)
    require(p.stat().st_size <= META_CAP, 'bounded registration required')
    return json.loads(p.read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def write(path, value, *, current=False):
    p = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    require(len(raw) <= META_CAP, 'bounded archive metadata required')
    p.parent.mkdir(parents=True, exist_ok=True)
    pending = p.with_name(p.name+'.pending') if current else p
    with pending.open('wb' if current else 'xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if current:
        os.replace(pending, p)


REMOTE_PROGRAM = r'''
import errno,json,os,signal,stat,sys,time
from pathlib import Path

P=json.loads(sys.argv[1]);PERMISSIONS=P['permissions']
KEYS=('st_dev','st_ino','st_size','st_mtime_ns','st_uid','st_nlink','st_mode')
META_CAP=128*2**20;FILE_CAP=64*2**20;META_BYTES=0
class Blocked(RuntimeError):pass
def req(ok,msg):
    if not ok:raise Blocked(msg)
def now():return time.time()
def rd(path):
    path=Path(path);req(path.stat().st_size<=META_CAP,'oversized remote metadata')
    return json.loads(path.read_text(),parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def wr(path,value):
    global META_BYTES
    path=Path(path);raw=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    req(len(raw)<=META_CAP and META_BYTES+len(raw)<=META_CAP,'remote total metadata128MiB cap')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    META_BYTES+=len(raw)
def proc(pid):
    base=Path('/proc')/str(pid)
    try:
        fields=(base/'stat').read_text().rpartition(') ')[2].split()
        return dict(pid=pid,birth_ticks=int(fields[19]),pgid=int(fields[2]),uid=base.stat().st_uid,state=fields[0])
    except (FileNotFoundError,ProcessLookupError):return None
def snap(path):
    s=os.lstat(path)
    return {k:getattr(s,k) for k in KEYS}
def check():
    req(now()<P['work_epoch'],'original absolute work stop; no new action')
    req(not (META/'STOP_REQUESTED').exists(),'archive STOP_REQUESTED')
    req(META_BYTES<=META_CAP,'remote archive owned metadata128MiB cap')
def group_empty(pins):
    wanted={int(p['pgid']) for p in pins};members=[]
    for base in Path('/proc').iterdir():
        if not base.name.isdigit():continue
        try:
            item=proc(int(base.name))
            if item and item['pgid'] in wanted:members.append(item)
        except OSError as e:
            if e.errno in (errno.ENOENT,errno.ESRCH):continue
            raise Blocked('UNKNOWN process-group scan: '+repr(e))
    return members
def cleanup(job):
    files=(job/'ALL-OWNED-CLEANUP.json',job/'GUARD-FINAL.json')
    if not all(x.is_file() for x in files):return None
    closed,guard=map(rd,files)
    req(closed['all_owned_groups_empty'] is True and guard['status']=='CLEAN','old job recorded cleanup not CLEAN')
    pins=[]
    paths=sorted(job.glob('OWNED-*.json'))
    req(paths and (job/'GUARD-PROCESS.json').is_file(),'missing actual Long ownership pins')
    for path in paths+[job/'GUARD-PROCESS.json']:
        p=rd(path)
        req(p['owner_job']==str(job) and p['pgid']==p['pid'],'old ownership namespace/PGID differs')
        uid=p.get('uid',os.getuid());req(uid==os.getuid(),'old owned UID differs')
        actual=proc(p['pid'])
        if actual and actual['birth_ticks']==p['birth_ticks']:
            req(actual['uid']==uid,'old pin UID changed')
            return None
        pins.append(dict(p,uid=uid))
    members=group_empty(pins)
    if members:return None
    return dict(job=str(job),actual_owned_groups_empty=True,pins_checked=len(pins),recorded_guard='CLEAN')
def long_gate():
    check();job=ROOTS['current'];prior=ROOTS['prior'];plan=rd(job/'plan.json')
    req(plan['schema']=='tf-fullsuite-four-worker-eval/v2' and plan['suite']=='libero_10'
        and len(plan['cases'])==500 and plan['frozen_permissions']==PERMISSIONS,'fixed actual Long500 source required')
    req(Path(plan['prior_job']).resolve(strict=True)==prior,'actual Long prior root differs')
    for name,root in ROOTS.items():
        old=rd(root/'plan.json')
        req(old['source_code_commit']==P['expected_source_commits'][name],'actual registered source commit differs')
    supfile=job/'SUPERVISOR-RESULT.json'
    if not supfile.is_file():return dict(ready=False,reason='WAITING_LONG_SUPERVISOR')
    sup=rd(supfile)
    req(sup['schema']=='tf-fullsuite-dual-supervisor/v1' and sup['status'] in
        ('COMPLETE_RECORDED_LONG500_THREE_ARM_COVERAGE','RECORDED_LONG500_WITH_CLOSURE_GAP')
        and sup['failure'] is None and sup['logical_complete_episodes']==1500
        and sup['all_owned_groups_empty'] is True,'Long PARTIAL/FAILED is not full1500; no automatic retry')
    reused=plan['prior_completed_case_ids']
    req(reused==sorted(set(reused)) and not plan.get('blocked_prior_partial_case_ids',[])
        and all(type(c) is int and 0<=c<500 for c in reused),'partial or ambiguous prior coverage')
    clean=[cleanup(root) for root in ROOTS.values()]
    if any(x is None for x in clean):return dict(ready=False,reason='WAITING_REAL_LONG_OWNED_EXIT')
    successes={a:0 for a in ('T','C','P')};coverage=[]
    for cid,spec in enumerate(plan['cases']):
        check()
        req((spec['case_id'],spec['task'],spec['init'])==(cid,cid%10,cid//10),'actual Long case identity differs')
        origin=prior if cid in reused else job
        for arm in ('T','C','P'):
            out=origin/'episodes'/('case-%04d'%cid)/arm
            terminal,counter,audit,finish=[rd(out/name) for name in
                ('EPISODE-TERMINAL.json','COUNTER.json','AUDIT.json','FINISH_READY.json')]
            req(terminal['schema']=='tf-fullsuite-full-episode-terminal/v1'
                and terminal['status']=='COMPLETE_RECORDED_FULL_EPISODE'
                and terminal['complete_episode'] is True and terminal['quality_terminal'] is True
                and terminal['horizon']==800 and type(terminal['success']) is bool
                and (terminal['success'] or terminal['horizon_reached'] is True)
                and type(terminal['env_steps']) is int and 0<terminal['env_steps']<=800
                and terminal['failure800_penalty']==(terminal['env_steps'] if terminal['success'] else 800)
                and terminal['context']==dict(case_id=cid,arm=arm,out=str(out))
                and counter['schema']=='tf-fullsuite-episode-counter/v1'
                and counter['policy_chunks']==terminal['policy_chunks']
                and audit['schema']=='tf-fullsuite-episode-audit/v1'
                and finish['status'] in ('PASS','FAILED_COMPONENT_AUDIT')
                and finish['full_episode_terminal'] is True and finish['async_drained'] is True,
                'missing complete natural/drained Long episode; failures must remain')
            req(all(x['case_id']==cid and x['arm']==arm and x['permissions']==PERMISSIONS
                for x in (terminal,counter,audit,finish)) and terminal['task']==spec['task']
                and terminal['init']==spec['init'],'original episode identity differs')
            successes[arm]+=int(terminal['success'])
        coverage.append(dict(case_id=cid,source_job=str(origin),all_three_complete=True))
    return dict(ready=True,complete_cases=500,complete_episodes=1500,success_counts=successes,
        coverage=coverage,cleanup=clean,algorithm_failures_and_component_FAILED_retained=True,
        not_a_quality_certificate=True,permissions=PERMISSIONS)
def inventory():
    entries=[];directories=[];total=0
    for alias,root in ROOTS.items():
        for directory,dirs,files in os.walk(root,followlinks=False):
            check();directory=Path(directory)
            req(directory.resolve(strict=True)==directory and stat.S_ISDIR(os.lstat(directory).st_mode),
                'source directory symlink/special is BLOCKED')
            directories.append(dict(root=alias,relative=directory.relative_to(root).as_posix(),stat=snap(directory)))
            for name in dirs:
                path=directory/name;s=os.lstat(path)
                req(stat.S_ISDIR(s.st_mode) and not stat.S_ISLNK(s.st_mode),'source contains non-directory/symlink: '+str(path))
            for name in files:
                path=directory/name;s=snap(path)
                req(stat.S_ISREG(s['st_mode']) and path.resolve(strict=True)==path,
                    'source symlink/special file is BLOCKED: '+str(path))
                req(s['st_size']<=FILE_CAP,'source file exceeds64MiB; no silent skip/split: '+str(path)+' bytes='+str(s['st_size']))
                total+=s['st_size'];req(total<=P['total_cap_bytes']-META_CAP,'complete raw archive plus metadata exceeds12GiB cap')
                entries.append(dict(root=alias,relative=path.relative_to(root).as_posix(),path=str(path),stat=s))
    entries.sort(key=lambda x:(x['root'],x['relative']));directories.sort(key=lambda x:(x['root'],x['relative']))
    return dict(schema='tf-long-raw-evidence-manifest/v1',roots={k:str(v) for k,v in ROOTS.items()},
        files=entries,directories=directories,file_count=len(entries),total_bytes=total,
        maximum_file_bytes=FILE_CAP,no_symlink_or_special=True,no_SHA=True,permissions=PERMISSIONS)
def exact_manifest(expected):
    actual=inventory();req(actual==expected,'remote full file/stat manifest changed; no deletion')
    return actual
def file_check(entry):
    req(entry['root'] in ROOTS,'unapproved source root')
    root=ROOTS[entry['root']];path=root/entry['relative']
    req(path.is_absolute() and path.resolve(strict=True)==path and path.is_relative_to(root)
        and str(path)==entry['path'] and snap(path)==entry['stat'],'registered original file stat/path changed')
    return dict(path=str(path),stat=entry['stat'])
def rotation_gate():
    job=Path(P['rotation_job']).resolve(strict=True)
    req(job.is_relative_to(APP/'投机采样/results/runtime') and job not in ROOTS.values(),'registered independent rotation root required')
    needed=('RESULT.json','LONG-COMPLETE-COVERAGE.json','STAGE-OWNED-CLEANUP.json','FINAL-OWNED-CLEANUP.json','STARTED.json')
    if not all((job/name).is_file() for name in needed):return None
    result,coverage,stage,final,started=[rd(job/name) for name in needed]
    req(result['status']=='COMPLETE_REGISTERED_CACHE_ROTATION_AND_SPATIAL_ASSET'
        and result['permissions']==PERMISSIONS and coverage['complete_cases']==500
        and coverage['complete_episodes']==1500 and stage['all_groups_empty'] is True
        and final['all_groups_empty'] is True,'rotation/full-counter prior consumption not fully complete')
    actual=proc(started['pid'])
    if actual and actual['birth_ticks']==started['birth_ticks']:return None
    pins=[]
    for path in job.glob('OWNED-*.json'):
        pin=rd(path);req(pin['owner_job']==str(job) and pin['uid']==os.getuid()
            and pin['pgid']==pin['pid'],'rotation owned stage identity differs')
        actual=proc(pin['pid'])
        if actual and actual['birth_ticks']==pin['birth_ticks']:return None
        pins.append(pin)
    req(pins,'rotation stage actual ownership pin missing')
    if group_empty(pins):return None
    return dict(job=str(job),status=result['status'],actual_rotation_and_stage_exited=True,
        full_counter_coverage_saved=str(job/'LONG-COMPLETE-COVERAGE.json'))
def allowed(entry):
    path=Path(entry['relative']);lower=str(path).lower();name=path.name
    if any(word in lower for word in ('failed','proof','plan')):return False
    if name=='COUNTER.json':
        return P['delete_counter_json'] is True and len(path.parts)==4 and path.parts[0]=='episodes' and path.parts[2] in ('T','C','P')
    return path.suffix in tuple(P['bulk_suffixes']) and path.suffix in ('.pt','.npz','.npy','.png','.jsonl')
def consumers(entries):
    targets={(e['stat']['st_dev'],e['stat']['st_ino']):e['path'] for e in entries}
    hits=[];unknown=[];count=0;fds=0;maps=0
    for base in Path('/proc').iterdir():
        if not base.name.isdigit():continue
        check();pid=int(base.name)
        try:
            before=proc(pid)
            if before is None:continue
            for fd in (base/'fd').iterdir():
                try:
                    s=fd.stat();fds+=1
                    if (s.st_dev,s.st_ino) in targets:hits.append(dict(pid=pid,kind='FD',fd=fd.name,path=targets[s.st_dev,s.st_ino]))
                except OSError as exc:
                    if exc.errno not in (errno.ENOENT,errno.ESRCH):raise
            with (base/'maps').open() as stream:
                for line in stream:
                    fields=line.split(None,5);req(len(fields)>=5,'unknown proc maps record')
                    major,minor=(int(x,16) for x in fields[3].split(':'));inode=int(fields[4]);maps+=1
                    if inode:
                        device=os.makedev(major,minor)
                        if (device,inode) in targets:hits.append(dict(pid=pid,kind='MAP',path=targets[device,inode]))
            after=proc(pid)
            if after and after['birth_ticks']!=before['birth_ticks']:unknown.append(dict(pid=pid,reason='PID lifetime changed'))
            count+=1
        except OSError as exc:
            if exc.errno not in (errno.ENOENT,errno.ESRCH):unknown.append(dict(pid=pid,error=repr(exc)))
        except (ValueError,IndexError,Blocked) as exc:unknown.append(dict(pid=pid,error=repr(exc)))
        req(len(hits)+len(unknown)<=1000,'consumer scan diagnostic cap')
    result=dict(scanned_processes=count,FDs_checked=fds,maps_rows_checked=maps,consumers=hits,unknowns=unknown,
        safe_to_retire=not hits and not unknown,scope='all visible /proc FD+maps; future opens cannot be excluded')
    wr(META/'PRE-DELETE-PROC-SCAN.json',result)
    req(result['safe_to_retire'],'consumer or UNKNOWN process scan blocks remote duplicate removal')
    return result
def dispatch(request):
    check();op=request['op']
    if op=='gate':
        result=long_gate()
        if result['ready'] and P['delete_counter_json']:
            rotation=rotation_gate()
            if rotation is None:return dict(ready=False,reason='WAITING_ROTATION_AND_ACTUAL_STAGE_EXIT_FOR_COUNTERS')
            result['counter_retirement_gate']=rotation
        return result
    if op=='manifest':
        gate=long_gate();req(gate['ready'],'full Long gate is not ready')
        if P['delete_counter_json']:req(rotation_gate() is not None,'rotation gate is not ready')
        manifest=inventory();wr(META/'SOURCE-MANIFEST.json',manifest);wr(META/'LONG-COMPLETE-GATE.json',gate)
        return manifest
    if op=='stat':return file_check(request['entry'])
    if op=='commit_copy':
        exact_manifest(request['manifest']);gate=long_gate();req(gate['ready'],'cleanup no longer strict')
        receipt=request['receipt'];req(receipt['status']=='COMPLETE_LOCAL_RAW_COPY_NO_HASH'
            and receipt['files_copied']==request['manifest']['file_count']
            and receipt['bytes_copied']==request['manifest']['total_bytes'],'incomplete local copy receipt')
        req(rd(META/'SOURCE-MANIFEST.json')==request['manifest'],'original stored source manifest differs')
        wr(META/'COPY-INDEX.json',dict(source_manifest=str(META/'SOURCE-MANIFEST.json'),
            file_count=request['manifest']['file_count'],total_bytes=request['manifest']['total_bytes']))
        wr(META/'COPY-RESULT.json',receipt)
        return dict(status='COPY_PROOF_RECORDED_BEFORE_DELETE',remote_metadata_job=str(META))
    if op=='delete':
        index=rd(META/'COPY-INDEX.json');receipt=rd(META/'COPY-RESULT.json')
        req(index['source_manifest']==str(META/'SOURCE-MANIFEST.json'),'copy index must reference original owned manifest')
        manifest=rd(META/'SOURCE-MANIFEST.json')
        req(index['file_count']==manifest['file_count'] and index['total_bytes']==manifest['total_bytes'],
            'copy index counts differ from original stored manifest')
        req(receipt==request['receipt'],'persistent local copy receipt differs')
        exact_manifest(manifest);gate=long_gate();req(gate['ready'],'strict Long cleanup changed before delete')
        rotation=rotation_gate() if P['delete_counter_json'] else None
        req(not P['delete_counter_json'] or rotation is not None,'rotation gate changed before counter delete')
        entries=[e for e in manifest['files'] if allowed(e)]
        for entry in entries:
            file_check(entry);s=entry['stat'];req(s['st_uid']==os.getuid() and s['st_nlink']==1,
                'deletion requires exact owned UID and nlink1')
        intent=dict(schema='tf-long-local-backup-remote-duplicate-delete-intent/v1',
            source_manifest=str(META/'SOURCE-MANIFEST.json'),
            files_manifest_indices=[i for i,e in enumerate(manifest['files']) if allowed(e)],
            local_archive_root=receipt['local_archive_root'],local_copy_receipt=receipt,
            rotation_gate=rotation,bulk_suffixes=P['bulk_suffixes'],delete_counter_json=P['delete_counter_json'],
            preserved='all other JSON, FAILED, md/csv/plan/proof and old statuses stay untouched',
            no_SHA=True,content_identity='UNKNOWN_NOT_HASHED; size/stat transport witness only',permissions=PERMISSIONS)
        wr(META/'DELETE-INTENT.json',intent);scan=consumers(entries);deleted=[];freed=0
        for i,entry in enumerate(entries):
            check();path=Path(entry['path']);file_check(entry)
            fd=os.open(str(path.parent),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            try:
                s=os.stat(path.name,dir_fd=fd,follow_symlinks=False)
                req({k:getattr(s,k) for k in KEYS}==entry['stat'],'immediate pre-unlink original stat differs')
                os.unlink(path.name,dir_fd=fd);os.fsync(fd)
            finally:os.close(fd)
            freed+=entry['stat']['st_size'];deleted.append(entry['path'])
            v=os.statvfs(path.parent)
            wr(META/('UNLINK-%06d.json'%i),dict(path=entry['path'],original_stat=entry['stat'],
                local_original=str(Path(receipt['local_archive_root'])/entry['root']/entry['relative']),
                path_absent=not os.path.lexists(path),actual_statfs_free_bytes=v.f_bavail*v.f_frsize,
                at_epoch=now(),no_SHA=True,permissions=PERMISSIONS))
        result=dict(status='COMPLETE_ARCHIVE_AND_REGISTERED_REMOTE_DUPLICATE_REMOVAL',deleted_files=len(deleted),
            deleted_bytes=freed,retained_local_archive=receipt['local_archive_root'],remote_metadata_job=str(META),
            all_original_raw_files_preserved_locally=True,original_JSON_statuses_not_rewritten=True,
            counter_json_moved_only_after_rotation=P['delete_counter_json'],no_SHA=True,permissions=PERMISSIONS)
        wr(META/'DELETE-RESULT.json',result);return result
    if op=='shutdown':return dict(status='REMOTE_CONTROLLER_EXITING',pin=PIN)
    raise Blocked('unknown archive operation')

req(os.getuid()==0 and os.sched_getaffinity(0)=={16} and os.environ.get('CUDA_VISIBLE_DEVICES')=='',
    'remote archive controller CPU16/noCUDA/root required')
APP=Path(P['remote_app']).resolve(strict=True);META=Path(P['remote_metadata_job'])
ROOTS={name:Path(P['long_'+name+'_job']).resolve(strict=True) for name in ('current','prior')}
req(len(set(ROOTS.values()))==2 and all(root.is_relative_to(APP/'投机采样/results/runtime') for root in ROOTS.values()),
    'exact two independent approved job roots required')
req(META.is_absolute() and META.parent.resolve(strict=True)==APP/'投机采样/results/runtime'
    and not META.exists() and all(not META.is_relative_to(root) for root in ROOTS.values()),'fresh independent archive metadata job required')
META.mkdir(exist_ok=False);PIN=proc(os.getpid())
wr(META/'REGISTRATION.json',P);wr(META/'OWNED-CONTROLLER.json',dict(**PIN,owner_job=str(META)))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('remote original master work stop')))
signal.setitimer(signal.ITIMER_REAL,max(.001,P['work_epoch']-now()))
print(json.dumps(dict(kind='PIN',pin=PIN,remote_metadata_job=str(META))),flush=True)
try:
    for line in sys.stdin:
        req(len(line)<=META_CAP,'bounded remote RPC request')
        request=json.loads(line)
        try:
            value=dispatch(request);print(json.dumps(dict(ok=True,result=value),ensure_ascii=False,allow_nan=False),flush=True)
            if request['op']=='shutdown':break
        except Exception as exc:
            wr(META/'BLOCKED.json',dict(status='BLOCKED_NO_RETRY',error=repr(exc),operation=request.get('op'),permissions=PERMISSIONS))
            print(json.dumps(dict(ok=False,error=repr(exc),type=type(exc).__name__)),flush=True);break
finally:
    signal.setitimer(signal.ITIMER_REAL,0)
    wr(META/'CONTROLLER-EXIT.json',dict(status='EXITING_NO_REMOTE_CHILD_CREATED',pin=PIN,
        instance_shutdown=False,no_GPU_process_created=True,permissions=PERMISSIONS))
'''


class Archive:
    def __init__(self, job, registration):
        self.job = Path(job).resolve()
        self.p = read(registration)
        p = self.p
        def no_credentials(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    require(not any(word in str(key).lower() for word in
                        ('password','passwd','secret','private_key','access_token','credentials')),
                        'credentials are forbidden in archive registration')
                    no_credentials(child)
            elif isinstance(value, list):
                for child in value:
                    no_credentials(child)
        no_credentials(p)
        require(p['schema'] == SCHEMA and p['permissions'] == PERMISSIONS and p['retirement_authorized'] is True,
            'explicit registered archive and duplicate-retirement authorization required')
        require((p['host'], p['port'], p['username']) == (HOST, PORT, USER), 'fixed AutoDL host identity required')
        require(len(os.sched_getaffinity(0)) == 1 and os.getenv('CUDA_VISIBLE_DEVICES') == '',
            'local archive must use CPU1 and explicit hidden CUDA')
        require(p['total_cap_bytes'] == TOTAL_CAP and p['per_file_cap_bytes'] == FILE_CAP
            and p['transfer_wall_cap_sec'] == 10800 and p['metadata_cap_bytes'] == META_CAP,
            'registered12GiB/64MiB/10800s budgets required')
        require(tuple(p['bulk_suffixes']) == BULK_SUFFIXES and type(p['delete_counter_json']) is bool,
            'exact bounded duplicate allowlist required; logs retained')
        require(not p['delete_counter_json'] or isinstance(p.get('rotation_job'), str),
            'COUNTER retirement requires explicit completed rotation job')
        require(set(p['expected_source_commits']) == {'current', 'prior'}
            and all(isinstance(x, str) and x for x in p['expected_source_commits'].values())
            and isinstance(p['source_code_commit'], str) and p['source_code_commit'], 'bound source provenance required')
        self.origin, self.work, self.hard = [datetime.fromisoformat(p[k]).timestamp() for k in
            ('authorization_origin', 'absolute_work_stop', 'absolute_cleanup_deadline')]
        require(self.origin <= time.time() < self.work < self.hard and self.hard-self.origin <= 96*3600
            and self.hard-self.work >= 900, 'original17:11 master96h lifetime and cleanup reserve required')
        self.archive = Path(p['local_archive_root']).resolve()
        self.pending = self.archive.with_name(self.archive.name+'.pending')
        require(not self.job.exists() and not self.archive.exists() and not self.pending.exists()
            and self.archive.parent.is_dir(), 'fresh local metadata/archive/partial roots required; no resume')
        require(not self.job.is_relative_to(self.archive) and not self.job.is_relative_to(self.pending),
            'archive must not contain its own metadata job')
        require(HOST_KEYS.is_file(), 'existing fixed REMOTE-HOST-KEYS required')
        self.job.mkdir(parents=True, exist_ok=False)
        write(self.job/'REGISTRATION.json', p)
        proc = Path('/proc/self/stat').read_text().rpartition(') ')[2].split()
        write(self.job/'STARTED.json', dict(pid=os.getpid(), birth_ticks=int(proc[19]),
            pgid=os.getpgrp(), CPU_affinity=sorted(os.sched_getaffinity(0)), at=stamp(), GPU=0,
            original_authorization_origin=p['authorization_origin'], waiting_counts_against_master_lifetime=True,
            credential_storage=False, no_SHA=True, permissions=PERMISSIONS))
        self.client = self.channel = self.sftp = self.stdin = self.stdout = None
        self.remote_pin = None
        self.transfer_deadline = None
        self.copied = self.bytes = 0
        self.finished = False

    def budget(self):
        deadline = min(self.work, self.transfer_deadline or self.work)
        require(time.time() < deadline and not (self.job/'STOP_REQUESTED').exists(), 'archive absolute deadline or STOP_REQUESTED')
        metadata = 0
        for directory, dirs, files in os.walk(self.job, followlinks=False):
            directory = Path(directory)
            dirs[:] = [name for name in dirs if (directory/name) not in (self.archive,self.pending)]
            for name in files:
                path = directory/name
                require(not path.is_symlink(), 'local metadata symlink is BLOCKED')
                metadata += path.stat().st_size
        require(metadata <= META_CAP and self.bytes+metadata <= TOTAL_CAP, 'local total/metadata archive cap')

    def current(self, status, **fields):
        write(self.job/'status.json', dict(status=status, at=stamp(), files_copied=self.copied,
            bytes_copied=self.bytes, authorization_origin=self.p['authorization_origin'],
            absolute_work_stop=self.p['absolute_work_stop'], CPU=1, GPU=0, permissions=PERMISSIONS, **fields), current=True)

    def connect(self):
        import paramiko
        self.client = paramiko.SSHClient()
        self.client.load_host_keys(str(HOST_KEYS))
        self.client.set_missing_host_key_policy(paramiko.RejectPolicy())
        secret = getpass.getpass('AutoDL SSH password (RAM only): ')
        try:
            self.client.connect(HOST, port=PORT, username=USER, password=secret, timeout=30,
                banner_timeout=30, auth_timeout=30, look_for_keys=False, allow_agent=False)
        finally:
            secret = None
        transport = self.client.get_transport()
        transport.set_keepalive(10)
        cfg = dict(self.p, work_epoch=self.work)
        self.channel = transport.open_session(window_size=128*2**20, max_packet_size=2**20)
        self.channel.settimeout(300)
        command = ['env', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1',
            'TF_EVIDENCE_ARCHIVE_OWNER_JOB='+self.p['remote_metadata_job'], 'taskset', '-c', '16',
            self.p['remote_python'], '-B', '-u', '-c', REMOTE_PROGRAM, json.dumps(cfg, ensure_ascii=False)]
        self.channel.exec_command(shlex.join(command))
        self.stdin = self.channel.makefile('wb')
        self.stdout = self.channel.makefile('rb')
        raw = self.stdout.readline(META_CAP+1)
        require(len(raw) <= META_CAP and raw, 'remote controller failed before ownership pin')
        pin = json.loads(raw)
        require(pin['kind'] == 'PIN' and pin['remote_metadata_job'] == self.p['remote_metadata_job'], 'remote archive ownership pin differs')
        self.remote_pin = pin['pin']
        write(self.job/'REMOTE-CONTROLLER-PIN.json', pin)
        self.sftp = paramiko.SFTPClient.from_transport(transport, window_size=128*2**20, max_packet_size=2**20)
        self.sftp.get_channel().settimeout(60)

    def rpc(self, op, **fields):
        self.budget()
        raw = (json.dumps(dict(op=op, **fields), ensure_ascii=False, allow_nan=False)+'\n').encode()
        require(len(raw) <= META_CAP, 'bounded archive RPC request')
        self.stdin.write(raw)
        self.stdin.flush()
        raw = self.stdout.readline(META_CAP+1)
        require(raw and len(raw) <= META_CAP, 'archive RPC lost response or exceeded cap')
        value = json.loads(raw)
        require(value.get('ok') is True, 'remote archive BLOCKED: '+str(value.get('error', 'missing response')))
        return value['result']

    def copy(self, manifest):
        require(manifest['schema'] == 'tf-long-raw-evidence-manifest/v1'
            and manifest['file_count'] == len(manifest['files'])
            and manifest['total_bytes'] <= TOTAL_CAP-META_CAP, 'complete bounded manifest required')
        self.pending.mkdir(parents=True, exist_ok=False)
        partials = self.pending/'file-transfer.pending'
        partials.mkdir(exist_ok=False)
        for directory in manifest['directories']:
            alias, relative = directory['root'], directory['relative']
            require(alias in ('current', 'prior') and not Path(relative).is_absolute()
                and '..' not in Path(relative).parts, 'confined archive directory required')
            (self.pending/alias/relative).mkdir(parents=True, exist_ok=True)
        start = time.time()
        self.transfer_deadline = min(self.work, start+self.p['transfer_wall_cap_sec'])
        write(self.job/'TRANSFER-STARTED.json', dict(at=stamp(), transfer_deadline_epoch=self.transfer_deadline,
            original_master_deadline_unchanged=True, manifest_bytes=manifest['total_bytes']))
        for ordinal, entry in enumerate(manifest['files']):
            self.budget()
            require(entry['root'] in ('current', 'prior') and not Path(entry['relative']).is_absolute()
                and '..' not in Path(entry['relative']).parts and entry['stat']['st_size'] <= FILE_CAP,
                'bounded confined individual archive file required')
            pre = self.rpc('stat', entry=entry)
            dest = self.pending/entry['root']/entry['relative']
            partial = partials/('%08d.pending'%ordinal)
            require(not dest.exists() and not partial.exists(), 'no automatic file retry or overwrite')
            received = 0
            with self.sftp.open(entry['path'], 'rb', bufsize=2**20) as source, partial.open('xb') as target:
                while True:
                    self.budget()
                    block = source.read(2**20)
                    if not block:
                        break
                    received += len(block)
                    require(received <= entry['stat']['st_size'] and self.bytes+received <= TOTAL_CAP-META_CAP,
                        'stream exceeded exact source bytes or total cap')
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            post = self.rpc('stat', entry=entry)
            require(pre == post and received == entry['stat']['st_size'] == partial.stat().st_size,
                'file source pre/post stat or local byte count differs')
            partial.rename(dest)
            self.copied += 1
            self.bytes += received
            self.current('COPYING_ALL_ORIGINAL_LONG_EVIDENCE', current_source=entry['path'],
                expected_files=manifest['file_count'], expected_bytes=manifest['total_bytes'])
        require(self.copied == manifest['file_count'] and self.bytes == manifest['total_bytes'], 'full raw copy incomplete')
        partials.rmdir()
        self.pending.rename(self.archive)
        fd = os.open(str(self.archive.parent), os.O_RDONLY|os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        # All source files, including all JSON/FAILED and natural failures, are
        # retained locally before the remote controller can receive delete.
        write(self.job/'INDEX.json', dict(schema='tf-long-complete-local-raw-archive-index/v1',
            local_archive_root=str(self.archive),source_manifest=str(self.job/'SOURCE-MANIFEST.json'),
            file_count=manifest['file_count'],total_bytes=manifest['total_bytes'],no_SHA=True,
            content_identity='UNKNOWN_NOT_HASHED; authenticated transport plus exact source stats and local bytes',
            permissions=PERMISSIONS))
        receipt = dict(status='COMPLETE_LOCAL_RAW_COPY_NO_HASH', local_archive_root=str(self.archive),
            local_index=str(self.job/'INDEX.json'), files_copied=self.copied, bytes_copied=self.bytes,
            all_regular_originals_copied=True, pre_post_remote_stats_equal=True, local_byte_counts_exact=True,
            metadata_and_FAILED_preserved=True, no_SHA=True, at=stamp(), permissions=PERMISSIONS)
        write(self.job/'COPY-RESULT.json', receipt)
        return receipt

    def close(self):
        state = dict(owned_SFTP_channel_closed=False, owned_RPC_channel_closed=False,
            remote_controller_exit='UNKNOWN', remote_pin=self.remote_pin,
            remote_GPU_processes_created=0, instance_shutdown='NOT_REQUESTED_NOT_EXECUTED')
        try:
            if self.sftp is not None:
                self.sftp.close()
            state['owned_SFTP_channel_closed'] = True
            if self.channel is not None and not self.channel.closed:
                if self.stdin is not None:
                    self.stdin.close()
                self.channel.shutdown_write()
                until = min(self.hard, time.time()+10)
                while time.time() < until and not self.channel.exit_status_ready():
                    time.sleep(.1)
                if self.channel.exit_status_ready():
                    state['remote_controller_exit'] = dict(returncode=self.channel.recv_exit_status())
        except Exception as exc:
            state['cleanup_error'] = type(exc).__name__
        finally:
            for item in (self.stdout, self.stdin, self.channel, self.client):
                if item is not None:
                    try:
                        item.close()
                    except Exception:
                        pass
            state['owned_RPC_channel_closed'] = True
        write(self.job/'CLEANUP.json', state)
        return state

    def run(self):
        handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM)}
        for sig in handlers:
            signal.signal(sig, lambda *_: (_ for _ in ()).throw(InterruptedError('registered archive signal/deadline')))
        signal.setitimer(signal.ITIMER_REAL, max(.001, self.work-time.time()))
        try:
            self.connect()
            while True:
                gate = self.rpc('gate')
                if gate['ready']:
                    break
                self.current('WAITING_FULL1500_AND_STRICT_OWNED_CLEANUP', reason=gate['reason'])
                time.sleep(min(60, max(.001, self.work-time.time())))
            write(self.job/'LONG-COMPLETE-GATE.json', gate)
            manifest = self.rpc('manifest')
            write(self.job/'SOURCE-MANIFEST.json', manifest)
            receipt = self.copy(manifest)
            self.rpc('commit_copy', manifest=manifest, receipt=receipt)
            intent = dict(status='REQUEST_ONLY_AFTER_DURABLE_FULL_LOCAL_ARCHIVE', local_copy_receipt=receipt,
                bulk_suffixes=self.p['bulk_suffixes'], delete_counter_json=self.p['delete_counter_json'],
                remote_metadata_job=self.p['remote_metadata_job'], no_SHA=True, permissions=PERMISSIONS)
            write(self.job/'DELETE-INTENT.json', intent)
            result = self.rpc('delete', receipt=receipt)
            write(self.job/'RESULT.json', result)
            self.rpc('shutdown')
            self.finished = True
            self.current(result['status'])
            return 0
        except BaseException as exc:
            failure = dict(status='BLOCKED_OR_PARTIAL_ARCHIVE_NO_RETRY', error=repr(exc),
                traceback=traceback.format_exc(), files_copied=self.copied, bytes_copied=self.bytes,
                local_archive_root=str(self.archive), partial_root=str(self.pending),
                partial_files_retained=True, original_master_lifetime_not_reset=True,
                remote_partial_deletion_receipts=self.p['remote_metadata_job'], permissions=PERMISSIONS)
            write(self.job/'FAILED.json', failure)
            self.current(failure['status'], error=repr(exc))
            return 2
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            try:
                self.close()
            finally:
                for sig, old in handlers.items():
                    signal.signal(sig, old)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--registration', type=Path, required=True)
    args = parser.parse_args()
    return Archive(args.job, args.registration).run()


if __name__ == '__main__':
    raise SystemExit(main())
