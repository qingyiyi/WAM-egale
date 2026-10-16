"""Bounded shared-GPU fixture capture; CPU preparation never opens NVML/CUDA."""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time

OWN = Path(__file__).resolve().parent; ROOT = OWN.parents[2]
RUNTIME = ROOT / '投机采样/results/runtime'
PRIOR = Path('/var/tmp/cxz-wam-quality-20261014/p0-megaverifier-profile-v2-20261014-marker-fix/plan.json')
sys.path.insert(0, str(OWN.parent / 'single_worker_chc_gpu_v4'))
sys.path.append(str(OWN.parent / 'speculative_e2e_v3'))
import ownership
from launch_identity import pin


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False); stream.write('\n')


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(value, message):
    if not value: raise ValueError(message)


def inspect_inputs():
    prior = read(PRIOR); files = {}; exceptions = []
    for path, frozen in prior['source_sha256'].items():
        actual = sha(path)
        if path in (str(ROOT / '.pipeline/config.yaml'), str(ROOT / 'AGENTS.md')):
            exceptions.append(dict(path=path, original_sha256=frozen, actual_sha256=actual,
                reason='root-authorized administrative changes; not math/runtime dependencies'))
        else: require(actual == frozen, 'frozen prior source changed: ' + path)
        files[path] = actual
    for name in ('capture_only.py', 'run_capture.py', 'full_matrix.py'):
        path = OWN / name; ast.parse(path.read_text()); files[str(path)] = sha(path)
    files[str(PRIOR)] = sha(PRIOR)
    weights = prior['target_weight_inventory']
    require(isinstance(weights, list) and len(weights) == 9, 'actual nine-entry prior weight stat inventory required')
    for row in weights:
        st = Path(row['path']).stat()
        require((st.st_size, st.st_mtime_ns) == (row['bytes'], row['mtime_ns']), 'target weight stat changed')
    artifacts = []
    for row in [prior['short'], prior['long']] + prior['middle_prefix_artifacts']:
        path = Path(row['path']); st = path.stat()
        require(st.st_size == row['bytes'], 'real frozen input size changed: ' + str(path))
        if 'mtime_ns' in row: require(st.st_mtime_ns == row['mtime_ns'], 'real frozen input mtime changed')
        artifacts.append(dict(path=str(path), bytes=st.st_size, mtime_ns=st.st_mtime_ns))
    draft = Path(prior['frozen_A_off'])
    require(sha(draft) == prior['frozen_reference']['draft_checkpoint_sha256'], 'A_off checkpoint differs')
    files[str(draft)] = sha(draft)
    return prior, dict(source_sha256=files, administrative_exceptions=exceptions,
        input_artifact_stat=artifacts, target_weight_inventory=weights, whole_target_model_hash=False)


def prepare(job):
    require(os.getenv('CUDA_VISIBLE_DEVICES') == '', 'CPU preparation requires explicit empty CUDA mask')
    job = Path(job).resolve()
    require(job.is_relative_to(RUNTIME) and not job.exists(), 'fresh isolated runtime directory required')
    prior, proof = inspect_inputs()
    p = dict(schema='wam-three-real-chunk-fixture-capture/v1', job=str(job),
        GPU_seconds_including_cleanup=1200, active_seconds=1140, cleanup_seconds=60,
        CPU_worker=7, target_forward_cap=1800, complete_chunks=3, episodes=0,
        initial_free_GiB=22., runtime_free_GiB=.5, disk_floor_GiB=5., output_cap_bytes=3*2**30,
        shared_GPU_allowed=True, formal_performance=False, profiler=False, timing_samples=0,
        new_engine=False, new_TF_entry=False, automatic_retry=False,
        lengths=[dict(name=n, request_history=h, verifier_total_KV=k) for n,h,k in
                 (('short',144,288),('middle',1008,1152),('long',2016,2160))],
        diagnostic_shapes=[1,3,7], one_forward_per_query_per_shape=True,
        requires_RAW_B14_matches_captured_O3_bitwise=True,
        all_valid_KV_RNG_parameter_unchanged_required=True,
        source_plan=str(PRIOR), prior_sha256=sha(PRIOR), source_provenance=proof,
        master_port=19393, policy_port=19392,
        lock_files=[str(RUNTIME/n) for n in ('.speculative_minimal_gpu.lock','.shared-prefix-GPU.lock')],
        old_failed_results_modified=False,
        consumer_rule='Require new RESULT+CAPTURE-RESULT+COUNTER-CLOSED+cleaned supervisor; old MATRIX gate is not forged.')
    job.mkdir(parents=True)
    write(job/'PLAN.json', p); write(job/'SOURCE-PLAN.json', prior); write(job/'SOURCE-PROVENANCE.json', proof)
    return p


def load_plan(job):
    job = Path(job).resolve(); p = read(job/'PLAN.json')
    require(p['schema']=='wam-three-real-chunk-fixture-capture/v1' and p['job']==str(job), 'capture plan identity')
    require((p['GPU_seconds_including_cleanup'],p['active_seconds'],p['cleanup_seconds'],p['CPU_worker'],
             p['target_forward_cap'],p['complete_chunks'],p['episodes'],p['output_cap_bytes'])
            ==(1200,1140,60,7,1800,3,0,3*2**30), 'capture fixed bounds changed')
    require(p['initial_free_GiB']==22. and p['runtime_free_GiB']==.5 and p['disk_floor_GiB']==5., 'capture resource floors changed')
    require(p['shared_GPU_allowed'] and not p['formal_performance'] and not p['profiler']
            and p['timing_samples']==0 and not p['automatic_retry'], 'correctness-only contract changed')
    require(p['lengths']==[dict(name=n,request_history=h,verifier_total_KV=k) for n,h,k in
            (('short',144,288),('middle',1008,1152),('long',2016,2160))]
            and p['diagnostic_shapes']==[1,3,7] and p['one_forward_per_query_per_shape']
            and p['requires_RAW_B14_matches_captured_O3_bitwise']
            and p['all_valid_KV_RNG_parameter_unchanged_required']
            and not p['new_engine'] and not p['new_TF_entry']
            and (p['master_port'],p['policy_port'])==(19393,19392)
            and p['lock_files']==[str(RUNTIME/n) for n in ('.speculative_minimal_gpu.lock','.shared-prefix-GPU.lock')],
            'fixed geometry/locks/oracle contract changed')
    require(p['prior_sha256']==sha(PRIOR) and read(job/'SOURCE-PLAN.json')==read(PRIOR), 'source plan changed')
    proof=p['source_provenance']
    require(read(job/'SOURCE-PROVENANCE.json')==proof, 'source proof differs')
    prior=read(PRIOR)
    expected=set(prior['source_sha256'])|{str(OWN/n) for n in ('capture_only.py','run_capture.py','full_matrix.py')}|{str(PRIOR),prior['frozen_A_off']}
    require(set(proof['source_sha256'])==expected
            and proof['target_weight_inventory']==prior['target_weight_inventory']
            and {x['path'] for x in proof['input_artifact_stat']}==
                {x['path'] for x in [prior['short'],prior['long']]+prior['middle_prefix_artifacts']},
            'complete actual source/input inventory required')
    for path, expected in proof['source_sha256'].items(): require(sha(path)==expected, 'capture frozen source changed: '+path)
    for row in proof['target_weight_inventory']+proof['input_artifact_stat']:
        st=Path(row['path']).stat()
        require((st.st_size,st.st_mtime_ns)==(row['bytes'],row['mtime_ns']), 'capture input/weight stat changed')
    return p


def bounded(fn, seconds=5):
    q=queue.Queue()
    def work():
        try:q.put((True,fn()))
        except BaseException as exc:q.put((False,exc))
    threading.Thread(target=work,daemon=True).start()
    try:ok,value=q.get(timeout=seconds)
    except queue.Empty:raise TimeoutError('bounded NVML metadata query timed out')
    if not ok:raise value
    return value


def lifetime(info): return (info.pid,info.pgid,info.start_ticks,info.uid)


def execute(out,p):
    from resource_guard import NVML
    nv=None;child=None;reason=None;identity=None;gpu_origin=None;pins={};cleanup_events=[];samples=[]
    start=time.monotonic()
    def remember(info):pins[info.pid]=info
    def refresh():
        for d in Path('/proc').iterdir():
            if not d.name.isdigit():continue
            try:
                info=ownership.read_process(int(d.name))
                if ownership.is_owned(info,pins):pins[info.pid]=info
            except (FileNotFoundError,PermissionError,ProcessLookupError,RuntimeError):continue
    def live():
        found=[]
        for pid,known in pins.items():
            try:
                info=ownership.read_process(pid)
                if lifetime(info)==lifetime(known):
                    raw=Path('/proc',str(pid),'stat').read_text()
                    if raw[raw.rfind(')')+2:].split()[0]!='Z':found.append(pid)
            except (FileNotFoundError,ProcessLookupError):pass
            except RuntimeError:found.append(pid)
        return found
    def signal_owned(sig):
        refresh()
        for pid in live():
            try:
                now=ownership.read_process(pid)
                if lifetime(now)!=lifetime(pins[pid]):raise RuntimeError('refuse changed PID lifetime')
                os.kill(pid,sig);cleanup_events.append(dict(pid=pid,signal=int(sig)))
            except (FileNotFoundError,ProcessLookupError):pass
    def sample():
        row=bounded(nv.sample);row['wall_epoch']=time.time();provenance=[]
        for app in [x for kind in ('compute','graphics') for x in row[kind]]:
            try:
                info=ownership.read_process(app['pid']);own=ownership.is_owned(info,pins)
                if own:pins[info.pid]=info
                provenance.append(dict(pid=info.pid,ppid=info.ppid,uid=info.uid,
                    user=pwd.getpwuid(info.uid).pw_name,start_ticks=info.start_ticks,owned=own,
                    executable=info.argv[0] if info.argv else None))
            except (FileNotFoundError,ProcessLookupError):
                provenance.append(dict(pid=app['pid'],exited_before_inspection=True))
        row['process_provenance']=provenance;samples.append(row);return row
    def output_guard():
        if sum(x.stat().st_size for x in out.rglob('*') if x.is_file())>p['output_cap_bytes']:
            raise RuntimeError('capture output exceeds3GiB')
        if shutil.disk_usage(out).free<p['disk_floor_GiB']*2**30:
            raise RuntimeError('capture disk floor below5GiB')
    try:
        output_guard();nv=bounded(NVML);initial=sample()
        require(initial['memory_free_bytes']>=p['initial_free_GiB']*2**30, 'shared GPU has less than22GiB free')
        prior=read(out/'SOURCE-PLAN.json');ref=prior['frozen_reference']
        env={k:v for k,v in os.environ.items() if not k.startswith(('WAM_P0_','P0_','LINGBOT_'))}
        env.update(CUDA_VISIBLE_DEVICES='0',WAM_SHAPE_OUT=str(out),WAM_CAPTURE_EXECUTE_REVIEWED='1',
            OMP_NUM_THREADS='7',MKL_NUM_THREADS='7',OPENBLAS_NUM_THREADS='7',NUMEXPR_NUM_THREADS='7',
            PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED='0',
            LINGBOT_SEED='20260922',LINGBOT_DISABLE_FSDP='1',LINGBOT_CACHE_MODE='original',
            LINGBOT_ATTN_MODE='torch',LINGBOT_NSYS_PROFILE='off',LINGBOT_VA_LIBERO_SUITE='libero_10',
            LINGBOT_VA_LIBERO_CHECKPOINT=ref['target_checkpoint_path'],LINGBOT_VA_LIBERO_CHECKPOINT_ID=ref['target_checkpoint_id'],
            LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=ref['registry'],WAM_VAE_DEVICE='cuda',WAM_P0_PHASE_PROFILE='0',
            WAM_P0_ACTION_DRAFT_MODE='off',WAM_P0_ACTION_DRAFT_SELECTOR='prefix_then_cost',
            WAM_P0_ACTION_DRAFT_THRESHOLD=str(ref['threshold']),WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=prior['frozen_A_off'],
            PREFIX_ACCEPT_ENABLED='1',PREFIX_ACCEPT_GPU_APPROVED='1',MASTER_ADDR='127.0.0.1',
            MASTER_PORT=str(p['master_port']),RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',
            PYTHONPATH=':'.join(map(str,[OWN,OWN.parent/'prefix_acceptance_v1',ROOT/'投机采样/wam-p0-worktree',ROOT/'投机采样/scripts'])))
        gpu_origin=time.monotonic()
        write(out/'RUN-STARTED.json',dict(schema='wam-real-fixture-capture-start/v1',
            GPU_origin_monotonic=gpu_origin,shared_GPU_allowed=True,one_attempt=True,performance_claim=False))
        cmd=['/rjs/cxz/miniconda3/envs/va/bin/python','-B',str(OWN/'capture_only.py'),
             '--config-name','libero','--port',str(p['policy_port']),'--save_root',str(out/'server')]
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,cwd=ROOT/'投机采样/wam-p0-worktree')
            identity=pin(child,cmd,on_boot=remember);pins[identity.pid]=identity
            write(out/'OWNED-PROCESS.json',vars(identity))
            while child.poll() is None:
                refresh();row=sample()
                if time.monotonic()-gpu_origin>p['active_seconds']:
                    raise TimeoutError('1200s total capture budget includes60s cleanup')
                if row['memory_free_bytes']<p['runtime_free_GiB']*2**30:
                    raise RuntimeError('shared capture free VRAM below0.5GiB')
                if (out/'STOP_REQUESTED').exists():raise RuntimeError('capture stop requested')
                output_guard();time.sleep(1)
            require(child.returncode==0,'capture worker exit '+str(child.returncode))
        worker=read(out/'CAPTURE-RESULT.json');counter=read(out/'COUNTER-CLOSED.json')
        require(worker['status']=='COMPLETE_THREE_REAL_CHUNK_FIXTURES_PENDING_OWNED_CLEANUP'
            and worker['complete_chunks']==3 and worker['episodes']==0
            and worker['all_valid_KV_RNG_parameters_unchanged']
            and worker['target_intents']==worker['host_returned']==counter['target_intents']==counter['host_returned']
            and worker['target_intents']<=1800 and counter['status']=='CLOSED'
            and counter['failed_forwards']==counter['active_forwards']==0,'incomplete capture/counter terminal')
        require(not any((out/n).exists() for n in ('WORKER-FAILED.json','FAILED.json','STOP_REQUESTED')), 'capture failure marker')
        output_guard()
    except BaseException as exc:reason=repr(exc)
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN)
        try:
            if child is not None and child.poll() is None:
                if not (out/'STOP_REQUESTED').exists():(out/'STOP_REQUESTED').write_text(reason or 'stop')
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    signal_owned(signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:signal_owned(signal.SIGKILL);child.wait(timeout=5)
            refresh()
            if live():
                signal_owned(signal.SIGTERM);limit=time.monotonic()+5
                while live() and time.monotonic()<limit:time.sleep(.2)
                if live():signal_owned(signal.SIGKILL)
                limit=time.monotonic()+5
                while live() and time.monotonic()<limit:time.sleep(.2)
            if live():raise RuntimeError('owned descendants remain: '+repr(live()))
        except BaseException as exc:reason=(reason or '')+'; cleanup: '+repr(exc)
        final=None
        if nv is not None:
            try:
                final=sample()
                recorded_owned=[]
                for app in [x for kind in ('compute','graphics') for x in final[kind]]:
                    if app['pid'] not in pins:continue
                    try:
                        now=ownership.read_process(app['pid'])
                        if lifetime(now)==lifetime(pins[app['pid']]):recorded_owned.append(app['pid'])
                    except (FileNotFoundError,ProcessLookupError):recorded_owned.append(app['pid'])
                if recorded_owned:raise RuntimeError('final NVML still records owned GPU PIDs: '+repr(recorded_owned))
            except BaseException as exc:reason=reason or repr(exc);final=dict(error=repr(exc))
            try:bounded(nv.close)
            except BaseException as exc:reason=reason or repr(exc)
        cleanup=dict(all_owned_live_exited=not live(),own_process_reaped=child is None or child.poll() is not None,
            owned_tree=[vars(x) for x in pins.values()],signals=cleanup_events,
            final_resources=final,foreign_process_signalled=False,
            final_NVML_own_GPU_empty=final is not None and 'error' not in final)
        write(out/'FINAL-OWNED-CLEANUP.json',cleanup)
        terminal=dict(status='COMPLETE_SHARED_REAL_FIXTURE_CAPTURE' if reason is None else 'FAILED_OR_BLOCKED',
            reason=reason,elapsed_resource_budget_sec=time.monotonic()-start,
            GPU_wall_sec=0 if gpu_origin is None else time.monotonic()-gpu_origin,
            shared_GPU_allowed=True,formal_performance=False,profiler_used=False,
            timing_samples=0,all_owned_live_exited=cleanup['all_owned_live_exited'],
            own_process_reaped=cleanup['own_process_reaped'],samples=samples)
        write(out/'SUPERVISOR-RESULT.json',terminal)
        if reason is None:
            write(out/'RESULT.json',dict(status='PASS_THREE_REAL_CHUNK_FIXTURES',
                complete_chunks=3,episodes=0,lengths=[288,1152,2160],
                all_owned_live_exited=True,own_process_reaped=True,
                source_provenance='SOURCE-PROVENANCE.json',worker='CAPTURE-RESULT.json',
                counter='COUNTER-CLOSED.json',cleanup='FINAL-OWNED-CLEANUP.json',
                shared_GPU_allowed=True,formal_performance=False,profiler_used=False,
                new_TF_entry=False,old_failed_results_modified=False))
        else:write(out/'FAILED.json',dict(status='FAILED_REAL_FIXTURE_CAPTURE',reason=reason,qualified_fixtures=False))
    if reason is not None:raise RuntimeError(reason)


def main():
    q=argparse.ArgumentParser();q.add_argument('--job',type=Path)
    g=q.add_mutually_exclusive_group(required=True)
    g.add_argument('--check-inputs',action='store_true');g.add_argument('--prepare',action='store_true')
    g.add_argument('--execute-reviewed',action='store_true');a=q.parse_args()
    os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
    if a.check_inputs:
        require(os.getenv('CUDA_VISIBLE_DEVICES')=='','CPU check requires empty CUDA mask')
        prior,proof=inspect_inputs();print(json.dumps(dict(status='PASS_CPU_CAPTURE_SOURCE_INPUTS',
            source_files=len(proof['source_sha256']),target_weight_stats=len(proof['target_weight_inventory']),
            actual_input_files=len(proof['input_artifact_stat']),GPU_started=False,NVML_called=False,
            A_off=prior['frozen_A_off'],CPU_affinity=sorted(os.sched_getaffinity(0)))));return
    require(a.job is not None,'--job required')
    if a.prepare:
        p=prepare(a.job);print(json.dumps(dict(status='PREPARED_CPU_CAPTURE_ONLY',job=p['job'],GPU_started=False)));return
    out=a.job.resolve();p=load_plan(out)
    require(not any((out/n).exists() for n in ('RUN-STARTED.json','RESULT.json','FAILED.json','CAPTURE-RESULT.json')),'one attempt only')
    with socket.socket() as sock:sock.bind(('127.0.0.1',p['master_port']))
    locks=[]
    try:
        for path in p['lock_files']:
            f=Path(path).open('a+')
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:f.close();raise RuntimeError('another project GPU job owns '+path)
            locks.append(f)
        def stop(*_):raise SystemExit('signal requested owned capture cleanup')
        signal.signal(signal.SIGTERM,stop)
        execute(out,p)
    finally:
        for f in reversed(locks):fcntl.flock(f,fcntl.LOCK_UN);f.close()


if __name__=='__main__':main()
