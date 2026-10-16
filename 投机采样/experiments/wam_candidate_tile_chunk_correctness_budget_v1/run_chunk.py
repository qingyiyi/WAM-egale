"""One budget-only continuation of exact S04 v2 mathematics; old failures retained."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'wam_shape_audit_v1'))
from run_graph import bounded, NVML, ownership, pin, lifetime, ROOT, OWN as SHAPE, PRIOR
OWN = Path(__file__).resolve().parent
RUNTIME = ROOT / '投机采样/results/runtime'
PILOT_SOURCE = OWN.parent / 'wam_candidate_tile_graph_pilot_v1/run_pilot.py'
FIXTURE = RUNTIME / 'p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4'
INPUT_PROOF = RUNTIME / 'p0-wam-s04-input-feasibility-v1-20261016-r1/ACTUAL-CPU-INPUTS.json'
FAILED_V1 = RUNTIME / 'p0-wam-candidate-tile-chunk-correctness-v1-20261016-r1'
FAILED_V2 = RUNTIME / 'p0-wam-candidate-tile-chunk-correctness-v2-20261016-r1'
V2_SOURCE = OWN.with_name('wam_candidate_tile_chunk_correctness_v2')

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(v, msg):
    if not v: raise RuntimeError(msg)
def write(p, v):
    with Path(p).open('x') as f:
        json.dump(v, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')

def cumulative_bytes(job):
    return sum(x.stat().st_size for folder in (FAILED_V1, FAILED_V2, Path(job))
        for x in folder.rglob('*') if x.is_file())

def prove_exact_v2_numerical_source():
    for name in ('observer.py', 'private_adapter.py'):
        require((OWN/name).read_bytes()==(V2_SOURCE/name).read_bytes(),
            'numerical observer/adapter must be byte-identical to frozen v2: '+name)
    source=(V2_SOURCE/'chunk_worker.py').read_text()
    edits=[('Runtime(server, out, 580','Runtime(server, out, 313'),
        ('<= 580','<= 313'),('780s work cap','1140s work cap'),
        ('116 * 2**20','75 * 2**20'),('116MiB output cap','75MiB output cap'),
        ("old_bytes = sum(p.stat().st_size for p in Path(plan['original_failed_v1']).rglob('*') if p.is_file())",
         "old_bytes = sum(p.stat().st_size for folder in plan['previous_failed_jobs'] for p in Path(folder).rglob('*') if p.is_file())")]
    for old,new in edits:
        require(old in source, 'exact v2 resource edit site missing: '+old)
        source=source.replace(old,new)
    require(source==(OWN/'chunk_worker.py').read_text(),
        'worker must differ only at explicitly enumerated resource/source-binding sites')

def original_failures():
    r=read(FAILED_V1/'CHUNK-RESULT.json');s=read(FAILED_V1/'SUPERVISOR-RESULT.json')
    c=read(FAILED_V1/'COUNTER-CLOSED.json');p=read(FAILED_V1/'PLAN.json')
    require(r['status']=='FAILED' and r['error']=="RuntimeError('read-only target changed any valid KV or metadata')"
        and r['complete_policy_cache_chunks']==0 and r['total_target_forwards']==19
        and not r['actual_guarded_path_used'] and not r['arms'] and not r['pairs'],
        'only the real original ordinary scratch-observer failure permits v2')
    require(s['status']=='FAILED_OR_STOPPED' and s['failure'] is not None
        and s['all_owned_live_exited'] and s['own_process_reaped']
        and c['target_intents']==c['host_returned']==19
        and c['failed_forwards']==c['active_forwards']==0, 'original failure cleanup/cumulative budget required')
    for n in ('PLAN.txt','private_adapter.py','observer.py','chunk_worker.py','run_chunk.py'):
        path=OWN.with_name('wam_candidate_tile_chunk_correctness_v1')/n
        require(sha(path)==p['sources'][str(path)], 'original v1 frozen source changed')
    r2=read(FAILED_V2/'CHUNK-RESULT.json');s2=read(FAILED_V2/'SUPERVISOR-RESULT.json')
    c2=read(FAILED_V2/'COUNTER-CLOSED.json');p2=read(FAILED_V2/'PLAN.json')
    require(r2['status']=='FAILED' and r2['error']=="TimeoutError('780s work cap, sixty-second cleanup reserve')"
        and r2['complete_policy_cache_chunks']==5 and len(r2['arms'])==5 and len(r2['pairs'])==2
        and r2['total_target_forwards']==268 and r2['first_difference'] is None,
        'only real v2 pure deadline failure permits this one budget continuation')
    require({(x['length'],x['arm']) for x in r2['arms']}==
        {('short','native'),('short','guarded'),('middle','native'),('middle','guarded'),('long','native')}
        and {x['length'] for x in r2['pairs']}=={'short','middle'}
        and all(x['complete_policy_cache_exact'] and x['actual_guarded_every_B14'] and x['actual_commits']==50 for x in r2['pairs']),
        'v2 five complete arms and two exact pairs must remain unchanged')
    require(s2['status']=='FAILED_OR_STOPPED' and s2['failure'] is not None
        and '780s work cap' in s2['failure']['error'] and s2['all_owned_live_exited'] and s2['own_process_reaped']
        and c2['target_intents']==c2['host_returned']==268 and c2['active_forwards']==c2['failed_forwards']==0,
        'v2 true closed counts, deadline and cleanup required')
    for name in ('PLAN.txt','private_adapter.py','observer.py','chunk_worker.py','run_chunk.py'):
        path=V2_SOURCE/name
        require(sha(path)==p2['sources'][str(path)], 'original v2 frozen source changed')
    prove_exact_v2_numerical_source()
    prior_seconds=s['GPU_wall_sec']+s2['GPU_wall_sec']
    old_bytes=sum(x.stat().st_size for folder in (FAILED_V1,FAILED_V2) for x in folder.rglob('*') if x.is_file())
    require(19+268+313<=600 and old_bytes+75*2**20<=128*2**20 and prior_seconds+1200<=3*3600,
        'cumulative target/output/original user S04 three-GPU-hour cap')
    names=('PLAN.json','CHUNK-RESULT.json','SUPERVISOR-RESULT.json','COUNTER-CLOSED.json','FAILED.json')
    return {str(folder/n):sha(folder/n) for folder in (FAILED_V1,FAILED_V2) for n in names},prior_seconds

def pilot_module():
    spec=importlib.util.spec_from_file_location('s04_actual_pilot_gate',PILOT_SOURCE)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def requirements():
    receipts,_=original_failures()
    pilot=pilot_module();receipts.update(pilot.require_actual_guarded_gate())
    for suffix in ('r1','r2'):
        job=RUNTIME / ('p0-wam-candidate-tile-graph-pilot-v1-20261016-'+suffix)
        require(not list(job.rglob('*FAILED*.json')) and not (job/'STOP_REQUESTED').exists(), 'pilot failure/stop')
        pilot.load(job)
        r=read(job/'PILOT-RESULT.json');s=read(job/'SUPERVISOR-RESULT.json')
        require(r['status']=='COMPLETE_THREE_ARM_STATIC_GRAPH_PILOT_PENDING_CLEANUP'
                and r['all_18_actual_eager_admissions_passed'] and len(r['admission'])==18
                and r['all_six_queries_all_three_arms_complete']
                and r['all_valid_KV_RNG_inputs_slots_versions_unchanged']
                and len(r['rows'])==6 and r['target_forward_intents']==198,
                'two actual full Graph recipe copies required')
        require(all(x['output']['bitwise'] and x['output']['finite'] for x in r['admission']), 'all actual eager admissions')
        require(all(all(len(row['timing'][arm]['samples'])==6 and all(x['frozen_output_bitwise_and_finite']
                    for x in row['timing'][arm]['samples']) for arm in pilot.ARMS) for row in r['rows']), 'all actual Graph replays exact')
        require(s['status']=='COMPLETE_THREE_ARM_STATIC_GRAPH_PILOT' and s['failure'] is None
                and s['own_process_reaped'] and s['all_owned_live_exited'], 'actual replica cleanup required')
        for name in ('PLAN.json','PILOT-RESULT.json','SUPERVISOR-RESULT.json'):
            receipts[str(job/name)]=sha(job/name)
    return receipts

def prepare(job):
    job=Path(job).resolve();require(job.parent==RUNTIME and not job.exists(),'fresh isolated S04 job')
    receipts=requirements();prior=read(PRIOR);pilot_plan=read(RUNTIME/'p0-wam-candidate-tile-graph-pilot-v1-20261016-r2/PLAN.json')
    sources=dict(pilot_plan['sources'])
    for p in [*OWN.glob('*.py'),OWN/'PLAN.txt']:
        sources[str(p.resolve())]=sha(p)
    for p in (PRIOR,INPUT_PROOF,FIXTURE/'MIDDLE-PREFIX.json'):
        receipts[str(p)]=sha(p)
    proof=read(INPUT_PROOF)
    require(proof['status']=='ACTUAL_CPU_COMPLETE_INPUT_METADATA_AND_POST_POLICY_OBSERVATIONS_PRESENT','actual input proof')
    stats={}
    for item in [prior['short'],prior['long'],*prior['middle_prefix_artifacts'],*proof['post_policy_inputs']]:
        p=Path(item['path']);st=p.stat();stats[str(p)]=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns)
        if 'sha256' in item:require(sha(p)==item['sha256'],'real small input changed')
    p=dict(schema='wam-candidate-tile-full-policy-cache-correctness-budget/v1',job=str(job),
        sources=sources,prerequisites=receipts,input_stats=stats,weight_stats=prior['target_weight_inventory'],
        original_source_plan=str(PRIOR),original_source_plan_sha256=sha(PRIOR),
        qualified_fixture_root=str(FIXTURE),post_policy_observations=proof['post_policy_inputs'],
        native_binary=pilot_plan['native_binary'],candidate_binary=pilot_plan['candidate_binary'],
        original_failed_v1=str(FAILED_V1),previous_failed_jobs=[str(FAILED_V1),str(FAILED_V2)],
        prior_GPU_seconds=original_failures()[1],prior_target_forwards=287,
        exact_v2_numerical_sources={n:sha(V2_SOURCE/n) for n in ('observer.py','private_adapter.py','chunk_worker.py')},
        cumulative_GPU_seconds_cap=10800,cumulative_target_forward_cap=600,cumulative_output_cap_bytes=128*2**20,
        GPU_wall_cap_seconds=1200,active_seconds=1140,cleanup_seconds=60,CPU_worker=7,
        output_cap_bytes=75*2**20,disk_floor_bytes=5*2**30,initial_free_bytes=22*2**30,runtime_free_bytes=2**29,
        expected_policy_cache_chunks=6,middle_preparation_forwards=18,target_forward_cap=313,episodes=0,
        arms=['native','guarded'],lengths=['short','middle','long'],
        selector='prefix_then_cost',threshold=prior['frozen_reference']['threshold'],mode='verifier_k2_b2_bonus',
        selection_rule='G2 iff ceil((P+32)/64)==ceil((P+16)/64), else G1',
        baseline_all_query_events_saved=True,actual_guarded_output_returned=True,
        trace_teacher_off=True,all50_actual_commits_required=True,all_valid_KV_state_required=True,
        shared_GPU_numerical_only=True,performance_claim=False,production_integration_permission=False,
        new_environment_closed_loop=False,TF_quality_certificate=False,no_retry=True,master_port=29625)
    job.mkdir();write(job/'PLAN.json',p);write(job/'SOURCE-PLAN.json',prior);return p

def load(job):
    job=Path(job).resolve();p=read(job/'PLAN.json')
    require(p['schema']=='wam-candidate-tile-full-policy-cache-correctness-budget/v1' and p['job']==str(job),'S04 job identity')
    require((p['GPU_wall_cap_seconds'],p['active_seconds'],p['cleanup_seconds'],p['CPU_worker'],
        p['output_cap_bytes'],p['target_forward_cap'],p['expected_policy_cache_chunks'],
        p['middle_preparation_forwards'],p['episodes'])==(1200,1140,60,7,75*2**20,313,6,18,0),'fixed S04 bounds')
    prove_exact_v2_numerical_source()
    require(p['original_failed_v1']==str(FAILED_V1) and p['previous_failed_jobs']==[str(FAILED_V1),str(FAILED_V2)]
        and p['prior_GPU_seconds']==original_failures()[1]
        and p['exact_v2_numerical_sources']=={n:sha(V2_SOURCE/n) for n in ('observer.py','private_adapter.py','chunk_worker.py')}
        and p['prior_target_forwards']==287 and p['cumulative_GPU_seconds_cap']==10800
        and p['cumulative_target_forward_cap']==600 and p['cumulative_output_cap_bytes']==128*2**20,
        'original FAILED remains explicit and cumulative budget immutable')
    require(p['arms']==['native','guarded'] and p['lengths']==['short','middle','long']
        and p['mode']=='verifier_k2_b2_bonus' and p['selector']=='prefix_then_cost'
        and p['selection_rule']=='G2 iff ceil((P+32)/64)==ceil((P+16)/64), else G1'
        and p['baseline_all_query_events_saved'] and p['actual_guarded_output_returned']
        and p['all50_actual_commits_required'] and p['all_valid_KV_state_required'] and p['trace_teacher_off']
        and not p['performance_claim'] and not p['production_integration_permission']
        and not p['new_environment_closed_loop'] and not p['TF_quality_certificate'],'fixed numerical scope')
    receipts=requirements();require(all(p['prerequisites'].get(k)==v for k,v in receipts.items()),'actual prerequisites binding')
    for path,h in {**p['sources'],**p['prerequisites']}.items():require(sha(path)==h,'frozen input/source changed: '+path)
    for path,x in p['input_stats'].items():
        st=Path(path).stat();require((st.st_size,st.st_mtime_ns)==(x['bytes'],x['mtime_ns']),'real replay source stat changed')
    for x in p['weight_stats']:
        st=Path(x['path']).stat();require((st.st_size,st.st_mtime_ns)==(x['bytes'],x['mtime_ns']),'model weight stat changed')
    require(read(job/'SOURCE-PLAN.json')==read(PRIOR) and sha(PRIOR)==p['original_source_plan_sha256'],'original source plan')
    return p

def execute(job, plan):
    os.sched_setaffinity(0, set(sorted(os.sched_getaffinity(0))[:2]))
    require(not any((job / n).exists() for n in
        ('RUN-STARTED.json', 'SUPERVISOR-RESULT.json', 'CHUNK-RESULT.json', 'FAILED.json', 'STOP_REQUESTED')),
        'one reviewed attempt only')
    require(not list(job.rglob('*FAILED*.json')), 'failure markers block execution')
    locks = []
    child = None
    pins = {}
    samples = []
    cleanups = []
    error = None
    nv = None
    started = None

    def refresh():
        for path in Path('/proc').iterdir():
            if not path.name.isdigit(): continue
            try:
                info = ownership.read_process(int(path.name))
                if ownership.is_owned(info, pins): pins[info.pid] = info
            except (FileNotFoundError, ProcessLookupError, PermissionError, RuntimeError): continue

    def live():
        result = []
        for pid, known in list(pins.items()):
            try:
                info = ownership.read_process(pid)
                raw = Path('/proc', str(pid), 'stat').read_text()
                if lifetime(info) == lifetime(known) and raw[raw.rfind(')') + 2:].split()[0] != 'Z':
                    result.append(pid)
            except (FileNotFoundError, ProcessLookupError): pass
            except (RuntimeError, PermissionError): result.append(pid)
        return result

    def stop(sig):
        refresh()
        for pid in live():
            try:
                info = ownership.read_process(pid)
                require(lifetime(info) == lifetime(pins[pid]), 'refuse reused PID')
                os.kill(pid, sig)
                cleanups.append(dict(pid=pid, signal=int(sig)))
            except (FileNotFoundError, ProcessLookupError): pass

    def stop_signal(*_): raise SystemExit('signal requested owned cleanup')

    signal.signal(signal.SIGTERM, stop_signal)
    try:
        for name in ('.speculative_minimal_gpu.lock', '.shared-prefix-GPU.lock'):
            handle = (RUNTIME / name).open('a+')
            try: fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException: handle.close(); raise RuntimeError('project GPU lock busy: ' + name)
            locks.append(handle)
        nv = bounded(NVML)
        sample = bounded(nv.sample)
        samples.append(sample)
        require(samples[-1]['memory_free_bytes'] >= plan['initial_free_bytes'], '22GiB headroom required')
        require(shutil.disk_usage(job).free >= plan['disk_floor_bytes'], '5GiB disk floor required')
        require(cumulative_bytes(job)<=plan['cumulative_output_cap_bytes'], 'initial cumulative output cap')
        with socket.socket() as sock: sock.bind(('127.0.0.1', plan['master_port']))
        prior = read(job / 'SOURCE-PLAN.json')
        ref = prior['frozen_reference']
        env = {k: v for k, v in os.environ.items() if not k.startswith(('WAM_P0_', 'P0_', 'LINGBOT_'))}
        env.update(CUDA_VISIBLE_DEVICES='0', WAM_SHAPE_OUT=str(job),
            OMP_NUM_THREADS='7', MKL_NUM_THREADS='7', OPENBLAS_NUM_THREADS='7', NUMEXPR_NUM_THREADS='7',
            PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            PYTHONHASHSEED='0', LINGBOT_SEED='20260922', LINGBOT_DISABLE_FSDP='1',
            LINGBOT_CACHE_MODE='original', LINGBOT_ATTN_MODE='torch', LINGBOT_NSYS_PROFILE='off',
            LINGBOT_VA_LIBERO_SUITE='libero_10', LINGBOT_VA_LIBERO_CHECKPOINT=ref['target_checkpoint_path'],
            LINGBOT_VA_LIBERO_CHECKPOINT_ID=ref['target_checkpoint_id'],
            LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=ref['registry'], WAM_VAE_DEVICE='cuda',
            WAM_P0_PHASE_PROFILE='0', WAM_P0_ACTION_DRAFT_MODE='off',
            WAM_P0_ACTION_DRAFT_SELECTOR='prefix_then_cost',
            WAM_P0_ACTION_DRAFT_THRESHOLD=str(ref['threshold']),
            WAM_P0_ACTION_DRAFT_B2_CHECKPOINT=prior['frozen_A_off'],
            PREFIX_ACCEPT_ENABLED='1', PREFIX_ACCEPT_GPU_APPROVED='1',
            MASTER_ADDR='127.0.0.1', MASTER_PORT=str(plan['master_port']), RANK='0', LOCAL_RANK='0', WORLD_SIZE='1',
            PYTHONPATH=':'.join(map(str, [OWN, SHAPE, OWN.parent / 'prefix_acceptance_v1',
                                      ROOT / '投机采样/wam-p0-worktree', ROOT / '投机采样/scripts'])))
        cmd = ['/rjs/cxz/miniconda3/envs/va/bin/python', '-B', str(OWN / 'chunk_worker.py'),
               '--config-name', 'libero', '--port', '19390', '--save_root', str(job / 'server')]
        with (job / 'worker.log').open('x') as log:
            started = time.monotonic()
            write(job / 'RUN-STARTED.json', dict(started_monotonic=started,
                active_deadline_monotonic=started + 1140, final_deadline_monotonic=started + 1200,
                no_automatic_retry=True, command=cmd, CUDA_visible_devices='0'))
            child = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, cwd=ROOT / '投机采样/wam-p0-worktree')
            identity = pin(child, cmd, on_boot=lambda info: pins.update({info.pid: info}))
            pins[identity.pid] = identity
            write(job / 'OWNED-PROCESS.json', vars(identity))
            while child.poll() is None:
                refresh()
                sample = bounded(nv.sample)
                samples.append(sample)
                # Shared-GPU numerical experiment; never signals foreign processes.
                require(time.monotonic() - started < 1140, '1140s work cap, sixty seconds cleanup reserve')
                require(sample['memory_free_bytes'] >= plan['runtime_free_bytes'], 'runtime half-GiB headroom')
                require(shutil.disk_usage(job).free >= plan['disk_floor_bytes'], 'disk floor reached')
                require(sum(x.stat().st_size for x in job.rglob('*') if x.is_file()) <= plan['output_cap_bytes'],
                        '75MiB output cap')
                require(cumulative_bytes(job)<=plan['cumulative_output_cap_bytes'], 'cumulative v1+v2 128MiB output cap')
                require(not (job / 'STOP_REQUESTED').exists(), 'STOP_REQUESTED')
                time.sleep(.5)
            require(child.returncode == 0, 'worker failed: ' + str(child.returncode))
        result = read(job / 'CHUNK-RESULT.json')
        require(result['status'] == 'COMPLETE_THREE_LENGTH_FULL_POLICY_CACHE_EXACT_PENDING_CLEANUP'
                and result['complete_policy_cache_chunks'] == 6 and len(result['pairs']) == 3
                and len(result['arms']) == 6 and result['actual_guarded_path_used']
                and result['middle_preparation_target_forwards'] == 18
                and result['complete_50_update_sequences'] == 6 and result['total_target_forwards'] <= 313
                and result['all_target_inputs_outputs_exact']
                and result['all_costs_routes_bonus_actual_commits_exact']
                and result['all_valid_KV_RNG_scheduler_VAE_server_state_exact']
                and result['original_history_not_polluted'],
                'six complete actual policy+own-action cache paths and three exact pairs required')
        require(all(x['complete_policy_cache_exact'] and x['actual_guarded_every_B14']
                    and x['actual_commits'] == 50 for x in result['pairs']), 'no probe-only path accepted')
        require(not result['performance_claim'] and not result['production_integration_permission']
                and not result['new_environment_closed_loop'] and not result['TF_quality_certificate'],
                'bounded A_off numerical replay only')
        require(not list(job.rglob('*FAILED*.json')), 'failure markers take precedence')
        require(sum(x.stat().st_size for x in job.rglob('*') if x.is_file()) <= plan['output_cap_bytes'],
                'final output cap')
        require(cumulative_bytes(job)<=plan['cumulative_output_cap_bytes'], 'final cumulative output cap')
    except BaseException as exc:
        error = dict(error=repr(exc), traceback=traceback.format_exc())
    finally:
        try:
            if child is not None and child.poll() is None:
                if not (job / 'STOP_REQUESTED').exists():
                    (job / 'STOP_REQUESTED').write_text('supervisor owned cleanup\n')
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    stop(signal.SIGTERM)
                    try: child.wait(timeout=10)
                    except subprocess.TimeoutExpired: stop(signal.SIGKILL); child.wait(timeout=5)
            refresh()
            if live():
                stop(signal.SIGTERM)
                limit = time.monotonic() + 5
                while live() and time.monotonic() < limit: time.sleep(.1)
                if live(): stop(signal.SIGKILL)
                limit = time.monotonic() + 5
                while live() and time.monotonic() < limit: time.sleep(.1)
            require(not live(), 'owned process remains live: ' + repr(live()))
        except BaseException as exc:
            error = error or dict(error='cleanup: ' + repr(exc))
        final = None
        if nv is not None:
            try:
                final = bounded(nv.sample)
                bounded(nv.close)
                require(not any(x['pid'] in pins for k in ('compute', 'graphics') for x in final[k]),
                        'own GPU process remains in NVML')
            except BaseException as exc: error = error or dict(error='final NVML: ' + repr(exc))
        for handle in reversed(locks):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()
        if list(job.rglob('*FAILED*.json')): error = error or dict(error='failure marker has priority')
        if (job / 'STOP_REQUESTED').exists(): error = error or dict(error='STOP_REQUESTED has priority')
        terminal = dict(status='COMPLETE_THREE_LENGTH_FULL_POLICY_CACHE_EXACT' if error is None else 'FAILED_OR_STOPPED',
            integration_permission=False, performance_claim=False,
            failure=error, own_process_reaped=child is None or child.poll() is not None,
            all_owned_live_exited=not live(), owned_tree=[vars(v) for v in pins.values()],
            cleanup_events=cleanups, final_resources=final, samples=samples,
            GPU_wall_sec=0 if started is None else time.monotonic() - started,
            prior_GPU_wall_sec=plan['prior_GPU_seconds'],
            cumulative_GPU_wall_sec=plan['prior_GPU_seconds']+(0 if started is None else time.monotonic()-started),
            formal_robot_performance=False)
        if error is not None and not (job / 'FAILED.json').exists(): write(job / 'FAILED.json', error)
        write(job / 'SUPERVISOR-RESULT.json', terminal)
    if error is not None: raise RuntimeError(error['error'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare',action='store_true');mode.add_argument('--inspect',action='store_true')
    mode.add_argument('--execute-reviewed',action='store_true');args=parser.parse_args()
    if args.prepare:
        p=prepare(args.job);print(json.dumps(dict(status='PREPARED_CPU_ONLY',job=p['job'],GPU_used=False)))
    else:
        p=load(args.job)
        if args.inspect: print(json.dumps(dict(status='PASS_COMPLETE_S04_PLAN_CPU_ONLY',job=p['job'],GPU_used=False)))
        else: execute(args.job.resolve(),p)
if __name__=='__main__':main()
