"""Fixed, S04-dependent static prompt-cache baseline; no GPU entry here."""
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = OWN.parents[2]
RUNTIME = ROOT / '投机采样/results/runtime'
FIXTURE = RUNTIME / 'p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4'
S04_SOURCE = EXP / 'wam_candidate_tile_chunk_correctness_v2/run_chunk.py'
DEFAULT_S04 = RUNTIME / 'p0-wam-candidate-tile-chunk-correctness-v2-20261016-r1'
ARMS = ('n1_dense_uncached', 'n1_dense_cached', 'n7_native_uncached',
        'n7_native_cached', 'n7_guarded_uncached', 'n7_guarded_cached')
ORDERS = ((0, 1, 5, 2, 4, 3), (1, 2, 0, 3, 5, 4), (2, 3, 1, 4, 0, 5),
          (3, 4, 2, 5, 1, 0), (4, 5, 3, 0, 2, 1), (5, 0, 4, 1, 3, 2))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def actual_s04(job):
    job = Path(job).resolve()
    require(job.parent == RUNTIME and re.fullmatch(
        r'p0-wam-candidate-tile-chunk-correctness-v2-20261016-r[1-9][0-9]*', job.name),
        'explicit real S04 job required; no automatic latest-job selection')
    require(all((job / name).is_file() for name in
        ('PLAN.json', 'SOURCE-PLAN.json', 'CHUNK-RESULT.json', 'SUPERVISOR-RESULT.json')),
        'WAIT: complete real S04 policy/cache result and owned cleanup required')
    require(not list(job.rglob('*FAILED*.json')) and not (job / 'STOP_REQUESTED').exists(),
        'S04 failure or STOP blocks ordinary baseline diagnostic')
    spec = importlib.util.spec_from_file_location('static_prompt_actual_s04_gate', S04_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    p = module.load(job)
    r, s = read(job / 'CHUNK-RESULT.json'), read(job / 'SUPERVISOR-RESULT.json')
    require(r['status'] == 'COMPLETE_THREE_LENGTH_FULL_POLICY_CACHE_EXACT_PENDING_CLEANUP'
        and r['complete_policy_cache_chunks'] == 6 and r['complete_50_update_sequences'] == 6
        and len(r['pairs']) == 3 and len(r['arms']) == 6
        and r['middle_preparation_target_forwards'] == 18 and r['total_target_forwards'] <= 580
        and r['actual_guarded_path_used'] and r['all_target_inputs_outputs_exact']
        and r['all_costs_routes_bonus_actual_commits_exact']
        and r['all_valid_KV_RNG_scheduler_VAE_server_state_exact']
        and r['original_history_not_polluted'], 'complete exact S04, never partial, required')
    require(not r['performance_claim'] and not r['production_integration_permission']
        and not r['new_environment_closed_loop'] and not r['TF_quality_certificate'],
        'S04 numerical scope changed')
    require(s['status'] == 'COMPLETE_THREE_LENGTH_FULL_POLICY_CACHE_EXACT'
        and s['failure'] is None and s['own_process_reaped'] and s['all_owned_live_exited'],
        'S04 all-owned terminal cleanup required')
    require({q['length'] for q in r['pairs']} == {'short', 'middle', 'long'}
        and all(q['complete_policy_cache_exact'] and q['actual_guarded_every_B14']
                and q['actual_commits'] == 50 for q in r['pairs']), 'all three exact S04 pairs required')
    receipts = {str(job / name): sha(job / name) for name in
        ('PLAN.json', 'SOURCE-PLAN.json', 'CHUNK-RESULT.json', 'SUPERVISOR-RESULT.json')}
    for label in ('short', 'middle', 'long'):
        values = {}
        for arm in ('native', 'guarded'):
            folder = job / label / arm
            values[arm] = {}
            for name in ('CHUNK.json', 'EVENTS.json', 'COMMITTED-50.json',
                         'SCHEDULER-TRANSITIONS.json', 'ACCEPTANCE-RECORDS.json',
                         'CACHE-FIELD-DIFFERENCES.json'):
                file = folder / name
                require(file.is_file(), 'complete actual S04 small evidence missing: ' + str(file))
                values[arm][name] = read(file)
                receipts[str(file)] = sha(file)
            require(len(values[arm]['COMMITTED-50.json']) == 50, 'all actual S04 commits required')
        for name in ('EVENTS.json', 'COMMITTED-50.json', 'SCHEDULER-TRANSITIONS.json',
                     'ACCEPTANCE-RECORDS.json', 'CACHE-FIELD-DIFFERENCES.json'):
            require(values['native'][name] == values['guarded'][name], 'S04 raw evidence differs: ' + name)
    return p, receipts


def prepare(job, s04_job):
    job = Path(job).resolve()
    require(job.parent == RUNTIME and not job.exists(), 'fresh isolated baseline runtime required')
    source, receipts = actual_s04(s04_job)  # No directory is written before this real gate.
    prior = read(Path(s04_job) / 'SOURCE-PLAN.json')
    paths = [*OWN.glob('*.py'), OWN / 'README.md', S04_SOURCE]
    inventory = {**source['sources'], **{str(p.resolve()): sha(p) for p in paths}}
    for path, expected in inventory.items():
        require(sha(path) == expected, 'frozen source changed: ' + path)
    producer = read(FIXTURE / 'MATRIX-RESULT.json')
    terminal = read(FIXTURE / 'SUPERVISOR-RESULT.json')
    require(producer['status'] == 'COMPLETE_THREE_LENGTH_FULL_VERIFIER_MECHANISM'
        and terminal['status'] == 'COMPLETE_MECHANISM_MATRIX'
        and terminal['reason'] is None and terminal['all_owned_live_exited']
        and terminal['own_process_reaped'], 'real complete r4 producer required')
    require(not list(FIXTURE.rglob('*FAILED*.json')) and not (FIXTURE / 'STOP_REQUESTED').exists(),
        'r4 failure/STOP takes precedence')
    for name in ('MATRIX-RESULT.json', 'SUPERVISOR-RESULT.json', 'SOURCE-PROVENANCE.json'):
        receipts[str(FIXTURE / name)] = sha(FIXTURE / name)
    fixtures = {}
    for label in ('short', 'middle', 'long'):
        path = FIXTURE / label / 'real-verifier-fixture.pt'
        stat = path.stat()
        fixtures[str(path)] = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    p = dict(schema='static-same-shape-prompt-baseline/v1', job=str(job), s04_job=str(Path(s04_job).resolve()),
        fixture_root=str(FIXTURE), sources=inventory, prerequisites=receipts, fixture_stats=fixtures,
        weight_stats=source['weight_stats'], original_source_plan_sha256=sha(Path(s04_job) / 'SOURCE-PLAN.json'),
        native_binary=source['native_binary'], candidate_binary=source['candidate_binary'],
        arms=list(ARMS), orders=[list(x) for x in ORDERS], queries=6, Graph_rows=36,
        prefill_forwards=2, oracle_forwards=12, eager_admissions=36, warm_forwards=72,
        capture_intents=36, first_replays=36, measured_replays=216,
        expected_forward_intents=410, expected_device_forward_equivalent=374, target_forward_cap=500,
        GPU_wall_cap_seconds=600, active_seconds=540, cleanup_seconds=60, CPU_worker=7,
        output_cap_bytes=128 * 2**20, disk_floor_bytes=5 * 2**30,
        initial_free_bytes=22 * 2**30, runtime_free_bytes=2**29, master_port=29626,
        B14_original_M=7168, B2_original_M=1024, separate_shape_packs=True,
        same_prompt_bytes_required=True, original_61_projection_outputs_only=True,
        all_36_eager_before_any_Graph=True, first_replay_before_parity=True,
        independent_Graph_pools=True, static_admission_not_production_guards=True,
        exclusive_GPU=True, no_retry=True, performance_claim=False,
        production_integration_permission=False, episodes=0, full_new_policy_chunks=0)
    job.mkdir()
    write(job / 'PLAN.json', p)
    write(job / 'SOURCE-PLAN.json', prior)
    return p


def load(job):
    job = Path(job).resolve()
    p = read(job / 'PLAN.json')
    require(p['schema'] == 'static-same-shape-prompt-baseline/v1' and p['job'] == str(job), 'baseline job identity')
    _, receipts = actual_s04(p['s04_job'])
    require(all(p['prerequisites'].get(k) == v for k, v in receipts.items()), 'real S04 receipts differ')
    expected = dict(queries=6, Graph_rows=36, prefill_forwards=2, oracle_forwards=12,
        eager_admissions=36, warm_forwards=72, capture_intents=36, first_replays=36,
        measured_replays=216, expected_forward_intents=410, expected_device_forward_equivalent=374,
        target_forward_cap=500, GPU_wall_cap_seconds=600, active_seconds=540, cleanup_seconds=60,
        CPU_worker=7, output_cap_bytes=128 * 2**20, disk_floor_bytes=5 * 2**30,
        initial_free_bytes=22 * 2**30, runtime_free_bytes=2**29, master_port=29626,
        B14_original_M=7168, B2_original_M=1024, separate_shape_packs=True,
        same_prompt_bytes_required=True, original_61_projection_outputs_only=True,
        all_36_eager_before_any_Graph=True, first_replay_before_parity=True,
        independent_Graph_pools=True, static_admission_not_production_guards=True,
        exclusive_GPU=True, no_retry=True, performance_claim=False,
        production_integration_permission=False, episodes=0, full_new_policy_chunks=0)
    require(all(p.get(k) == v for k, v in expected.items()) and p['arms'] == list(ARMS)
        and p['orders'] == [list(x) for x in ORDERS] and p['fixture_root'] == str(FIXTURE),
        'fixed scope, full recipe, resource constants or arithmetic changed')
    for path, expected_sha in {**p['sources'], **p['prerequisites']}.items():
        require(sha(path) == expected_sha, 'frozen source/evidence changed: ' + path)
    for path, value in p['fixture_stats'].items():
        stat = Path(path).stat()
        require((stat.st_size, stat.st_mtime_ns) == (value['bytes'], value['mtime_ns']), 'fixture stat changed')
    for value in p['weight_stats']:
        stat = Path(value['path']).stat()
        require((stat.st_size, stat.st_mtime_ns) == (value['bytes'], value['mtime_ns']), 'target weight stat changed')
    require(sha(job / 'SOURCE-PLAN.json') == p['original_source_plan_sha256'], 'saved original source plan differs')
    return p
