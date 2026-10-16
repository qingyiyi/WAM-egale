"""Prospective four-shard expanded DEV contract; TEST remains disabled."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = EXP.parent.parent
RUNTIME = ROOT/'投机采样/results/runtime'
WORKTREE = ROOT/'投机采样/wam-p0-worktree'
ARMS = ('T', 'TF_CONT', 'MLP_CONT')
ARCH = {'TF_CONT': 'transformer64', 'MLP_CONT': 'mlp96'}
TODAY_JOB = RUNTIME/'p0-draft-freeze-quality-v1-local-20261016-r1'
MANIFEST = OWN/'DEV-MANIFEST.json'
TEST_MANIFEST = OWN/'TEST-MANIFEST.json'
SHARDS = ('init36', 'init37', 'init38', 'init39')
TODAY_RESULT_STATUS = 'COMPLETE50_PROSPECTIVE_DEV_ANALYSIS'
SHARD_RESULT_STATUS = 'COMPLETE30_FROZEN_VALIDATION_SHARD_ANALYSIS'
SHARD_COLLECTION_STATUS = 'PASS_COMPLETE_FROZEN_VALIDATION_SHARD_COLLECTION'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'
    if replace:
        tmp = path.with_name(path.name+'.pending')
        tmp.write_text(raw)
        tmp.replace(path)
    else:
        with path.open('x') as stream:
            stream.write(raw)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _no_failure_tree(job):
    return not any(job.rglob('*FAILED*.json')) and not any(job.rglob('*STOP*'))


def _manifest(path=MANIFEST):
    path = Path(path).resolve(strict=True)
    if path != MANIFEST or not TEST_MANIFEST.is_file():
        raise ValueError('only the frozen expanded-DEV manifest is executable')
    test = read(TEST_MANIFEST)
    if (test.get('schema') != 'draft-frozen-validation-test-reference-disabled/v1'
            or test.get('launch_allowed') is not False
            or test.get('prepare_or_run_rejected') is not True):
        raise ValueError('TEST40-49 manifest must stay reference-only and disabled')
    m = read(path)
    if (m.get('schema') != 'draft-frozen-validation-dev-manifest/v1'
            or m.get('status') != 'PREPARED_NOT_RUN_EXPANDED_DEV'
            or m.get('arms') != list(ARMS)
            or m.get('horizon') != 800 or m.get('seed') != 20260922
            or [s.get('id') for s in m.get('shards', [])] != list(SHARDS)):
        raise ValueError('expanded DEV manifest schema/order differs')
    for idx, shard in enumerate(m['shards']):
        init = 36+idx
        rows = shard.get('cases')
        if shard.get('init') != init or not isinstance(rows, list) or len(rows) != 30:
            raise ValueError('each fixed init shard requires 30 complete cases')
        expected = [dict(arm=arm, task=task, init=init, group='expanded_DEV', rep=0,
                         key=f'{arm}-expanded_DEV-t{task}-i{init}-r0', text_bank=True,
                         shard=shard['id']) for arm in ARMS for task in range(10)]
        if rows != expected:
            raise ValueError('expanded DEV task/init/arm keys or order differ')
    return m


def cases(shard, manifest=MANIFEST):
    m = _manifest(manifest)
    if shard not in SHARDS:
        raise ValueError('only init36..39 DEV shards may run')
    return next(s['cases'] for s in m['shards'] if s['id'] == shard)


def source_files():
    # No glob: the actual frozen today50 inference inventory and this package's
    # explicit, reviewable files are the complete executable source binding.
    baseline = read(TODAY_JOB/'plan.json')['source_sha256']
    own = (OWN/'protocol.py', OWN/'run.py', OWN/'server_entry.py',
           OWN/'client_entry.py', OWN/'analyze.py', OWN/'README.md',
           MANIFEST, TEST_MANIFEST)
    return sorted(set(map(Path, baseline)).union(own))


def _today_gate():
    job = TODAY_JOB.resolve(strict=True)
    from draft_freeze_quality_v1.protocol import load as load_today
    load_today(job)
    if not _no_failure_tree(job):
        raise ValueError('today50 has failure/STOP; expanded DEV admission denied')
    plan = read(job/'plan.json')
    collection = read(job/'COLLECTION.json')
    result = read(job/'RESULT.json')
    cleanup = read(job/'FINAL-OWNED-CLEANUP.json')
    clock = read(job/'CLOCK-RESULT.json')
    pair = read(job/'PAIRING-PIXELS.json')
    if (plan.get('schema') != 'draft-freeze-complete-dev/v1'
            or collection.get('status') != 'PASS_COMPLETE_DRAFT_FREEZE_COLLECTION'
            or collection.get('full_episodes') != 50
            or collection.get('completed') != [row['key'] for arm in ARMS
                for row in plan['cases'] if row['arm'] == arm]
            or result.get('status') != TODAY_RESULT_STATUS
            or result.get('retained_primary_for_prospective_independent_TEST') is not True
            or result.get('physical_RNG_pairing_passed') is not True
            or pair.get('status') != 'COMPLETE28_INITIAL_PAIRING_AUDIT'
            or pair.get('physical_RNG_pairing_passed') is not True
            or cleanup.get('all_roots_reaped') is not True
            or clock.get('all_observed_owned_live_processes_exited') is not True
            or clock.get('error') is not None):
        raise ValueError('today50 terminal collection/retention/cleanup gate failed')
    if any(read(job/'arms'/arm/'OWNED-CLEANUP.json').get('all_roots_reaped') is not True
           for arm in ARMS):
        raise ValueError('today50 arm cleanup incomplete')
    frozen = read(ROOT/'投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/PREREGISTRATION.json')
    for arm, key in (('TF_CONT', 'fixed_TF_checkpoint_sha256'),
                     ('MLP_CONT', 'fixed_MLP_checkpoint_sha256')):
        if plan['selected'][arm]['checkpoint_sha256'] != frozen[key]:
            raise ValueError('fixed draft differs from original preregistration')
    if any(result.get('permissions', {}).get(k) is not False for k in
           ('engineering_quality_screen', 'independent_test_complete',
            'noninferiority_established', 'formal_performance_permission')):
        raise ValueError('today50 old four permissions must remain false')
    names = ('plan.json', 'COLLECTION.json', 'RESULT.json',
             'FINAL-OWNED-CLEANUP.json', 'CLOCK-RESULT.json',
             'PAIRING-PIXELS.json')
    return {str(job/name): sha(job/name) for name in names}


def _registration(path, manifest):
    path = Path(path).resolve(strict=True)
    r = read(path)
    metadata_root = path.parent
    if (metadata_root.parent != RUNTIME
            or not metadata_root.name.startswith('p0-draft-frozen-validation-v1-cohort-')
            or r.get('cohort_metadata_root') != str(metadata_root)):
        raise ValueError('cohort registration must be an independent runtime metadata directory')
    if (r.get('schema') != 'draft-frozen-validation-execution-registration/v1'
            or r.get('status') != 'FROZEN_BEFORE_EXPANDED_DEV'
            or r.get('manifest_sha256') != sha(MANIFEST)
            or r.get('test_manifest_sha256') != sha(TEST_MANIFEST)
            or r.get('shard_order') != list(SHARDS)
            or set(r.get('shard_jobs', {})) != set(SHARDS)
            or len(set(r['shard_jobs'].values())) != 4
            or r.get('per_shard_max_seconds') != 14400
            or r.get('cleanup_reserve_seconds') != 300
            or r.get('cohort_gpu_job_cap_seconds') != 28800
            or r.get('cohort_wall_cap_seconds') != 43200
            or r.get('cohort_output_budget_GiB') != 5
            or r.get('per_shard_output_budget_GiB') != 3
            or r.get('disk_floor_GiB') != 5):
        raise ValueError('prospective four-shard budget/manifest registration absent or altered')
    start = datetime.fromisoformat(r['cohort_started_at'])
    work = datetime.fromisoformat(r['absolute_work_stop'])
    hard = datetime.fromisoformat(r['absolute_cleanup_deadline'])
    if (start.tzinfo is None or work.tzinfo is None or hard.tzinfo is None
            or not start < work < hard
            or (hard-start).total_seconds() > 43200
            or (hard-work).total_seconds() < 300):
        raise ValueError('registered cohort wall clock exceeds twelve hours or lacks cleanup reserve')
    wrapper = Path(r['clock_wrapper_path']).resolve(strict=True)
    if (str(wrapper) != r['clock_wrapper_path']
            or wrapper.parent == OWN or sha(wrapper) != r['clock_wrapper_sha256']
            or r.get('clock_provenance_name') != 'DRAFT-FROZEN-VALIDATION-CLOCK-PROVENANCE.json'):
        raise ValueError('new independent clock source/provenance not frozen')
    for shard, name in r['shard_jobs'].items():
        job = Path(name).resolve()
        if (str(job) != name or job.parent != RUNTIME or job == metadata_root
                or not job.name.startswith('p0-draft-frozen-validation-v1-local-'+shard)):
            raise ValueError('independent shard job path not fixed under runtime')
    return path, r


def _prior_gate(registration, shard, registration_sha256):
    idx = SHARDS.index(shard)
    proofs = {}
    charged = 0.0
    source_inventory = {str(path): sha(path) for path in source_files()} if idx else None
    for old_id in SHARDS[:idx]:
        old = Path(registration['shard_jobs'][old_id]).resolve(strict=True)
        if not _no_failure_tree(old):
            raise ValueError('prior shard has failure/STOP; next shard denied')
        plan = read(old/'plan.json')
        collection = read(old/'COLLECTION.json')
        result = read(old/'RESULT.json')
        final = read(old/'FINAL-OWNED-CLEANUP.json')
        clock = read(old/'CLOCK-RESULT.json')
        if (plan.get('shard') != old_id
                or plan.get('registration_sha256') != registration_sha256
                or plan.get('shard_jobs') != registration['shard_jobs']
                or plan.get('manifest_sha256') != sha(MANIFEST)
                or plan.get('source_sha256') != source_inventory
                or plan.get('selected') != read(TODAY_JOB/'plan.json')['selected']
                or collection.get('status') != SHARD_COLLECTION_STATUS
                or collection.get('full_episodes') != 30
                or collection.get('completed') != [row['key'] for row in plan['cases']]
                or result.get('status') != SHARD_RESULT_STATUS
                or final.get('all_roots_reaped') is not True
                or clock.get('status') != 'CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS'
                or clock.get('all_observed_owned_live_processes_exited') is not True
                or clock.get('original_quality_RESULT_present') is not True
                or clock.get('original_quality_FAILED_present') is not False
                or clock.get('external_stop_reason') is not None
                or clock.get('no_foreign_signals') is not True
                or clock.get('signals')
                or clock.get('error') is not None):
            raise ValueError('prior shard lacks full terminal result and owned cleanup')
        if not (old/'REPORT.md').is_file() or not (old/'PAIRING-PIXELS.json').is_file():
            raise ValueError('prior shard actual report/pairing evidence absent')
        if any(read(old/'arms'/arm/'OWNED-CLEANUP.json').get('all_roots_reaped') is not True
               for arm in ARMS):
            raise ValueError('prior shard arm cleanup incomplete')
        elapsed = clock.get('elapsed_quality_process_seconds')
        if not isinstance(elapsed, (int, float)) or not 0 < elapsed <= 14400:
            raise ValueError('prior shard actual GPU-job lifetime unavailable')
        charged += elapsed
        names = ('plan.json', 'COLLECTION.json', 'RESULT.json',
                 'FINAL-OWNED-CLEANUP.json', 'CLOCK-RESULT.json')
        proofs.update({str(old/name): sha(old/name) for name in names})
    for old_id, next_id in zip(SHARDS[:idx], SHARDS[1:idx+1]):
        previous = Path(registration['shard_jobs'][old_id])
        release = previous/'NEXT-SHARD-RELEASE.json'
        marker = read(release)
        if (marker.get('status') != 'ALLOW_NEXT_FROZEN_VALIDATION_SHARD'
                or marker.get('next_shard') != next_id
                or marker.get('prior_result_sha256') != sha(previous/'RESULT.json')
                or marker.get('prior_clock_result_sha256') != sha(previous/'CLOCK-RESULT.json')):
            raise ValueError('explicit post-terminal outer release for next shard absent')
        proofs[str(release)] = sha(release)
    if 28800-charged <= 300:
        raise ValueError('cohort charged GPU-job lifetime has no active-work budget beyond cleanup')
    return proofs, charged


def validate(plan, job, *, verify=True):
    job = Path(job).resolve()
    manifest = _manifest(plan['manifest_path'])
    shard = plan['shard']
    if shard not in SHARDS:
        raise ValueError('TEST and unknown shards cannot be prepared or run')
    reg_path, reg = _registration(plan['registration'], manifest)
    if (plan.get('schema') != 'draft-frozen-validation-shard/v1'
            or plan.get('profile') != 'expanded_DEV_init36_39'
            or plan.get('job') != str(job)
            or reg['shard_jobs'][shard] != str(job)
            or plan.get('shard_jobs') != reg['shard_jobs']
            or plan.get('site') != 'local'
            or plan.get('arms') != list(ARMS)
            or plan.get('cases') != cases(shard)
            or plan.get('manifest_sha256') != sha(MANIFEST)
            or plan.get('test_manifest_sha256') != sha(TEST_MANIFEST)
            or plan.get('registration_sha256') != sha(reg_path)
            or plan.get('horizon') != 800 or plan.get('seed') != 20260922
            or plan.get('mode') != 'verifier_k2_b2_bonus'
            or plan.get('selector') != 'prefix_then_cost'
            or plan.get('observation_protocol') != 'unchanged_legacy_cached_observations'
            or plan.get('max_run_wall_sec') != 14400
            or plan.get('cleanup_reserve_sec') != 300
            or plan.get('cohort_gpu_job_cap_seconds') != 28800
            or plan.get('cohort_gpu_job_seconds_remaining') != 28800-plan.get('charged_gpu_job_seconds_before_shard', -1)
            or plan.get('cohort_gpu_job_seconds_remaining', 0) <= 300
            or plan.get('cohort_output_budget_GiB') != 5
            or plan.get('cohort_metadata_root') != reg['cohort_metadata_root']
            or plan.get('clock_wrapper_path') != reg['clock_wrapper_path']
            or plan.get('clock_wrapper_sha256') != reg['clock_wrapper_sha256']
            or plan.get('clock_provenance_name') != reg['clock_provenance_name']
            or plan.get('output_budget_GiB') != 3
            or plan.get('workers') != 1
            or plan.get('OMP_NUM_THREADS') != 2 or plan.get('client_threads') != 2
            or plan.get('absolute_work_stop') != reg['absolute_work_stop']
            or plan.get('absolute_cleanup_deadline') != reg['absolute_cleanup_deadline']
            or plan.get('cohort_started_at') != reg['cohort_started_at']
            or plan.get('global_started') != reg['cohort_started_at']
            or plan.get('init') != next(s['init'] for s in manifest['shards'] if s['id'] == shard)
            or plan.get('init_role') != 'expanded_DEV_init36to39_only'
            or 'combined_two_site_GPU_seconds_cap' in plan
            or plan.get('cohort_cleanup_deadline') != reg['absolute_cleanup_deadline']
            or plan.get('performance_claim') is not False
            or plan.get('formal_performance_permission') is not False
            or plan.get('independent_test_complete') is not False
            or plan.get('engineering_quality_screen') is not False
            or plan.get('noninferiority_established') is not False):
        raise ValueError('frozen expanded-DEV shard plan contract differs')
    if verify:
        today = _today_gate()
        if today != plan['today_terminal_receipts']:
            raise ValueError('today50 terminal receipt bytes differ')
        prior, charged = _prior_gate(reg, shard, sha(reg_path))
        if prior != plan['prior_shard_evidence'] or charged != plan['charged_gpu_job_seconds_before_shard']:
            raise ValueError('prior shard source/budget evidence differs')
        if {str(p): sha(p) for p in source_files()} != plan['source_sha256']:
            raise ValueError('frozen inference/analyzer/manifest sources changed')
        if sha(reg['clock_wrapper_path']) != reg['clock_wrapper_sha256']:
            raise ValueError('separate future clock source changed')
        for arm, expected in manifest['fixed_checkpoint_sha256'].items():
            selected = plan['selected'][arm]
            if (selected['checkpoint_sha256'] != expected
                    or sha(selected['checkpoint']) != expected
                    or selected['architecture'] != ARCH[arm]):
                raise ValueError('draft candidate identity differs: '+arm)
        for path, stat in plan['target_weight_stats'].items():
            current = Path(path).stat()
            if current.st_size != stat['bytes'] or current.st_mtime_ns != stat['mtime_ns']:
                raise ValueError('original target weightstat differs')
        for key in ('bank', 'real_CAL_data'):
            path = Path(plan[key] if key == 'bank' else plan[key]['path'])
            stat = plan['bank_stat'] if key == 'bank' else plan[key]
            current = path.stat()
            if current.st_size != stat['bytes'] or current.st_mtime_ns != stat['mtime_ns']:
                raise ValueError('actual bank/CAL stat differs')
    return plan


def prepare(job, *, manifest, shard, registration):
    job = Path(job).resolve()
    if job.exists():
        raise ValueError('fresh one-shot shard job required')
    m = _manifest(manifest)
    if shard not in SHARDS:
        raise ValueError('only the four fixed expanded DEV shards are runnable')
    reg_path, reg = _registration(registration, m)
    if str(job) != reg['shard_jobs'][shard]:
        raise ValueError('job does not equal prospectively fixed shard path')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(reg['absolute_work_stop']):
        raise TimeoutError('registered expanded DEV work deadline passed')
    today = _today_gate()
    prior, charged = _prior_gate(reg, shard, sha(reg_path))
    today_plan = read(TODAY_JOB/'plan.json')
    base = dict(today_plan)
    base.pop('combined_two_site_GPU_seconds_cap', None)
    base['today_selection_rule'] = base.pop('selection')
    base.update(schema='draft-frozen-validation-shard/v1', profile='expanded_DEV_init36_39',
        job=str(job), site='local', shard=shard, shard_order=list(SHARDS),
        init=next(s['init'] for s in m['shards'] if s['id'] == shard),
        init_role='expanded_DEV_init36to39_only',
        cases=cases(shard), arms=list(ARMS), planned_full_episodes=30,
        manifest_path=str(MANIFEST), manifest_sha256=sha(MANIFEST),
        test_manifest_path=str(TEST_MANIFEST), test_manifest_sha256=sha(TEST_MANIFEST),
        registration=str(reg_path), registration_sha256=sha(reg_path),
        today_terminal_receipts=today, prior_shard_evidence=prior,
        charged_gpu_job_seconds_before_shard=charged,
        cohort_gpu_job_cap_seconds=28800,
        cohort_gpu_job_seconds_remaining=28800-charged,
        cohort_output_budget_GiB=5, output_budget_GiB=3,
        cohort_metadata_root=reg['cohort_metadata_root'],
        cohort_started_at=reg['cohort_started_at'],
        global_started=reg['cohort_started_at'],
        cohort_cleanup_deadline=reg['absolute_cleanup_deadline'],
        absolute_work_stop=reg['absolute_work_stop'],
        absolute_cleanup_deadline=reg['absolute_cleanup_deadline'],
        max_run_wall_sec=14400, cleanup_reserve_sec=300,
        shard_jobs=reg['shard_jobs'],
        clock_wrapper_path=reg['clock_wrapper_path'],
        clock_wrapper_sha256=reg['clock_wrapper_sha256'],
        clock_provenance_name=reg['clock_provenance_name'],
        OMP_NUM_THREADS=2, client_threads=2, workers=1,
        observation_protocol='unchanged_legacy_cached_observations',
        source_sha256={str(p): sha(p) for p in source_files()},
        formal_performance_permission=False,
        independent_test_complete=False, no_old_T_reuse=True,
        engineering_quality_screen=False, noninferiority_established=False,
        performance_claim=False, absolute_no_AutoDL=True)
    validate(base, job)
    job.mkdir(parents=True)
    write(job/'plan.json', base)
    return base


def load(job, *, verify=True):
    job = Path(job).resolve(strict=True)
    return validate(read(job/'plan.json'), job, verify=verify)
