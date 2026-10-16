"""One-shot, two-site, complete-episode TF usability DEV contract (CPU only)."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = OWN.parents[2]
RUNTIME = ROOT / '投机采样/results/runtime'
WORKTREE = ROOT / '投机采样/wam-p0-worktree'
THRESHOLD = 0.007839680526450805
ARMS = ('T', 'TF_CHILD', 'TF_CONT', 'MLP_CHILD')
ARCH = {'TF_CHILD': 'transformer64', 'TF_CONT': 'transformer64',
        'MLP_CHILD': 'mlp96'}
GLOBAL_STARTED = '2026-10-16T08:48:35+08:00'
WORK_STOP = '2026-10-16T12:43:35+08:00'
FINAL_CLEANUP = '2026-10-16T12:48:35+08:00'
SITE = {
    'local': dict(init=33, gpu_name='NVIDIA A100-SXM4-40GB',
                  va='/rjs/cxz/miniconda3/envs/va/bin/python',
                  client='/rjs/cxz/miniconda3/envs/libero-va/bin/python',
                  server_threads=2, client_threads=2, port=19553, master_port=29553),
    'autodl': dict(init=34, gpu_name='NVIDIA A100-PCIE-40GB',
                   va='/root/autodl-tmp/va-env/bin/python',
                   client='/rjs/cxz/miniconda3/envs/libero-va/bin/python',
                   server_threads=7, client_threads=2, port=19993, master_port=29993),
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temp = path.with_name(path.name + '.pending')
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temp.replace(path)
    else:
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def cases(site):
    init = SITE[site]['init']
    return [dict(arm=arm, task=task, init=init, text_bank=True,
                 key=f'{arm}-t{task}-i{init}')
            for arm in ARMS for task in range(10)]


def source_files():
    files = list(OWN.glob('*.py'))
    for name in ('prefix_acceptance_v1', 'single_worker_chc_gpu_v4',
                 'speculative_e2e_v3', 'shared_prefix_gpu_v3',
                 'shared_prefix_torch_v2', 'bonus_acceptance_v2'):
        files += list((EXP / name).glob('*.py'))
    files += list((EXP / 'shared_prefix_torch_v2').rglob('*.so'))
    files += [EXP / 'formal_e2e_speed_v1' / name
              for name in ('run.py', 'server_entry.py', 'speed_contract.py')]
    files += [EXP / 'target_trace_draft_finetune_v1' / name
              for name in ('adapter.py', 'gpu_actual_cal.py')]
    files += [EXP / 'target_trace_draft_train_v1/adapter.py',
              EXP / 'tf_usable_train_v1/adapter.py',
              EXP / 'tf_usable_train_v1/gpu_actual_cal.py',
              EXP / 'draft_architecture_v1/models.py',
              EXP / 'reset_text_bank_v1/bank.py',
              EXP / 'morning_onpolicy_long8_v4/cleanup.py',
              ROOT / '投机采样/scripts/p0_libero_multichunk_client.py',
              ROOT / '投机采样/scripts/p0_batched_verify.py']
    files += [p for p in (WORKTREE / 'wan_va').rglob('*')
              if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.json')
              and '__pycache__' not in p.parts]
    found = sorted(set(files))
    if any(not p.is_file() for p in found) or not any(p.suffix == '.so' for p in found):
        raise ValueError('actual target/legacy/native source inventory incomplete')
    return found


def _checkpoint(path, architecture, expected_schema):
    import torch
    path = Path(path).resolve(strict=True)
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if (payload.get('schema') != expected_schema or payload.get('architecture') != architecture
            or payload.get('metadata',{}).get('seed') != 20260923):
        raise ValueError('draft checkpoint schema/architecture differs')
    return dict(checkpoint=str(path), checkpoint_sha256=sha(path),
                architecture=architecture, checkpoint_schema=expected_schema,
                bytes=path.stat().st_size)


def normalize_weight_stats(value):
    if isinstance(value,list):
        value={row['path']:dict(bytes=row['bytes'],mtime_ns=row['mtime_ns']) for row in value}
    if not isinstance(value,dict) or len(value)<3:
        raise ValueError('integrated original target weightstat inventory required')
    return value


def training_gate(selected, training_dirs, *, deep=False):
    """Require completed real CPU training before any DEV model can be admitted."""
    if deep:
        from tf_usable_train_v1.adapter import load_selected, load_candidate
    seen=[]
    for arm in ARMS[1:]:
        directory=Path(training_dirs[arm]).resolve(strict=True)
        if (directory/'FAILED.json').exists() or (directory/'STOP_REQUESTED').exists():
            raise ValueError('failed/stopped candidate training: '+arm)
        checkpoint=Path(selected[arm]['checkpoint']).resolve(strict=True)
        arch=ARCH[arm]
        if arm=='TF_CONT':
            final=read(directory/'FINAL-RESULT.json')
            cell=read(checkpoint.parent/'RESULT.json')
            if (final.get('status')!='PASS_PRIMARY_TF_CONT_500_COMPLETE'
                    or final.get('complete_runs')!=1
                    or cell.get('status')!='PASS_TF_CONT_CELL'
                    or cell.get('full_CAL_new_adapter_bitwise_exact') is not True
                    or cell.get('same_CPU_decode_batch_bitwise_exact') is not True
                    or cell.get('normalization_exact_parent') is not True
                    or checkpoint!=directory/'TF_CONT-seed20260923'/'checkpoint.pt'):
                raise ValueError('complete fixed TF_CONT training/cell receipt required')
            if deep:
                model=load_selected(directory,'transformer64',device='cpu',expected_channels=range(7))
        else:
            final=read(directory/'FINAL-RESULT.json')
            cell=read(checkpoint.parent/'RESULT.json')
            if (final.get('status')!='PASS_SIX_TWO_STAGE_FINETUNING_RUNS'
                    or final.get('complete_runs')!=6
                    or cell.get('architecture')!=arch or cell.get('seed')!=20260923
                    or cell.get('full_CAL_new_adapter_bitwise_exact') is not True
                    or cell.get('normalization_exact_parent') is not True
                    or checkpoint!=directory/f'{arch}-seed20260923'/'checkpoint.pt'):
                raise ValueError('complete matched child2500 training/cell receipt required')
            if deep:
                model=load_candidate(checkpoint,arch,device='cpu',expected_channels=range(7),
                                     expected_sha256=selected[arm]['checkpoint_sha256'])
        if deep:
            if (Path(model.checkpoint_path).resolve()!=checkpoint
                    or model.checkpoint_sha256!=selected[arm]['checkpoint_sha256']):
                raise ValueError('training loader returned a different actual candidate')
            del model
        seen.append(arm)
    if seen!=list(ARMS[1:]):
        raise ValueError('three candidate train gates incomplete')


def prepare(job, *, site, child_checkpoint, cont_checkpoint, mlp_checkpoint,
            child_training_dir, cont_training_dir, mlp_training_dir,
            bank, bank_result, reset_equivalence_result, target_checkpoint_path,
            target_checkpoint_id, registry, target_weight_stats, cal_data):
    if site not in SITE:
        raise ValueError('site must be local or autodl')
    job = Path(job).resolve()
    if job.exists() or not job.name.startswith('p0-tf-usable-quality-v1-'):
        raise ValueError('new isolated one-shot job path required')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(WORK_STOP):
        raise TimeoutError('registered common work deadline already passed')
    child = _checkpoint(child_checkpoint, 'transformer64', 'target-trace-draft-two-stage/v1')
    cont = _checkpoint(cont_checkpoint, 'transformer64', 'tf-usable-draft-continuation/v1')
    mlp = _checkpoint(mlp_checkpoint, 'mlp96', 'target-trace-draft-two-stage/v1')
    if 'mlp96-seed20260923' not in mlp['checkpoint']:
        raise ValueError('fixed matched MLP seed23 checkpoint required, not selected seed22')
    selected={'TF_CHILD':child,'TF_CONT':cont,'MLP_CHILD':mlp}
    training_dirs={'TF_CHILD':str(Path(child_training_dir).resolve(strict=True)),
                   'TF_CONT':str(Path(cont_training_dir).resolve(strict=True)),
                   'MLP_CHILD':str(Path(mlp_training_dir).resolve(strict=True))}
    training_gate(selected,training_dirs,deep=True)
    bank = Path(bank).resolve(strict=True)
    bank_receipt = read(bank_result)
    if (bank_receipt.get('status')!=('PASS_COMPLETE_CPU_TEXT_BANK' if site=='local'
                                    else 'PASS_COMPLETE_REMOTE_CPU7_TEXT_BANK')
            or bank_receipt.get('prompts')!=11
            or bank_receipt.get('CUDA_initialized') is not False
            or bank_receipt.get('bank_bytes')!=bank.stat().st_size):
        raise ValueError('actual complete CPU BF16 eleven-prompt bank receipt required')
    reset_receipt=read(reset_equivalence_result)
    expected_pairs=4 if site=='local' else 2
    if (reset_receipt.get('status')!=('PASS_COMPLETE_RESET_BANK_EQUIVALENCE' if site=='local'
                                     else 'PASS_COMPLETE_REMOTE_RESET_BANK_EQUIVALENCE')
            or reset_receipt.get('pairs')!=expected_pairs
            or reset_receipt.get('full_episodes')!=2*expected_pairs
            or reset_receipt.get('errors') not in (None, 0, [])
            or bank.stat().st_size < 1_000_000):
        raise ValueError('actual complete reset-bank equivalence/eleven-prompt bank required')
    cal_data = Path(cal_data).resolve(strict=True)
    if not cal_data.is_file() or cal_data.stat().st_size < 1_000_000:
        raise ValueError('actual complete CAL tensor pack required')
    model = Path(target_checkpoint_path).resolve(strict=True)
    if not model.is_dir() or not Path(registry).is_file():
        raise ValueError('actual original target checkpoint/registry required')
    target_weight_stats=normalize_weight_stats(target_weight_stats)
    for name,stat in target_weight_stats.items():
        actual=Path(name).stat()
        if actual.st_size!=stat['bytes'] or actual.st_mtime_ns!=stat['mtime_ns']:
            raise ValueError('original target weightstat differs: '+name)
    source = source_files()
    profile = SITE[site]
    plan = dict(schema='tf-usable-quality-complete-dev/v1', job=str(job), site=site,
        cases=cases(site), arms=list(ARMS), init=profile['init'], seed=20260922,
        horizon=800, mode='verifier_k2_b2_bonus', selector='prefix_then_cost',
        threshold=THRESHOLD, selected=selected,training_dirs=training_dirs,
        target_checkpoint_path=str(model), target_checkpoint_id=target_checkpoint_id,
        registry=str(Path(registry).resolve(strict=True)),
        target_weight_stats=target_weight_stats,
        bank=str(bank), bank_result=str(Path(bank_result).resolve(strict=True)),
        bank_result_status=bank_receipt['status'],
        reset_equivalence_result=str(Path(reset_equivalence_result).resolve(strict=True)),
        reset_equivalence_status=reset_receipt['status'],bank_reset_equivalence_pairs=expected_pairs,
        bank_stat=dict(bytes=bank.stat().st_size, mtime_ns=bank.stat().st_mtime_ns),
        real_CAL_data=dict(path=str(cal_data), bytes=cal_data.stat().st_size,
                           mtime_ns=cal_data.stat().st_mtime_ns),
        source_sha256={str(p): sha(p) for p in source},
        global_started=GLOBAL_STARTED, absolute_work_stop=WORK_STOP,
        absolute_cleanup_deadline=FINAL_CLEANUP, max_run_wall_sec=10800,
        cleanup_reserve_sec=300, combined_two_site_GPU_seconds_cap=21600,
        init_role='DEV_exposed_not_TEST_40_49', no_old_T_reuse=True,
        workers=1, persistent_model_within_arm=True,
        fresh_initial_RNG_per_episode=True, trace_mode='off',
        teacher_enabled=False, performance_claim=False, shared_GPU_allowed=True,
        initial_free_GiB=22., runtime_free_GiB=.5,
        output_budget_GiB=3., disk_floor_GiB=5.,
        per_episode_wall_sec=40*60, model_ready_wall_sec=900,
        job_target_forward_cap=60000, per_episode_target_forward_cap=4000,
        port=profile['port'], master_port=profile['master_port'],
        OMP_NUM_THREADS=profile['server_threads'], client_threads=profile['client_threads'],
        va_python=profile['va'], client_python=profile['client'],
        required_gpu_name=profile['gpu_name'],
        selection=dict(TF_cont_vs_child_MLP_and_T='descriptive_10pair_DEV',
                       success_at_least_T=True, success_at_least_MLP=True,
                       both_success_P50_max=1.10,
                       both_success_P90_max=1.20, both_success_max_max=1.30,
                       no_3pp_noninferiority_claim=True,
                       no_formal_speed_permission_without_quality_gate=True))
    job.mkdir(parents=True)
    write(job / 'plan.json', plan)
    return plan


def load(job, *, verify=True):
    job = Path(job).resolve(); p = read(job / 'plan.json')
    site = p.get('site')
    if (site not in SITE or p.get('schema') != 'tf-usable-quality-complete-dev/v1'
            or p.get('job') != str(job) or p.get('cases') != cases(site)
            or p.get('arms') != list(ARMS) or p.get('init') != SITE[site]['init']
            or p.get('seed') != 20260922 or p.get('horizon') != 800
            or p.get('mode') != 'verifier_k2_b2_bonus'
            or p.get('selector') != 'prefix_then_cost' or p.get('threshold') != THRESHOLD
            or p.get('global_started') != GLOBAL_STARTED
            or p.get('absolute_work_stop') != WORK_STOP
            or p.get('absolute_cleanup_deadline') != FINAL_CLEANUP
            or p.get('max_run_wall_sec') != 10800 or p.get('cleanup_reserve_sec') != 300
            or p.get('combined_two_site_GPU_seconds_cap') != 21600
            or p.get('output_budget_GiB') != 3. or p.get('disk_floor_GiB') != 5.
            or p.get('initial_free_GiB') != 22. or p.get('runtime_free_GiB') != .5
            or p.get('job_target_forward_cap') != 60000
            or p.get('per_episode_target_forward_cap') != 4000
            or p.get('trace_mode') != 'off' or p.get('teacher_enabled') is not False
            or p.get('workers') != 1 or p.get('performance_claim') is not False
            or p.get('no_old_T_reuse') is not True
            or p.get('bank_reset_equivalence_pairs')!=(4 if site=='local' else 2)):
        raise ValueError('frozen two-site complete DEV protocol differs')
    if verify:
        for name,stat in p['target_weight_stats'].items():
            actual=Path(name).stat()
            if actual.st_size!=stat['bytes'] or actual.st_mtime_ns!=stat['mtime_ns']:
                raise ValueError('frozen target weightstat changed: '+name)
        if set(p['source_sha256']) != {str(f) for f in source_files()}:
            raise ValueError('source inventory changed')
        for path, digest in p['source_sha256'].items():
            if sha(path) != digest:
                raise ValueError('small source changed: ' + path)
        for arm, schema in (('TF_CHILD', 'target-trace-draft-two-stage/v1'),
                            ('TF_CONT', 'tf-usable-draft-continuation/v1'),
                            ('MLP_CHILD', 'target-trace-draft-two-stage/v1')):
            c = p['selected'][arm]
            if c['checkpoint_schema'] != schema or sha(c['checkpoint']) != c['checkpoint_sha256']:
                raise ValueError('candidate checkpoint changed: ' + arm)
        training_gate(p['selected'],p['training_dirs'])
        for item in (dict(path=p['bank'], **p['bank_stat']), p['real_CAL_data']):
            stat = Path(item['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (item['bytes'], item['mtime_ns']):
                raise ValueError('bank/CAL metadata changed')
        receipt=read(p['bank_result']);reset=read(p['reset_equivalence_result'])
        if (receipt.get('status')!=p['bank_result_status'] or receipt.get('prompts')!=11
                or receipt.get('CUDA_initialized') is not False
                or receipt.get('bank_bytes')!=p['bank_stat']['bytes']
                or reset.get('status')!=p['reset_equivalence_status']
                or reset.get('pairs')!=p['bank_reset_equivalence_pairs']
                or reset.get('full_episodes')!=2*p['bank_reset_equivalence_pairs']):
            raise ValueError('frozen complete bank build/equivalence receipt changed')
    return p
