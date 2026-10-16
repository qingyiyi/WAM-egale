"""Local complete-episode reset/quality work; no performance claim on shared GPU."""
from __future__ import annotations

import json
from pathlib import Path

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
ROOT = OWN.parents[2]
WORKTREE = ROOT / '投机采样/wam-p0-worktree'
RUNTIME = ROOT / '投机采样/results/runtime'
VA = Path('/rjs/cxz/miniconda3/envs/va/bin/python')
CLIENT = Path('/rjs/cxz/miniconda3/envs/libero-va/bin/python')
REFERENCE = Path('/var/tmp/cxz-formal-e2e-speed/p0-formal-e2e-speed-v1-multi-r3-20261015/plan.json')
BANK = RUNTIME / 'p0-reset-text-bank-v1-20261015-cpu7-r2/bank.pt'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if replace:
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
        temporary.replace(path)
    else:
        with path.open('x') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')


def load(job):
    job = Path(job).resolve()
    p = read(job / 'plan.json')
    if (p.get('schema') != 'local-complete-reset-quality/v1'
            or p['job'] != str(job) or p['horizon'] != 800
            or p['performance_claim'] is not False or p['workers'] != 1
            or p['trace_mode'] != 'off' or p['teacher_enabled'] is not False
            or p['seed'] != 20260922 or not p['fresh_initial_RNG_per_episode']):
        raise ValueError('local complete-task protocol differs')
    return p


def prepare(job):
    job = Path(job).resolve()
    if job.parent != RUNTIME or job.exists():
        raise ValueError('fresh project runtime output required')
    reference = read(REFERENCE)
    cases = []
    for arm in ('T', 'A_OFF'):
        for task in (0, 1):
            for cache in ((False, True) if task == 0 else (True, False)):
                cases.append(dict(arm=arm, task=task, init=29, text_bank=cache,
                                  key=f'{arm}-t{task}-i29-' + ('bank' if cache else 'original')))
    p = dict(schema='local-complete-reset-quality/v1', job=str(job),
             purpose='Original versus cached final T5 embeddings on complete LIBERO episodes',
             bank=str(BANK), reference_plan=str(REFERENCE), cases=cases,
             arms=['T', 'A_OFF'], seed=20260922, horizon=800,
             threshold=reference['threshold'], mode=reference['mode'],
             draft_checkpoint=reference['draft_checkpoint'],
             target_checkpoint_path=reference['target_checkpoint_path'],
             target_checkpoint_id=reference['target_checkpoint_id'], registry=reference['registry'],
             workers=1, worker_reason='One local shared 40GB GPU; preserve memory headroom for other users',
             trace_mode='off', teacher_enabled=False, performance_claim=False,
             fresh_initial_RNG_per_episode=True, persistent_model_within_arm=True,
             no_libero_horizon_truncation=True, per_episode_wall_sec=2400,
             max_run_wall_sec=14400, initial_free_GiB=22, runtime_free_GiB=.5,
             output_budget_GiB=2, disk_floor_GiB=5,
             port=19521, master_port=29521,
             OMP_NUM_THREADS=7, client_threads=2,
             selection='All four original/bank pairs require bitwise commands, initial/final simulator/controller/RNG, reset state, valid final KV and streaming VAE state equality')
    job.mkdir(parents=True)
    write(job / 'plan.json', p)
    return p
