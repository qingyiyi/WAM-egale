"""Freeze a fresh quality experiment within the existing five-hour authority."""
import argparse
import copy
import json
import socket
import subprocess
from pathlib import Path

from experiment import ROOT, OWN, PREFIX, assignments, fixed_slots, implementation, metadata, digest, write, stamp
from settings import read


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    ap.add_argument('--cpu', type=Path, required=True)
    ap.add_argument('--evidence', type=Path, required=True)
    a = ap.parse_args()
    job, evidence = a.job.resolve(), a.evidence.resolve()
    if job.parent != ROOT / '投机采样/results/runtime' or not job.name.startswith(PREFIX):
        raise ValueError('fresh isolated acceptance job required')
    cpu = json.loads(a.cpu.read_text())
    if (cpu['status'] != 'PASS_CPU_BONUS_ACCEPTANCE_FULL50_BF16_PARITY' or
            len(cpu['pairs']) != 12 or cpu['CUDA_initialized']):
        raise ValueError('complete bonus/selector CPU gate required')
    previous = ROOT / '投机采样/results/runtime/p0-bonus-e2e-v1-20261013-183704'
    old = json.loads((previous / 'plan.json').read_text())
    carry = json.loads((previous / 'FINAL_RESULT.json').read_text())['collection']
    if carry['aggregate_target_intents'] != 42216 or carry['aggregate_GPU_seconds'] != 13143.63557540439:
        raise ValueError('audited resource carry mismatch')
    for path, expected in old['frozen_file_stats'].items():
        if metadata(path) != expected:
            raise ValueError('old frozen metadata changed: ' + path)
    frozen = {**old['frozen_sha256'], **old['implementation_sha256']}
    for name in ['bonus-e2e-20261013', 'speedup-evaluation-literature-20261013']:
        seal = ROOT / '.pipeline/research/speculative-sampling/evidence' / name / 'ARCHIVE-FREEZE.json'
        for path, h in json.loads(seal.read_text())['files'].items():
            if digest(path) != h:
                raise ValueError('old sealed evidence changed: ' + path)
            frozen[path] = h
        frozen[str(seal)] = digest(seal)
    frozen[str(a.cpu.resolve())] = digest(a.cpu)
    critical = {path: h for path, h in frozen.items()
        if Path(path).suffix in ('.py', '.so', '.yaml', '.json') and
        ('/wan_va/' in path or '/experiments/' in path or
         path in {str(ROOT / '.pipeline/state.yaml'), str(ROOT / '.pipeline/rules.yaml')})}
    for path, h in critical.items():
        if digest(path) != h:
            raise ValueError('critical original source changed: ' + path)
    failure_seal = evidence / 'V1-FAILED-FREEZE.json'
    for path,h in json.loads(failure_seal.read_text())['files'].items():
        if digest(path)!=h: raise ValueError('failed instrumentation evidence changed: '+path)
        frozen[path]=h
    frozen[str(failure_seal)]=digest(failure_seal)
    failed=ROOT/'投机采样/results/runtime/p0-bonus-acceptance-v1-20261013-194000/FAILED.json'
    charged=json.loads(failed.read_text())
    if charged['aggregate_target_intents']!=42856 or charged['aggregate_GPU_seconds']!=13205.520916400477:
        raise ValueError('conservative failed charge carry mismatch')
    ports = [19841, 19842, 29741, 29742]
    for port in ports:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    preset = read()
    if preset['selector'] != 'global' or preset['draft_checkpoint'] != old['frozen_A_off']:
        raise ValueError('frozen bonus-default anchor mismatch')
    job.mkdir()
    write(job / 'preset.json', preset, exclusive=True, durable=True)
    frozen[str(job / 'preset.json')] = digest(job / 'preset.json')
    p = copy.deepcopy(old)
    p.update(job=str(job), created_at=stamp(), revision='bonus-acceptance-depth-first-quality-v2-request-context',
        arms=['GLOBAL', 'DEPTH'], full_episode_cap=4, EGL_initialization_cap=4,
        model_worker_cap=5, fixed_target_cap=640, fixed_full_chunk_cap=8,
        target_cap=16640, GPU_hard_limit_seconds=2100,
        aggregate_prior_target_intents=42856, aggregate_prior_GPU_seconds=13205.520916400477,
        previous_aggregate_target_cap=50000, aggregate_target_cap=60000,
        assignments=assignments(), fixed_slots=fixed_slots(), ports=ports,
        CPU_core=str(a.cpu.resolve()), evidence=str(evidence), prior_bonus_job=str(previous),
        new_target_cap_decided_by='agent finite quality reservation under persistent self-decision authority; old agent-created plans/caps unchanged',
        disk_initial_min_bytes=15*2**30, new_output_bytes_cap=2*2**30,
        frozen_sha256=frozen, frozen_file_stats={path: metadata(path) for path in frozen},
        critical_source_sha256=critical, implementation_sha256=implementation(),
        public_worktree_status=subprocess.check_output(['git', '-C', str(ROOT / '投机采样/wam-p0-worktree'),
            'status', '--porcelain', '--untracked-files=all'], text=True),
        preset_snapshot=str(job / 'preset.json'), speed_comparison_deferred=True, chunk_events_ON=True,
        bonus_only_policy_change=False, acceptance_selector_only_policy_change=True,
        fixed_scope='8 complete numerical oracles; both selectors vs raw Torch at 2 archived long states; no timed repetitions',
        primary_timer='diagnostic only; quality first; no acceleration claim',
        fixed_primary_timer='numerical correctness only; no acceleration claim',
        promotion_gate='acceptance improves, both matched successes preserved, total environment steps no worse than GLOBAL and original T; one-task development pilot only')
    p.pop('CPU_bonus', None)
    write(job / 'plan.json', p, exclusive=True, durable=True)
    write(job / 'AUTHORIZATION.json', dict(authorized=True, at=stamp(), plan_sha256=digest(job/'plan.json'),
        user_text=['把bonus先打开，或者我们先确定bonus，和trace中步数的关系，提高接受率，然后再确认加速比',
                   '中途你自己决定就行，我只要结果', 'gpu消耗上限你可以暂时拓展到5小时'],
        decision='Bonus ON/global default; constant-gate longest-admissible-path quality pilot; no new training, threshold relaxation, shortened diffusion or speed comparison.',
        new_target_cap=16640, aggregate_target_cap=60000, new_GPU_seconds_cap=2100,
        aggregate_GPU_seconds_cap=18000, no_repeat_permission_required=True), exclusive=True, durable=True)
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / 'JOB.txt').write_text(str(job) + '\n')
    print(json.dumps(dict(job=str(job), plan_sha256=digest(job/'plan.json'), frozen_files=len(frozen)), ensure_ascii=False))


if __name__ == '__main__':
    main()
