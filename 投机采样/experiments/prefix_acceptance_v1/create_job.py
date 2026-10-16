"""Freeze the authorized valid-prefix quality pilot; preserve all prior jobs."""
import argparse
import copy
import json
import socket
import subprocess
from pathlib import Path
from experiment import ROOT, OWN, PREFIX, assignments, fixed_slots, implementation, metadata, digest, write, stamp
from settings import read


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--cpu',type=Path,required=True)
    ap.add_argument('--evidence',type=Path,required=True)
    a=ap.parse_args();job=a.job.resolve();evidence=a.evidence.resolve()
    if job.parent!=ROOT/'投机采样/results/runtime' or not job.name.startswith(PREFIX):
        raise ValueError('fresh isolated prefix quality job required')
    cpu=json.loads(a.cpu.read_text())
    if cpu['status']!='PASS_CPU_PREFIX_ACCEPTANCE_FULL50_BF16_PARITY' or len(cpu['pairs'])!=12 or cpu['CUDA_initialized']:
        raise ValueError('complete fresh CPU semantic gate required')
    previous=ROOT/'投机采样/results/runtime/p0-bonus-acceptance-v2-20261013-202930'
    old=json.loads((previous/'plan.json').read_text())
    result=json.loads((previous/'FINAL_RESULT.json').read_text())
    carry=result['collection']
    if carry['aggregate_target_intents']!=46654 or carry['aggregate_GPU_seconds']!=14118.697842976078:
        raise ValueError('latest audited cumulative resource carry mismatch')
    frozen={**old['frozen_sha256'],**old['implementation_sha256']}
    for name in ['bonus-acceptance-20261013','draft-design-literature-20261013']:
        seal=ROOT/'.pipeline/research/speculative-sampling/evidence'/name/'ARCHIVE-FREEZE.json'
        for path,h in json.loads(seal.read_text())['files'].items():
            if digest(path)!=h:raise ValueError('sealed prior evidence changed: '+path)
            frozen[path]=h
        frozen[str(seal)]=digest(seal)
    frozen[str(a.cpu.resolve())]=digest(a.cpu)
    protocol=OWN/'QUALITY-PROTOCOL.md';frozen[str(protocol)]=digest(protocol)
    critical={path:h for path,h in frozen.items() if Path(path).suffix in ('.py','.so','.yaml','.json')
        and ('/wan_va/' in path or '/experiments/' in path or path in
        {str(ROOT/'.pipeline/state.yaml'),str(ROOT/'.pipeline/rules.yaml')})}
    for path,h in critical.items():
        if digest(path)!=h:raise ValueError('frozen source changed: '+path)
    preset=read()
    if preset['selector']!='depth_then_cost' or preset['draft_checkpoint']!=old['frozen_A_off']:
        raise ValueError('current DEPTH reference identity mismatch')
    ports=[19851,19852,29751,29752]
    for port in ports:
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    job.mkdir()
    write(job/'preset.json',preset,exclusive=True,durable=True)
    frozen[str(job/'preset.json')]=digest(job/'preset.json')
    p=copy.deepcopy(old)
    p.update(job=str(job),created_at=stamp(),revision='accepted-prefix-cost-quality-v1',
        arms=['T','DEPTH','PREFIX'],inits=[6,7],full_episode_cap=6,EGL_initialization_cap=6,
        model_worker_cap=7,fixed_target_cap=640,fixed_full_chunk_cap=8,
        target_cap=24640,GPU_hard_limit_seconds=2100,
        aggregate_prior_target_intents=46654,aggregate_prior_GPU_seconds=14118.697842976078,
        aggregate_target_cap=75000,previous_aggregate_target_cap=60000,
        assignments=assignments(),fixed_slots=fixed_slots(),ports=ports,
        CPU_core=str(a.cpu.resolve()),evidence=str(evidence),prior_quality_job=str(previous),
        disk_initial_min_bytes=15*2**30,new_output_bytes_cap=2*2**30,
        frozen_sha256=frozen,frozen_file_stats={path:metadata(path) for path in frozen},
        critical_source_sha256=critical,implementation_sha256=implementation(),
        public_worktree_status=subprocess.check_output(['git','-C',str(ROOT/'投机采样/wam-p0-worktree'),
            'status','--porcelain','--untracked-files=all'],text=True),
        preset_snapshot=str(job/'preset.json'),quality_protocol=str(protocol),
        speed_comparison_deferred=True,chunk_events_ON=True,
        fixed_scope='8 complete raw-Torch/shared-O3 numerical chunks for both selectors; same archived states',
        primary_timer='quality diagnostics only; no speed claim',fixed_primary_timer='numerical correctness only',
        promotion_gate='all three arms 2/2 successes, PREFIX environment total no greater than DEPTH and max no greater than T; development pilot only')
    write(job/'plan.json',p,exclusive=True,durable=True)
    write(job/'AUTHORIZATION.json',dict(authorized=True,at=stamp(),plan_sha256=digest(job/'plan.json'),
        user_text=['我同意你的计划，你继续做','中途你自己决定就行，我只要结果','gpu消耗上限你可以暂时拓展到5小时'],
        decision='Valid accepted-prefix selector CPU then raw Torch GPU parity and matched T/DEPTH/PREFIX quality pilot; no training/threshold change/speed claim.',
        new_target_cap=24640,aggregate_target_cap=75000,new_GPU_seconds_cap=2100,
        aggregate_GPU_seconds_cap=18000,no_repeat_permission_required=True),exclusive=True,durable=True)
    (evidence/'JOB.txt').write_text(str(job)+'\n')
    print(json.dumps(dict(job=str(job),plan_sha256=digest(job/'plan.json'),frozen_files=len(frozen)),ensure_ascii=False))


if __name__=='__main__':main()
