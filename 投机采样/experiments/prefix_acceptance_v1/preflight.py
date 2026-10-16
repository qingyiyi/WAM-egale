"""Entry, finite budget, provenance and strict idle tests; CPU only."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess

from experiment import ROOT, OWN, CLIENT, MODES, plan, digest, write, stamp
from paths import S0, load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    args = ap.parse_args()
    if os.getenv('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU preflight boundary')
    job = args.job.resolve()
    p = plan(job)
    out = job / 'preflight'
    out.mkdir()
    checks = []
    for path in OWN.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    checks.append('all_sources_compile_manifest_immutable')
    cpu = json.loads(Path(p['CPU_core']).read_text())
    assert cpu['status']=='PASS_CPU_PREFIX_ACCEPTANCE_FULL50_BF16_PARITY' and len(cpu['pairs']) == 12 and not cpu['CUDA_initialized']
    assert cpu['reference_attention_backend'] == 'shared'
    assert cpu['raw_unshared_original_Torch_bonus_GPU_gate_still_required']
    checks.append('six_full50_BF16_same_attention_backend_helper_pairs_raw_Torch_GPU_oracle_pending')
    assert p['aggregate_prior_target_intents']+p['target_cap'] == 71294
    assert p['aggregate_prior_GPU_seconds']+p['GPU_hard_limit_seconds'] < 18000
    assert len(p['fixed_slots']) * len(p['snapshots']) == 8
    checks.append('finite_reservations_and_user_five_hour_boundary')
    for port in p['ports']:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    checks.append('server_master_ports_available')
    disk = load('bonus_frozen_disk', S0 / 'disk_guard.py')
    disk.entry(shutil.disk_usage(job).free, p)
    disk.runtime(16*2**30, 0, p)
    for free, used in [(11*2**30, 0), (15*2**30, 2*2**30+1)]:
        try:
            disk.runtime(free, used, p)
            raise AssertionError('disk negative guard missing')
        except RuntimeError:
            pass
    checks.append('actual_free_disk_and_floor_output_guards')
    code = 'import sys;sys.path.insert(0,' + repr(str(ROOT/'投机采样/scripts')) + ');import p0_libero_multichunk_client;import libero.libero.benchmark;print("PASS_CLIENT_CPU_IMPORT")'
    with (out / 'client-import.log').open('xb') as f:
        r = subprocess.run([CLIENT, '-B', '-c', code], stdout=f, stderr=subprocess.STDOUT,
            env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'MUJOCO_GL': 'egl',
                 'PYOPENGL_PLATFORM': 'egl', 'PYTHONDONTWRITEBYTECODE': '1'}, timeout=60)
    assert r.returncode == 0
    checks.append('actual_LIBERO_client_CPU_import')
    from supervisor import Campaign
    q = object.__new__(Campaign)
    q.job, q.p = job, p
    for wave in p['assignments']:
        assert len(wave) == 2 and {x['init'] for x in wave} == {6, 7}
        for x in wave:
            folder = job/'campaigns'/x['arm']/'workers'/f"worker{x['init']-6}"
            env = q.environment(x['arm'], folder)
            assert env['BONUS_ACCEPT_INIT'] == str(x['init'])
            assert env['WAM_P0_ACTION_DRAFT_MODE'] == MODES[x['arm']]
            assert env['MUJOCO_GL'] == 'egl' and env['CUDA_VISIBLE_DEVICES'] == '0'
            assert not any(env.get(k) for k in ['WAM_P0_STATE_TEACHER_DIR', 'P0_NATIVE_DIAGNOSTIC_DIR'])
            assert str(OWN/'server_entry.py') in q.model_cmd('server_entry.py', p['ports'][0], p['ports'][2], folder/'server_output')
    checks.append('all_six_exact_three_arm_launches_provenance_diagnostics_off')
    assert not Campaign.startup_ready({'utilization_percent': 0}, [999])
    assert not Campaign.startup_ready({'utilization_percent': 3}, [])
    assert Campaign.startup_ready({'utilization_percent': 2}, [])
    code = (OWN/'supervisor.py').read_text()
    assert code.index('while quiet<15:') < code.index('self.pre_inference=False;self.next_sample=0')
    assert 'if self.latest_foreign and not self.pre_inference:' in code
    checks.append('strict_idle_and_foreign_GPU_negatives_formal_guard_retained')
    from experiment import worker_entry
    import inspect
    assert 'durable forward reservation required' in inspect.getsource(worker_entry)
    assert (OWN/'cleanup.py').read_bytes() == (OWN.parent/'optimized_e2e_resume_v2/cleanup.py').read_bytes()
    checks.append('frozen_owned_only_cleanup_and_prelaunch_reservations')
    for ref in p['snapshots']:
        assert digest(ref['path']) == ref['file_sha256']
    plan(job)
    checks.append('original_fixtures_frozen_sources_checkpoint_and_authorization')
    write(out/'CPU-GATE.json', dict(status='PASS', at=stamp(), plan_sha256=digest(job/'plan.json'),
        checks=checks, checks_run=len(checks), CUDA_initialized=False, GPU_started=False,
        actual_free_bytes=shutil.disk_usage(job).free), exclusive=True, durable=True)
    print(json.dumps(dict(status='PASS', checks=len(checks))))


if __name__ == '__main__':
    main()
