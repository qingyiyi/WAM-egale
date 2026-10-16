"""One diagnostic deployment, exclusive GPU, finite reservations and owned cleanup."""
import argparse
import fcntl
import json
import os
import signal
import socket
import time
import traceback
from pathlib import Path
from mv_profile_contract import OWN, ROOT, S0, VA, plan, load, write, stamp, digest

old = load('mv_profile_frozen_supervisor', S0 / 'supervisor.py')


class Campaign(old.Campaign):
    def __init__(self, job):
        self.job = Path(job).resolve()
        self.p = plan(job, gpu=True)
        self.nv = old.NVML()
        self.roots, self.pins, self.boots, self.reservations = [], {}, {}, []
        self.start = self.deadline = None
        self.stage, self.next_sample = 'pre_gpu_gate', 0
        self.completed = self.model_launches = 0
        self.exit = load('mv_profile_frozen_exit', OWN.parent / 'prompt_kv_gpu_v4/exit_guard.py')

    def status(self, **extra):
        write(self.job / 'status.json', dict(stage=self.stage, at=stamp(), supervisor_pid=os.getpid(),
            GPU_elapsed_seconds=0 if self.start is None else time.monotonic()-self.start,
            model_launches=self.model_launches, formal_timing_samples=0, **extra))

    def reserve(self, out):
        assert not self.reservations
        out.mkdir(parents=True)
        r = dict(out=str(out), cap=self.p['fixed_target_cap'], kind='diagnostic-nine-complete-chunks',
                 ordinal=1, at=stamp(), plan_sha256=digest(self.job / 'plan.json'))
        write(out / 'FORWARD-RESERVATION.json', r, exclusive=True, durable=True)
        self.reservations.append(r)
        write(self.job / 'RESERVATIONS.json', self.reservations, exclusive=True, durable=True)

    def environment(self, arm, out):
        env = super().environment(arm, out)
        from experiment import ENABLED, APPROVED
        env.update({ENABLED: '1', APPROVED: '1', 'MV_PROFILE_GPU_APPROVED': '1',
                    'WAM_P0_ACTION_DRAFT_MODE': 'verifier_k2_b2_bonus',
                    'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'})
        return env

    def run(self):
        error = cleanup = None
        try:
            self.status()
            self.idle(initial=True)
            for port in self.p['ports']:
                with socket.socket() as s:
                    s.bind(('127.0.0.1', port))
            self.start = time.monotonic()
            self.deadline = self.start + self.p['GPU_hard_limit_seconds']
            write(self.job / 'GPU-ATTEMPT.json', dict(at=stamp(), attempt=1,
                monotonic_start=self.start, monotonic_deadline=self.deadline,
                plan_sha256=digest(self.job / 'plan.json')), exclusive=True, durable=True)
            out = self.job / 'fixed/worker0'
            self.reserve(out)
            env = self.environment('CURRENT', out)
            self.stage = 'single_model_nine_chunks_six_profiles'
            cmd = [VA, '-m', 'torch.distributed.run', '--nproc_per_node=1',
                '--master_port='+str(self.p['ports'][1]), '--local-ranks-filter=0', '--tee=3',
                str(OWN / 'profile_worker.py'), '--config-name', 'libero', '--port', str(self.p['ports'][0]),
                '--save_root', str(out / 'server_output')]
            proc = self.spawn(cmd, out / 'server.log', env, model=True)
            while proc.poll() is None:
                self.guard()
                time.sleep(1)
            if proc.returncode != 0:
                raise RuntimeError('diagnostic worker failed; preserved original log')
            result = json.loads((out / 'WORKER-RESULT.json').read_text())
            assert result['status'] == 'PASS_MEGAVERIFIER_PROFILE' and result['chunks'] == 9
            assert result['profiled_verifier_calls'] == 6
            self.stage = 'collection_complete'
        except BaseException as exc:
            error = dict(error=repr(exc), traceback=traceback.format_exc(), failed_stage=self.stage)
            write(self.job / 'STOP_REQUESTED', dict(at=stamp(), reason=repr(exc)))
            self.stage = 'failed'
        finally:
            try:
                from cleanup import roots
                closed = roots(self.roots, self.pins,
                    deadline=min(time.monotonic()+25, self.deadline or time.monotonic()+25), wait_seconds=1)
                write(self.job / 'CLEANUP.json', closed, exclusive=True, durable=True)
                cleanup = self.nv.sample()
                assert all(p.poll() is not None for p in self.roots)
            except BaseException as exc:
                error = error or dict(error=repr(exc), failed_stage='cleanup')
            finally:
                self.nv.close()
        seconds = 0 if self.start is None else time.monotonic()-self.start
        charges = old.charged_reservations(self.job, self.reservations)
        final = dict(status='PASS_MEGAVERIFIER_PROFILE_COLLECTION' if error is None else 'FAILED', at=stamp(),
            new_GPU_seconds=seconds, aggregate_GPU_seconds=self.p['aggregate_prior_GPU_seconds']+seconds,
            aggregate_GPU_seconds_cap=36000, aggregate_target_cap=self.p['aggregate_target_cap'],
            aggregate_target_intents=self.p['aggregate_prior_target_intents']+charges['charged_target_intents'],
            cleanup=cleanup, owned_roots_exited=all(p.poll() is not None for p in self.roots), **charges)
        if error:
            final.update(error)
        write(self.job / ('GPU-COLLECTION.json' if error is None else 'FAILED.json'), final, exclusive=True, durable=True)
        self.status(terminal_result=final['status'])
        print(json.dumps(final, ensure_ascii=False), flush=True)
        if error:
            raise RuntimeError(error['error'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    args = ap.parse_args()
    def stop(*_):
        old.STOP = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    runtime = ROOT / '投机采样/results/runtime'
    with (ROOT / '.pipeline/.chunk-template-gpu.lock').open('a') as a, (runtime / '.speculative_minimal_gpu.lock').open('a') as b, (runtime / '.shared-prefix-GPU.lock').open('a') as c:
        for lock in (a, b, c):
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        Campaign(args.job).run()


if __name__ == '__main__':
    main()
