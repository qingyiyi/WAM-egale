"""Prepare/run bounded full-task reset equivalence, preserving other users' jobs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

from protocol import CLIENT, EXP, OWN, VA, WORKTREE, load, prepare, read, write

sys.path.insert(0, str(EXP))
sys.path.insert(0, str(EXP / 'formal_e2e_speed_v1'))
from formal_e2e_speed_v1.run import Campaign as ProvenLauncher, cleanup_roots


class Campaign(ProvenLauncher):
    def environment(self, row, out):
        env = super().environment(dict(arm=row['arm'], key=row['key']), out)
        for key in list(env):
            if key.startswith('FORMAL_SPEED_'):
                env.pop(key)
        env.update(RESET_ONLINE_JOB=str(self.job), RESET_ONLINE_ARM=row['arm'],
                   RESET_ONLINE_CASE=row['key'], RESET_TEXT_BANK_ENABLED='1',
                   HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   HF_DATASETS_OFFLINE='1')
        return env

    def telemetry(self, stage):
        sample = self.nv.sample()
        row = dict(monotonic=time.monotonic(), stage=stage, nvml=sample,
                   shared_GPU_allowed=True, performance_claim=False)
        with (self.job / 'GPU-TELEMETRY.jsonl').open('a') as stream:
            import json
            stream.write(json.dumps(row) + '\n')
        return row

    def free_GiB(self):
        run = subprocess.run(['nvidia-smi', '-i', '0',
            '--query-gpu=name,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=15, check=True)
        name, free = run.stdout.strip().rsplit(',', 1)
        if 'A100-SXM4-40GB' not in name:
            raise ValueError('this job is local A100 algorithm verification')
        return float(free) / 1024

    def guard(self, stage):
        self.stage = stage
        if time.monotonic() > self.deadline:
            raise TimeoutError('bounded complete-task wall deadline')
        if (self.job / 'STOP_REQUESTED').exists():
            raise RuntimeError('STOP_REQUESTED')
        if shutil.disk_usage(self.job).free < self.plan['disk_floor_GiB'] * 2**30:
            raise RuntimeError('project disk floor')
        if time.monotonic() >= self.next_sample:
            self.telemetry(stage)
            if self.free_GiB() < self.plan['runtime_free_GiB']:
                raise RuntimeError('runtime GPU headroom floor')
            self.next_sample = time.monotonic() + 15
            write(self.job / 'status.json', dict(stage=stage, completed=len(self.completed),
                planned=len(self.plan['cases']), active=self.current,
                performance_claim=False), replace=True)

    def admission(self):
        while True:
            self.guard('waiting_memory')
            if self.free_GiB() >= self.plan['initial_free_GiB']:
                return
            time.sleep(5)

    def ready(self, server, arm):
        limit = min(time.monotonic() + 900, self.deadline)
        while True:
            self.guard('model_loading_' + arm)
            if server.poll() is not None:
                raise RuntimeError('model exited before READY; see arm server log')
            if (self.job / f'{arm}-READY.json').is_file():
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{self.plan["port"]}/healthz', timeout=1) as r:
                        if r.status == 200:
                            return
                except (OSError, TimeoutError):
                    pass
            if time.monotonic() > limit:
                raise TimeoutError('model readiness')
            time.sleep(1)

    def arm(self, arm):
        rows = [r for r in self.plan['cases'] if r['arm'] == arm]
        server = client = None
        root = self.job / 'arms' / arm
        root.mkdir(parents=True, exist_ok=False)
        self.admission()
        env = self.environment(rows[0], root)
        command = [str(VA), '-m', 'torch.distributed.run', '--nproc_per_node=1',
                   '--master_port=' + str(self.plan['master_port']), '--local-ranks-filter=0',
                   '--tee=3', str(OWN / 'server_entry.py'), '--config-name', 'libero',
                   '--port', str(self.plan['port']), '--save_root', str(root / 'server_output')]
        try:
            server = self.spawn(command, root / 'server.log', env)
            self.ready(server, arm)
            for row in rows:
                self.current = row['key']
                out = self.job / 'episodes' / row['key']
                out.mkdir(parents=True, exist_ok=False)
                cenv = self.environment(row, out)
                cenv.update({key: str(self.plan['client_threads']) for key in
                             ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
                cmd = [str(CLIENT), str(OWN / 'client_entry.py'), '--port', str(self.plan['port']),
                    '--master-port', str(self.plan['master_port']), '--benchmarks', 'libero_10',
                    '--task-indices', str(row['task']), '--episodes-per-task', '1',
                    '--episode-start', str(row['init']), '--condition', row['key'],
                    '--campaign-id', self.job.name, '--worker-id', row['key'], '--trace-mode', 'off',
                    '--latency-benchmark', '--out-dir', str(out / 'client'), '--fail-fast',
                    '--no-resume', '--worktree', str(WORKTREE)]
                client = self.spawn(cmd, out / 'client.log', cenv)
                limit = min(time.monotonic() + self.plan['per_episode_wall_sec'], self.deadline)
                while client.poll() is None:
                    self.guard('complete_episode')
                    if server.poll() is not None or time.monotonic() > limit:
                        raise RuntimeError('model exit or complete episode deadline')
                    time.sleep(1)
                if client.returncode:
                    raise RuntimeError('complete LIBERO client failed; see client log')
                (out / 'FINISH_REQUESTED').write_text('Complete client; drain/snapshots outside task timer\n')
                limit = min(time.monotonic() + 180, self.deadline)
                while not (out / 'FINISH_READY.json').exists():
                    self.guard('post_episode_state')
                    if ((self.job / f'{arm}-MONITOR-FAILED.json').exists()
                            or server.poll() is not None or time.monotonic() > limit):
                        raise RuntimeError('persistent model state accounting failed')
                    time.sleep(.5)
                self.completed.append(row['key'])
                cleanup = cleanup_roots([client], self.pins, deadline=time.monotonic() + 15, wait_seconds=.5)
                write(out / 'CLIENT-CLEANUP.json', cleanup)
                client = None
            self.current = None
        finally:
            processes = [p for p in (client, server) if p is not None]
            if processes:
                cleanup = cleanup_roots(processes, self.pins, deadline=time.monotonic() + 30, wait_seconds=.5)
                write(root / 'OWNED-CLEANUP.json', cleanup)

    def run(self):
        try:
            for arm in self.plan['arms']:
                self.arm(arm)
            write(self.job / 'COLLECTION.json', dict(status='COMPLETE',
                completed=self.completed, elapsed_sec=time.monotonic() - self.started,
                performance_claim=False, all_own_processes_cleaned=True))
            from reset_online_v1.analyze import main as analyze
            analyze(self.job)
        except BaseException as exc:
            write(self.job / 'FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc(),
                  completed=self.completed, elapsed_sec=time.monotonic() - self.started))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        p = prepare(args.job)
        print({'job': str(args.job), 'episodes': len(p['cases']), 'GPU_used': False})
        return
    p = load(args.job)
    for port in (p['port'], p['master_port']):
        import socket
        with socket.socket() as s:
            s.bind(('127.0.0.1', port))
    Campaign(args.job.resolve(), p).run()


if __name__ == '__main__':
    main()
