"""CPU-only build parent; records the real compiler exit and owned identities."""
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

BASE = Path('/root/cxz/ops/whole-layer-warp-20261020')
SOURCE = Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')


def write(name, value):
    p = BASE / name
    p.write_text(json.dumps(value, indent=2) + '\n')


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, birth_ticks=int(stat[19]))


def main():
    os.sched_setaffinity(0, {4, 5})
    BASE.mkdir(parents=True, exist_ok=True)
    for name in ('tmp', 'cache', 'build'):
        (BASE / name).mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='',
        CUDA_HOME='/root/cxz/ops/persistent-tile-20261019/cuda-12.4',
        TORCH_CUDA_ARCH_LIST='8.0', MAX_JOBS='2', OMP_NUM_THREADS='2',
        TMPDIR=str(BASE/'tmp'), XDG_CACHE_HOME=str(BASE/'cache'),
        TORCH_EXTENSIONS_DIR=str(BASE/'cache/torch'))
    command = ['/root/cxz/envs/va/bin/python', str(SOURCE/'build_whole_layer.py'),
        'build_ext', '--build-lib', str(SOURCE), '--build-temp', str(BASE/'build')]
    start = time.monotonic()
    attempt = 1
    while (BASE/f'BUILD-{attempt}-RESULT.json').exists():
        attempt += 1
    with (BASE/f'BUILD-{attempt}.log').open('wb') as log:
        child = subprocess.Popen(command, cwd=SOURCE, env=env, stdout=log, stderr=log)
        write(f'BUILD-{attempt}-OWNED.json', dict(parent=identity(os.getpid()),
            compiler=identity(child.pid), argv=command, CPU=[4, 5], GPU_allowed=False,
            at=datetime.now(timezone.utc).isoformat()))
        try:
            code = child.wait(timeout=1800)
            status = 'COMPILED' if code == 0 else 'COMPILER_FAILED'
        except subprocess.TimeoutExpired:
            child.terminate()
            try:
                code = child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                child.kill()
                code = child.wait(timeout=30)
            status = 'BUILD_TIMEOUT'
    write(f'BUILD-{attempt}-RESULT.json', dict(status=status, returncode=code,
        lifecycle_seconds=time.monotonic()-start, compiler_reaped=True,
        at=datetime.now(timezone.utc).isoformat(), model_forwards=0))


if __name__ == '__main__':
    main()
