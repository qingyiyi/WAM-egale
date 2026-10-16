"""CPU audit of every saved action, video latent and observation in four pairs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from protocol import load, write


def equal(a, b, context):
    if isinstance(a, torch.Tensor):
        if (not isinstance(b, torch.Tensor) or a.shape != b.shape or a.dtype != b.dtype
                or not torch.equal(a, b)):
            raise ValueError('saved tensor differs: ' + context)
        return 1
    if isinstance(a, np.ndarray):
        if (not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype
                or not np.array_equal(a, b)):
            raise ValueError('saved array differs: ' + context)
        return 1
    if isinstance(a, dict):
        if not isinstance(b, dict) or set(a) != set(b):
            raise ValueError('saved dictionary differs: ' + context)
        return sum(equal(a[k], b[k], context + '/' + str(k)) for k in a)
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError('saved sequence differs: ' + context)
        return sum(equal(x, y, context + '/' + str(i)) for i, (x, y) in enumerate(zip(a, b)))
    if type(a) is not type(b) or a != b:
        raise ValueError('saved scalar differs: ' + context)
    return 0


def inventory(directory):
    files = list(directory.rglob('*.pt'))
    result = {}
    for path in files:
        if path.name in result or not path.name.startswith(('actions_', 'latents_', 'obs_data_')):
            raise ValueError('unexpected/duplicate original saved output: ' + str(path))
        result[path.name] = path
    if not result:
        raise ValueError('complete original output is absent')
    return result


def main(job):
    if os.getenv('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('complete artifact audit is CPU-only')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    job = Path(job).resolve()
    p = load(job)
    complete = json.loads((job / 'RESULT.json').read_text())
    if complete['status'] != 'PASS_COMPLETE_RESET_BANK_EQUIVALENCE' or complete['full_episodes'] != 8:
        raise ValueError('all eight complete trajectories must finish first')
    rng = torch.get_rng_state().clone()
    rows, errors = [], []
    for arm in p['arms']:
        for task in (0, 1):
            a = next(r for r in p['cases'] if r['arm'] == arm and r['task'] == task and not r['text_bank'])
            b = next(r for r in p['cases'] if r['arm'] == arm and r['task'] == task and r['text_bank'])
            ia, ib = (inventory(job / 'episodes' / r['key'] / 'server_output') for r in (a, b))
            count = 0
            names = sorted(set(ia) | set(ib))
            if set(ia) != set(ib):
                errors.append(dict(arm=arm, task=task, reason='saved output inventory differs'))
            else:
                for name in names:
                    try:
                        # Original observation files are ordinary NumPy/pickle
                        # records produced by our own completed server.
                        x = torch.load(ia[name], map_location='cpu', weights_only=False)
                        y = torch.load(ib[name], map_location='cpu', weights_only=False)
                        count += equal(x, y, arm + '/' + str(task) + '/' + name)
                    except Exception as exc:
                        errors.append(dict(arm=arm, task=task, file=name, reason=str(exc)))
            rows.append(dict(arm=arm, task=task, init=29, files=len(names),
                action_files=sum(n.startswith('actions_') for n in names),
                video_latent_files=sum(n.startswith('latents_') for n in names),
                observation_files=sum(n.startswith('obs_data_') for n in names),
                tensor_array_values_checked=count))
    rng_same = torch.equal(rng, torch.get_rng_state())
    if not rng_same or torch.cuda.is_initialized():
        errors.append(dict(reason='CPU audit changed RNG or initialized CUDA'))
    report = dict(status='PASS_ALL_SAVED_EPISODE_TENSORS' if not errors else 'FAIL_SAVED_EPISODE_TENSORS',
        pairs=len(rows), full_episodes=8, errors=len(errors), differences=errors, rows=rows,
        files_compared=sum(r['files'] for r in rows),
        tensor_arrays_compared=sum(r['tensor_array_values_checked'] for r in rows),
        CPU_RNG_unchanged=rng_same, CUDA_initialized=torch.cuda.is_initialized(),
        elapsed_sec=time.monotonic() - started,
        scope='All complete saved normalized action endpoints, video latents, raw observations; includes unused final-chunk action tail. Value/dtype/shape equality, not zip-file bytes.')
    write(job / 'SAVED-TENSOR-PAIR-AUDIT.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise ValueError('complete saved output audit failed')


if __name__ == '__main__':
    import sys
    main(Path(sys.argv[1]))
