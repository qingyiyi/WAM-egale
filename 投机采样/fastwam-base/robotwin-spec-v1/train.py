"""Prepared RoboTwin trainer; run only after its own canonical/panel authorization.

Same fixed final3000/FIT-only-normalization/CALQ80 recipe, 14 joint channels.
Every panel row supplies a complete episode identifier; task IDs need not be LIBERO IDs.
"""
import argparse
import hashlib
import json
import os
import signal
import time
from pathlib import Path
import torch
from draft import DirectionPredictor

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT/'training'
OUT.mkdir(exist_ok=False)
reg = json.loads((ROOT/'REGISTRATION.json').read_text())
panel = json.loads((ROOT/'PANEL.json').read_text())
start = time.time()
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('training cap')))
signal.alarm(max(1, int(min(reg['deadline_epoch']-300-start, 3600))))
torch.set_num_threads(2)
torch.manual_seed(panel['training']['seed'])
device = 'cuda'


def write(name, data):
    p = OUT/name
    q = p.with_suffix('.tmp')
    q.write_text(json.dumps(data, indent=2)+'\n')
    q.replace(p)


def read_split(split):
    columns = {k: [] for k in ['state', 'previous', 'target', 'scalars', 'delta', 'step']}
    groups, manifest = [], []
    actual_episodes = set()
    cursor = 0
    for row in panel[split]:
        ep = ROOT/'results/episodes'/row['episode']
        assert ep.parent == ROOT/'results/episodes' and ep.name == row['episode']
        terminal = json.loads((ep/'EPISODE.json').read_text())
        assert terminal.get('complete') and terminal['identity']['split'] == split, ep
        assert terminal['identity']['group'] == row['group'], ep
        # Expert filtering can advance a requested seed. Different requested
        # starts/group names must not conceal the same accepted physical scene.
        actual_key = (row['task'], int(terminal['scene_seed']), terminal['identity']['task_config'])
        assert actual_key not in actual_episodes, ('duplicate accepted scene', split, actual_key)
        actual_episodes.add(actual_key)
        files = sorted(ep.glob('chunk-*.pt'))
        assert len(files) == terminal['chunks'] > 0, ep
        group_start = cursor
        for file in files:
            data = torch.load(file, map_location='cpu', weights_only=True)
            c = data['columns']
            assert data['schema'] == 'fastwam-robotwin-canonical-teacher/v1'
            assert c['state'].shape == c['target'].shape == c['previous'].shape == (10, 32, 14)
            assert torch.equal(c['has_previous'], torch.arange(10)>0)
            sigma = c['timestep']/1000.
            for k in ['state', 'previous', 'target']:
                assert torch.isfinite(c[k]).all()
                columns[k].append(c[k][1:])
            columns['scalars'].append(torch.stack((sigma, sigma+c['delta'], sigma), 1)[1:])
            columns['delta'].append(c['delta'][1:])
            columns['step'].append(torch.arange(1, 10))
            cursor += 9
        groups.append(dict(task=row['task'], group=row['group'], start=group_start, end=cursor))
        manifest.append(dict(episode=ep.name, chunks=len(files), queries=cursor-group_start,
                             complete_status=terminal['status'], success=terminal['success'],
                             task=row['task'], actual_scene_seed=int(terminal['scene_seed']),
                             task_config=terminal['identity']['task_config']))
    return {k: torch.cat(v).to(device) for k, v in columns.items()}, groups, manifest


try:
    assert panel['canonical_review_passed'] and panel['action_dim'] == 14
    assert panel['checkpoint_revision'] == '8eaceeb24c3cc92ff2a9c9a9d266a4941b836705'
    fit_ids = {r['group'] for r in panel['FIT']}
    cal_ids = {r['group'] for r in panel['CAL']}
    assert fit_ids and cal_ids and fit_ids.isdisjoint(cal_ids)
    assert len(fit_ids) == len(panel['FIT']) and len(cal_ids) == len(panel['CAL'])
    seen_groups = fit_ids | cal_ids
    for split in ('DEV', 'TEST'):
        other_ids = {r['group'] for r in panel.get(split, [])}
        assert len(other_ids) == len(panel.get(split, [])), ('duplicate complete group', split)
        assert other_ids.isdisjoint(seen_groups), split
        seen_groups |= other_ids
    assert panel['training']['updates'] == 3000 and panel['training']['batch'] == 256
    assert panel['training']['weight_decay'] == .01
    fit, groups, fit_manifest = read_split('FIT')
    cal, _, cal_manifest = read_split('CAL')
    fit_scenes = {(r['task'], r['actual_scene_seed']) for r in fit_manifest}
    cal_scenes = {(r['task'], r['actual_scene_seed']) for r in cal_manifest}
    assert fit_scenes.isdisjoint(cal_scenes), 'accepted scene leakage, even across clean/randomized conditions'
    write('DATA-MANIFEST.json', dict(FIT=fit_manifest, CAL=cal_manifest,
          exposure='canonical rollout states only; no draft rollout augmentation; DEV never read for fitting/selection'))
    model = DirectionPredictor().to(device)
    features = torch.cat((fit['state'].flatten(1), fit['previous'].flatten(1), fit['scalars']), 1)
    residuals = (fit['target']-fit['previous']).flatten(1)
    with torch.no_grad():
        model.feature_mean.copy_(features.mean(0))
        model.feature_std.copy_(features.std(0).clamp_min(1e-3))
        model.residual_mean.copy_(residuals.mean(0))
        model.residual_std.copy_(residuals.std(0).clamp_min(1e-3))
    del features, residuals
    optimizer = torch.optim.AdamW(model.parameters(), lr=panel['training']['lr'], weight_decay=.01)
    gen = torch.Generator().manual_seed(panel['training']['seed'])
    task_ids = sorted({g['task'] for g in groups})
    task_groups = {t: [g for g in groups if g['task']==t] for t in task_ids}
    history = []
    model.train()
    for step in range(1, 3001):
        # Task -> complete episode -> query, with no preference for successful episodes.
        tasks = [task_ids[i] for i in torch.randint(len(task_ids), (256,), generator=gen).tolist()]
        indices = []
        for t in tasks:
            g = task_groups[t][int(torch.randint(len(task_groups[t]), (), generator=gen))]
            indices.append(int(torch.randint(g['start'], g['end'], (), generator=gen)))
        idx = torch.tensor(indices, device=device)
        pred = model.decode(fit['state'][idx], fit['previous'][idx], fit['scalars'][idx])
        std = model.residual_std.reshape(1, 1, 32, 14)
        errors = (pred-fit['target'][idx, None]).square().mean((2, 3))
        gap = ((pred[:, 0]-pred[:, 1])/std[:, 0]).square().mean((1, 2))
        loss = errors.min(1).values.mean()+.1*errors.mean()+.01*torch.relu(.05-gap).mean()
        assert torch.isfinite(loss), step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
        if step == 1 or step % 100 == 0:
            history.append(dict(step=step, loss=float(loss.detach()), elapsed=time.time()-start))
            write('STATUS.json', dict(status='TRAINING', step=step, pid=os.getpid(), history=history))
        if step in (1000, 2000, 3000):
            torch.save(dict(schema='fastwam-robotwin-direction-v1', step=step, action_dim=14, action_horizon=32,
                       state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                       metadata=panel['training']), OUT/f'checkpoint-{step}.pt')
    model.eval().requires_grad_(False)
    scores, steps = [], []
    with torch.no_grad():
        for i in range(0, len(cal['state']), 256):
            sl = slice(i, i+256)
            pred = model.decode(cal['state'][sl], cal['previous'][sl], cal['scalars'][sl]).to(torch.bfloat16).float()
            error = (pred-cal['target'][sl, None]).flatten(2).norm(dim=2)*cal['delta'][sl, None].abs()
            keep = cal['step'][sl] <= 8
            scores.append(error.min(1).values[keep].cpu())
            steps.append(cal['step'][sl][keep].cpu())
    scores, steps = torch.cat(scores), torch.cat(steps)
    tau = float(torch.quantile(scores, .8))
    assert tau >= 0 and torch.isfinite(scores).all()
    torch.save(dict(score=scores, step=steps), OUT/'CAL-SCORES.pt')
    sha = hashlib.sha256((OUT/'checkpoint-3000.pt').read_bytes()).hexdigest()
    write('SELECTED.json', dict(schema='fastwam-robotwin-selected-v1', checkpoint='checkpoint-3000.pt',
          checkpoint_sha256=sha, selection='fixed final 3000 only', tau=tau, CAL_quantile=.8,
          CAL_queries=len(scores), CAL_empirical_min_head_pass=float((scores<=tau).float().mean()),
          parameters=sum(p.numel() for p in model.parameters()), FIT_queries=len(fit['state']),
          dtype='FP32 predictor, BF16 deployed proposals, FP32 abs(actual BF16 delta)*L2 scoring',
          quality_claim=False, independent_TEST=False, end=time.time(), seconds=time.time()-start))
    write('STATUS.json', dict(status='COMPLETE', pid=os.getpid(), tau=tau, end=time.time(), history=history))
except BaseException as exc:
    import traceback
    write('FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc(), time=time.time()))
    raise
finally:
    signal.alarm(0)
