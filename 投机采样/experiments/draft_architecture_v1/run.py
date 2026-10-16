"""Finite 18-run CPU architecture comparison on frozen, episode-split teacher data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
    raise RuntimeError('CUDA_VISIBLE_DEVICES must be empty before importing torch')
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
if os.sched_getaffinity(0) != {0} or torch.cuda.is_initialized():
    raise RuntimeError('CPU0-only, no CUDA initialization required')

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
sys.path.insert(0, str(OWN.parent / 'trace_preserving_train_v1'))
import diagnostics as d
from models import Predictor, SPECS, make_model, position_tokens

SEEDS = (20260922, 20260923, 20260924)
LRS = (1e-4, 1e-3)
ARCHITECTURES = tuple(SPECS)
UPDATES, BATCH = 1000, 512
LIMIT = 1800


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')


def tensor_sha(x):
    return hashlib.sha256(x.contiguous().numpy().tobytes()).hexdigest()


def normalize(train):
    features = torch.cat((train['state'], train['previous'], train['scalars']), 1)
    residual = train['target'] - train['previous']
    return {'feature_mean': features.mean(0), 'feature_std': features.std(0, correction=0).clamp_min(1e-5),
            'residual_mean': residual.mean(0), 'residual_std': residual.std(0, correction=0).clamp_min(1e-5)}


def source_inventory():
    paths = list(OWN.glob('*.py')) + [
        OWN.parent / 'trace_preserving_train_v1/diagnostics.py',
        OWN.parent / 'draft_teacher_train_v1/training.py',
        ROOT / '投机采样/scripts/p0_minimal_frozen_train.py',
        ROOT / '投机采样/scripts/p0_batched_verify.py',
        ROOT / '投机采样/configs/current_speculative_candidate.json',
        ROOT / '.pipeline/state.yaml', ROOT / '.pipeline/rules.yaml',
        ROOT / '.pipeline/config.yaml', ROOT / 'AGENTS.md',
    ]
    return {str(p): d.digest(p) for p in paths if p.exists()}


def self_check(normalization, data):
    checks = {}
    x = torch.arange(227.).reshape(1, 227)
    tokens = position_tokens(x)
    checks['sixteen_position_mapping'] = all(
        tokens[0, pos, channel].item() == channel * 16 + pos and
        tokens[0, pos, 7 + channel].item() == 112 + channel * 16 + pos
        for pos in range(16) for channel in range(7))
    checks['all_three_time_scalars_repeated'] = torch.equal(tokens[:, :, 14:], x[:, 224:].unsqueeze(1).expand(-1,16,-1))
    checks['normalizers_exact_train_only'] = all(torch.equal(v, normalize(data)[k]) for k, v in normalization.items())
    for name in ARCHITECTURES:
        torch.manual_seed(SEEDS[0])
        p = Predictor(name, normalization)
        state = data['state'][0].reshape(7,4,4).to(torch.bfloat16)
        previous = data['previous'][0].reshape(7,4,4).to(torch.bfloat16)
        before = (state.clone(), previous.clone(), torch.get_rng_state().clone())
        scalars = data['scalars'][0].tolist()
        out = p.predict(state, previous, scalars[0], scalars[1], scalars[2] * 1000.)
        checks[name + '_shape_dtype'] = out.shape == (2,7,4,4) and out.dtype == torch.bfloat16
        checks[name + '_input_and_rng_unchanged'] = (torch.equal(state,before[0]) and torch.equal(previous,before[1])
                                                   and torch.equal(torch.get_rng_state(),before[2]))
        b = {k: v[:4] for k,v in data.items()}
        pred = p.decode_batch(b)
        loss = d.source_training.loss_value(pred,b,p.residual_std,'DATA',1.)
        loss.backward()
        checks[name + '_finite_nonzero_gradient'] = (bool(torch.isfinite(loss)) and
            all(t.grad is None or bool(torch.isfinite(t.grad).all()) for t in p.model.parameters()) and
            any(t.grad is not None and bool((t.grad != 0).any()) for t in p.model.parameters()))
    if not all(checks.values()):
        raise RuntimeError('CPU contract checks failed: ' + str(checks))
    return checks


def prepare(job):
    started = time.monotonic()
    job.mkdir(parents=True, exist_ok=False)
    collection_plan, train_rows, val_rows, manifest, old = d.load_checked()
    train = d.prepared(train_rows)
    normalization = normalize(train)
    checks = self_check(normalization, train)
    counts = {name: sum(p.numel() for p in make_model(name).parameters()) for name in ARCHITECTURES}
    relative_gap = abs(counts['mlp96']-counts['transformer64'])/min(counts['mlp96'],counts['transformer64'])
    if counts['mlp51'] != 25928 or relative_gap >= .05:
        raise RuntimeError('capacity match failed')
    manifest['normalization_frozen'] = False
    manifest['normalization'] = 'recomputed only from 3906 training rows; common to all scratch architectures'
    write(job / 'dataset_manifest.json', manifest)
    cache = {'train_rows': train_rows, 'validation_rows': val_rows, 'normalization': normalization}
    torch.save(cache, job / 'dataset.pt')
    samples = {str(seed): tensor_sha(torch.randint(3906,(UPDATES,BATCH),generator=torch.Generator().manual_seed(seed))) for seed in SEEDS}
    plan = dict(schema='draft-architecture-plan/v1', at=stamp(), job=str(job),
        architectures=SPECS, parameters=counts, capacity_relative_gap=relative_gap,
        seeds=SEEDS, learning_rates=LRS, updates=UPDATES, batch_size=BATCH,
        optimizer='AdamW', weight_decay=.01, initialization='all three architectures from scratch',
        baseline='frozen deployed A_off; different pretraining history, reference only',
        baseline_path=collection_plan['frozen_A_off'], baseline_sha256=collection_plan['frozen_A_off_sha256'],
        loss='mean(min_head direction_MSE)+0.10*mean(direction_MSE)+0.01*mean(relu(0.05-normalized_head_gap))',
        train_rows=3906, validation_rows=1890, training_inits=[0,1], validation_inits=[8],
        no_test_read=True, action_shape=[7,4,4], threshold=d.THRESHOLD,
        normalization_source='train only, FP32, population std clamped to 1e-5',
        normalizer_sha256={k:tensor_sha(v) for k,v in normalization.items()}, sample_indices_sha256=samples,
        checkpoint='fixed final update 1000; no early stopping; retain every complete run',
        run_order=[dict(seed=s,lr=lr,architecture=a) for s in SEEDS for lr in LRS for a in ARCHITECTURES],
        selection=dict(per_seed=['root pass count >= A_off','each task root pass count >= A_off',
             'all mean successor L2 <= A_off','root mean successor L2 <= A_off','root old-pass loss rate <= 0.01'],
             per_configuration='all five componentwise median gates AND at least 2 of 3 seeds pass every gate',
             configuration_rank=['descending median root pass count','ascending median root mean L2',
                 'ascending median all mean L2','ascending parameters','architecture declaration order','ascending learning rate'],
             checkpoint='median by root pass count, then ascending root mean L2, then ascending seed; must itself pass; otherwise configuration not eligible',
             no_eligible='keep A_off; do not change gates or launch GPU'),
        cpu=dict(affinity=[0],torch_intraop=1,torch_interop=1,cuda_visible_devices='',nice=os.getpriority(os.PRIO_PROCESS,0)),
        total_execution_limit_seconds=LIMIT, preparation_elapsed_seconds=time.monotonic()-started,
        source_sha256=source_inventory(), dataset_manifest_sha256=d.digest(job/'dataset_manifest.json'),
        dataset_sha256=d.digest(job/'dataset.pt'), cpu_contract_checks=checks,
        limitation='Only stored OLD-query local predictions; no NEW recursive acceptance, closed-loop success, trajectory, or GPU timing claim.')
    write(job/'plan.json',plan)
    print(json.dumps(dict(status='PREPARED_NO_TRAINING',job=str(job),plan_sha256=d.digest(job/'plan.json'),parameters=counts,seconds=plan['preparation_elapsed_seconds'])),flush=True)


@torch.no_grad()
def predictions(predictor, rows):
    output = []
    for row in rows:
        state = row['state'].reshape(7,4,4).to(torch.bfloat16)
        previous = row['previous'].reshape(7,4,4).to(torch.bfloat16)
        value = predictor.predict(state,previous,row['sigma'],row['next_sigma'],row['timestep'])
        output.append(value.reshape(2,112).float())
    return torch.stack(output)


def evaluate(pred, rows, old_pred):
    data = d.prepared(rows)
    errors = (pred-data['target'][:,None]).square().sum(2).sqrt()*data['delta'][:,None]
    old_errors = (old_pred-data['target'][:,None]).square().sum(2).sqrt()*data['delta'][:,None]
    best, old_best = errors.min(1).values, old_errors.min(1).values
    ok, old_ok = best <= d.THRESHOLD, old_best <= d.THRESHOLD
    groups = d.groups(rows)
    groups['child'] = [i for i,r in enumerate(rows) if r['node'] != 0]
    result = {}
    for name,ix in groups.items():
        old_count = int(old_ok[ix].sum())
        lost = int((old_ok[ix] & ~ok[ix]).sum())
        result[name] = dict(rows=len(ix),pass_count=int(ok[ix].sum()),old_pass_count=old_count,
            local_gate_pass_rate=float(ok[ix].float().mean()),
            old_pass_to_fail_count=lost,old_pass_to_fail_rate=lost/old_count if old_count else 0.,
            old_fail_to_pass_count=int((~old_ok[ix] & ok[ix]).sum()),
            successor_l2_mean=float(best[ix].mean()),old_successor_l2_mean=float(old_best[ix].mean()),
            successor_l2_p95=float(torch.quantile(best[ix],.95)))
    return dict(groups=result, rounded_predictions_sha256=tensor_sha(pred),
                prediction_interface='per-row predict at recorded BF16 query; no batched proxy',
                limitation='OLD stored same-query local metric only; not new tree acceptance or task quality')


def gate(metrics):
    g = metrics['groups']
    values = dict(root_pass_count=g['root']['pass_count']-g['root']['old_pass_count'],
        task0_root_pass_count=g['task0/root']['pass_count']-g['task0/root']['old_pass_count'],
        task1_root_pass_count=g['task1/root']['pass_count']-g['task1/root']['old_pass_count'],
        all_l2_improvement=g['all']['old_successor_l2_mean']-g['all']['successor_l2_mean'],
        root_l2_improvement=g['root']['old_successor_l2_mean']-g['root']['successor_l2_mean'],
        root_retention_margin=.01-g['root']['old_pass_to_fail_rate'])
    return dict(values=values,checks={k:v >= 0 for k,v in values.items()},passed=all(v >= 0 for v in values.values()))


def summarize(plan, complete):
    configs = []
    for architecture in ARCHITECTURES:
        for lr in LRS:
            runs = [v for v in complete if v['architecture']==architecture and v['lr']==lr]
            if len(runs) != 3:
                configs.append(dict(architecture=architecture,lr=lr,complete_seeds=len(runs),eligible=False))
                continue
            margins = {key:statistics.median(r['gates']['values'][key] for r in runs) for key in runs[0]['gates']['values']}
            ordered = sorted(runs,key=lambda r:(r['validation']['groups']['root']['pass_count'],
                r['validation']['groups']['root']['successor_l2_mean'],r['seed']))
            median_seed = ordered[1]
            median_root = statistics.median(r['validation']['groups']['root']['pass_count'] for r in runs)
            median_root_l2 = statistics.median(r['validation']['groups']['root']['successor_l2_mean'] for r in runs)
            median_all_l2 = statistics.median(r['validation']['groups']['all']['successor_l2_mean'] for r in runs)
            eligible = all(v>=0 for v in margins.values()) and sum(r['gates']['passed'] for r in runs)>=2 and median_seed['gates']['passed']
            configs.append(dict(architecture=architecture,lr=lr,complete_seeds=3,median_gate_margins=margins,
                seeds_passing=sum(r['gates']['passed'] for r in runs),eligible=eligible,
                median_root_pass_count=median_root,median_root_l2=median_root_l2,median_all_l2=median_all_l2,
                checkpoint=median_seed['checkpoint'],seed=median_seed['seed'],
                rank=[-median_root,median_root_l2,median_all_l2,plan['parameters'][architecture],ARCHITECTURES.index(architecture),lr]))
    eligible = sorted([c for c in configs if c['eligible']],key=lambda c:c['rank'])
    return dict(configurations=configs, selected=eligible[0] if eligible else None,
                decision='ONE_LOCAL_CANDIDATE_FOR_LATER_GPU_CHECK_ONLY' if eligible else 'KEEP_FROZEN_A_OFF')


def train(job):
    started = time.monotonic()
    plan = json.loads((job/'plan.json').read_text())
    remaining = LIMIT-plan['preparation_elapsed_seconds']
    if remaining <= 0 or (job/'TRAINING-STARTED.json').exists():
        raise RuntimeError('expired or already attempted finite job')
    def timeout(signum,frame):
        raise TimeoutError('1800-second total execution budget reached')
    signal.signal(signal.SIGALRM,timeout)
    signal.setitimer(signal.ITIMER_REAL,remaining)
    write(job/'TRAINING-STARTED.json',dict(at=stamp(),plan_sha256=d.digest(job/'plan.json'),remaining_seconds=remaining,pid=os.getpid()))
    complete = []
    current = None
    try:
        if source_inventory() != plan['source_sha256'] or d.digest(job/'dataset.pt') != plan['dataset_sha256']:
            raise RuntimeError('frozen source or dataset changed')
        if d.digest(plan['baseline_path']) != plan['baseline_sha256']:
            raise RuntimeError('baseline changed')
        cache = torch.load(job/'dataset.pt',map_location='cpu',weights_only=False)
        train_rows,val_rows,normalization = cache['train_rows'],cache['validation_rows'],cache['normalization']
        data = d.prepared(train_rows)
        if any(tensor_sha(v) != plan['normalizer_sha256'][k] for k,v in normalize(data).items()):
            raise RuntimeError('train-only normalization changed')
        old = d.Predictor.from_checkpoint(plan['baseline_path'])
        old_pred = predictions(old,val_rows)
        write(job/'BASELINE.json',evaluate(old_pred,val_rows,old_pred))
        for spec in plan['run_order']:
            architecture,seed,lr = spec['architecture'],spec['seed'],spec['lr']
            name = f'{architecture}-lr{lr:g}-seed{seed}'
            folder = job/name
            folder.mkdir()
            current = dict(name=name,updates_completed=0,architecture=architecture,seed=seed,lr=lr)
            torch.manual_seed(seed)
            predictor = Predictor(architecture,normalization)
            predictor.model.train()
            indices = torch.randint(len(train_rows),(UPDATES,BATCH),generator=torch.Generator().manual_seed(seed))
            if tensor_sha(indices) != plan['sample_indices_sha256'][str(seed)]:
                raise RuntimeError('sample sequence mismatch')
            optimizer = torch.optim.AdamW(predictor.model.parameters(),lr=lr,weight_decay=.01)
            run_start = time.monotonic()
            logs = []
            for step,ix in enumerate(indices,1):
                batch = {k:v[ix] for k,v in data.items()}
                pred = predictor.decode_batch(batch)
                loss = d.source_training.loss_value(pred,batch,predictor.residual_std,'DATA',1.)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError('nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in predictor.model.parameters()):
                    raise RuntimeError('nonfinite gradient')
                optimizer.step()
                current['updates_completed'] = step
                if step == 1 or step % 100 == 0:
                    entry=dict(step=step,loss=float(loss.detach()),elapsed_seconds=time.monotonic()-run_start)
                    logs.append(entry)
                    print(json.dumps(dict(run=name,**entry)),flush=True)
            elapsed = time.monotonic()-run_start
            metadata = dict(**spec,updates=UPDATES,batch_size=BATCH,normalization='train only',initialization='scratch',
                plan_sha256=d.digest(job/'plan.json'),sample_indices_sha256=tensor_sha(indices))
            checkpoint=folder/'checkpoint.pt'
            predictor.model.eval()
            predictor.save(checkpoint,metadata)
            before_rng=torch.get_rng_state().clone()
            val_pred=predictions(predictor,val_rows)
            if not torch.equal(before_rng,torch.get_rng_state()):
                raise RuntimeError('prediction changed RNG')
            reload_model=Predictor.load(checkpoint)
            reloaded=predictions(reload_model,val_rows)
            if not torch.equal(val_pred,reloaded):
                raise RuntimeError('complete validation predict reload is not exact')
            metrics=evaluate(val_pred,val_rows,old_pred)
            result=dict(**spec,updates=UPDATES,training_elapsed_seconds=elapsed,checkpoint=str(checkpoint),
                checkpoint_sha256=d.digest(checkpoint),loss_log=logs,validation=metrics,gates=gate(metrics),
                reload_all_1890_predictions_exact=True,prediction_rng_unchanged=True,finite_gradients_every_update=True)
            write(folder/'RESULT.json',result)
            complete.append(result)
            print(json.dumps(dict(status='RUN_COMPLETE',run=name,gate=result['gates']['passed'],root=metrics['groups']['root'],completed=len(complete))),flush=True)
            current=None
        if torch.cuda.is_initialized() or source_inventory()!=plan['source_sha256']:
            raise RuntimeError('CPU or frozen source boundary failed')
        summary=summarize(plan,complete)
        result=dict(status='PASS_CPU_18_RUN_ARCHITECTURE_SCREEN',at=stamp(),complete_runs=len(complete),
            total_execution_seconds=time.monotonic()-started+plan['preparation_elapsed_seconds'],
            summary=summary,CUDA_initialized=False,GPU_executed=False,source_unchanged=True,
            source_teacher_manifest=str(job/'dataset_manifest.json'),plan_sha256=d.digest(job/'plan.json'),
            limitation=plan['limitation'])
        write(job/'FINAL-RESULT.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
    except BaseException as exc:
        partial=dict(status='PARTIAL_CPU_SCREEN' if isinstance(exc,TimeoutError) else 'FAILED_CPU_SCREEN',at=stamp(),
            error_type=type(exc).__name__,error=str(exc),complete_runs=len(complete),current_run=current,
            total_execution_seconds=time.monotonic()-started+plan['preparation_elapsed_seconds'],
            complete_run_checkpoints=[r['checkpoint'] for r in complete],
            decision='INCOMPLETE_DESIGN_NO_SELECTION_KEEP_A_OFF',CUDA_initialized=torch.cuda.is_initialized())
        write(job/'PARTIAL-OR-FAILURE.json',partial)
        print(json.dumps(partial,ensure_ascii=False),flush=True)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['prepare','train'])
    parser.add_argument('--job',type=Path,required=True)
    args=parser.parse_args()
    if not str(args.job.resolve()).startswith('/var/tmp/cxz-wam-quality-20261014/p0-draft-architecture-v1-'):
        raise ValueError('isolated authorized output prefix required')
    (prepare if args.mode=='prepare' else train)(args.job.resolve())
