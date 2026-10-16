"""Build the complete Long10 T5 bank using original helpers, CPU only.

This produces reusable inference inputs and times text preparation. It does
not run LIBERO, load the target transformer, change a server, or benchmark GPU.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / '投机采样/wam-p0-worktree/wan_va/wan_va_server.py'
UTILS = SOURCE.parent / 'modules/utils.py'
OLD = ROOT / '投机采样/results/runtime/p0-libero-libero_10-continuation-targetonly-transformer-output-par2-20261004-1110/client/episodes.jsonl'
PLAN = Path('/var/tmp/cxz-formal-e2e-speed/p0-formal-e2e-speed-v1-multi-r3-20261015/plan.json')


def write(path, value):
    tmp = path.with_suffix(path.suffix + '.pending')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def functions(path, names, namespace, cls=None):
    tree = ast.parse(path.read_text())
    body = tree.body if cls is None else next(n for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == cls).body
    selected = [n for n in body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('original helper names differ')
    for node in selected:
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return {name: namespace[name] for name in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=7)
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('explicit CUDA_VISIBLE_DEVICES empty required')
    if not 1 <= args.threads <= 8 or not 60 <= args.max_seconds <= 1800:
        raise ValueError('finite CPU bounds required')
    if args.job.exists():
        raise ValueError('fresh job required; previous evidence never overwritten')
    args.job.mkdir(parents=True)
    started = time.monotonic()
    rows = []
    current = 'imports'

    def status(state='RUNNING', **extra):
        write(args.job/'status.json', dict(status=state, active=current,
            completed=len(rows), planned=11, elapsed_sec=time.monotonic()-started,
            threads=args.threads, GPU_used=False, new_LIBERO_episodes=0, **extra))

    def timeout(*_):
        raise TimeoutError('finite CPU text-bank deadline reached')

    signal.signal(signal.SIGALRM, timeout)
    signal.alarm(args.max_seconds)
    try:
        os.nice(10)
        status()
        import torch
        from transformers import UMT5EncoderModel, T5TokenizerFast
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        assert not torch.cuda.is_initialized()
        plan = json.loads(PLAN.read_text())
        checkpoint = Path(plan['target_checkpoint_path'])
        registry = json.loads(Path(plan['registry']).read_text())['suites']['libero_10']
        if (Path(registry['local_path']).resolve() != checkpoint.resolve()
                or registry['official_checkpoint'] != plan['target_checkpoint_id']):
            raise ValueError('Long checkpoint registry mismatch')
        current = 'public_instruction_manifest'
        status()
        prompts = {}
        excluded = []
        for line in OLD.read_text().splitlines():
            row = json.loads(line)
            if row.get('status') == 'error':
                excluded.append(dict(task=row.get('task_index'), init=row.get('episode_index')))
                continue
            if row.get('status') != 'completed' or not isinstance(row.get('prompt'),str):
                raise ValueError('completed public instruction metadata incomplete')
            task = row['task_index']
            if task in prompts and prompts[task] != row['prompt']:
                raise ValueError('task prompt not unique')
            prompts[task] = row['prompt']
        if set(prompts) != set(range(10)) or len(set(prompts.values())) != 10:
            raise ValueError('complete ten-task public instruction bank required')
        write(args.job/'PROMPTS.json',dict(prompts=prompts,excluded_error_metadata=excluded,
            source=str(OLD),uses_public_task_instructions_only=True))
        ns = dict(torch=torch, UMT5EncoderModel=UMT5EncoderModel,
                  T5TokenizerFast=T5TokenizerFast, prompt_clean=prompt_clean)
        helpers = functions(UTILS, ['load_text_encoder', 'load_tokenizer'], ns)
        methods = functions(SOURCE, ['_get_t5_prompt_embeds', 'encode_prompt'], ns, cls='VA_Server')
        timers = []

        class Encoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.inner = model

            def forward(self, *call_args, **kwargs):
                begin = time.monotonic()
                result = self.inner(*call_args, **kwargs)
                timers.append(dict(kind='T5_forward', sec=time.monotonic()-begin))
                return result

        class Tokenizer:
            def __init__(self, tokenizer):
                self.inner = tokenizer

            def __call__(self, *call_args, **kwargs):
                begin = time.monotonic()
                result = self.inner(*call_args, **kwargs)
                timers.append(dict(kind='tokenize', sec=time.monotonic()-begin,
                                   valid_tokens=int(result.attention_mask.sum())))
                return result

        current = 'load_CPU_T5_and_tokenizer'
        status()
        begin = time.monotonic()
        encoder = helpers['load_text_encoder'](str(checkpoint/'text_encoder'), torch.bfloat16, 'cpu')
        encoder.eval().requires_grad_(False)
        tokenizer = helpers['load_tokenizer'](str(checkpoint/'tokenizer'))
        load_sec = time.monotonic()-begin
        proxy_cls = type('OriginalTextHelpers', (), methods)
        proxy = proxy_cls()
        proxy.device, proxy.dtype = torch.device('cpu'), torch.bfloat16
        proxy.text_encoder, proxy.tokenizer = Encoder(encoder), Tokenizer(tokenizer)
        embeddings = {}
        cases = [('empty_negative', '')] + [(f'task{t}', prompts[t]) for t in range(10)]
        with torch.inference_mode():
            for key, prompt in cases:
                current = key
                status()
                timers.clear()
                before_rng = torch.random.get_rng_state().clone()
                begin = time.monotonic()
                emb = proxy._get_t5_prompt_embeds(prompt, num_videos_per_prompt=1,
                    max_sequence_length=512, device=torch.device('cpu'), dtype=torch.bfloat16)
                elapsed = time.monotonic()-begin
                if (emb.shape != (1,512,4096) or emb.dtype != torch.bfloat16
                        or not bool(torch.isfinite(emb).all())
                        or not torch.equal(before_rng, torch.random.get_rng_state())):
                    raise ValueError('production embedding shape/dtype/finite/RNG check failed')
                embeddings[prompt] = emb.detach().clone()
                rows.append(dict(key=key, prompt=prompt, elapsed_sec=elapsed,
                    phases=list(timers), layout_and_other_sec=elapsed-sum(t['sec'] for t in timers)))
                write(args.job/'ENCODING-TIMES.json', rows)
                status()
        payload = dict(schema='long10-original-T5-bank/v1',
            target_checkpoint_id=plan['target_checkpoint_id'], checkpoint_path=str(checkpoint),
            sequence_length=512, dtype='bfloat16', output_shape=[1,512,4096],
            threads=args.threads, torch_version=str(torch.__version__),
            server_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            public_task_prompts={str(t):prompts[t] for t in range(10)}, embeddings=embeddings)
        current = 'save_and_verify_complete_bank'
        tmp = args.job/'bank.pending.pt'
        torch.save(payload, tmp)
        tmp.replace(args.job/'bank.pt')
        restored = torch.load(args.job/'bank.pt', map_location='cpu', weights_only=True)
        if set(restored['embeddings']) != set(embeddings) or any(
                not torch.equal(embeddings[k],restored['embeddings'][k]) for k in embeddings):
            raise ValueError('saved bank roundtrip is not exact')
        assert not torch.cuda.is_initialized()
        current = None
        result = dict(status='PASS_COMPLETE_CPU_TEXT_BANK', elapsed_sec=time.monotonic()-started,
            CPU_load_sec=load_sec, encoding_sec=sum(r['elapsed_sec'] for r in rows),
            T5_forward_sec=sum(t['sec'] for r in rows for t in r['phases'] if t['kind']=='T5_forward'),
            prompts=11, tasks=10, bank_bytes=(args.job/'bank.pt').stat().st_size,
            CUDA_initialized=False, target_transformer_loaded=False, LIBERO_episodes=0,
            records=rows, caveat='CPU text preparation only. No online reset, GPU transfer parity, closed-loop equivalence or E2E speedup is established. Do not inject into a running frozen job.')
        write(args.job/'RESULT.json',result)
        status('COMPLETE')
        print(json.dumps({k:v for k,v in result.items() if k!='records'},ensure_ascii=False),flush=True)
    except BaseException as exc:
        write(args.job/'FAILED.json',dict(error=repr(exc),active=current,
            elapsed_sec=time.monotonic()-started,completed=len(rows),traceback=traceback.format_exc()))
        status('FAILED',error=repr(exc))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
