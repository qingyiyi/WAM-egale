"""Evaluate removal of masked T5 padding on all eleven real bank inputs.

This does not enable the change. Any numerical difference is preserved as a
failure of bitwise parity, even if the maximum error is small.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import time
import traceback

from build_cpu import functions, write, SOURCE, UTILS, PLAN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bank-job',type=Path,required=True)
    ap.add_argument('--job',type=Path,required=True)
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or args.job.exists():
        raise ValueError('CUDA invisible and fresh output required')
    result = json.loads((args.bank_job/'RESULT.json').read_text())
    if result['status'] != 'PASS_COMPLETE_CPU_TEXT_BANK':
        raise ValueError('all production reference embeddings required')
    args.job.mkdir(parents=True)
    start=time.monotonic()
    rows=[]
    def deadline(*_):
        raise TimeoutError('finite 300-second padding comparison budget reached')
    signal.signal(signal.SIGALRM,deadline)
    signal.alarm(300)
    try:
        os.nice(10)
        import torch
        from transformers import UMT5EncoderModel,T5TokenizerFast
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        from bank import TextBank
        data=TextBank(args.bank_job/'bank.pt').data
        torch.set_num_threads(data['threads'])
        torch.set_num_interop_threads(1)
        ns=dict(torch=torch,UMT5EncoderModel=UMT5EncoderModel,
                T5TokenizerFast=T5TokenizerFast,prompt_clean=prompt_clean)
        loaders=functions(UTILS,['load_text_encoder','load_tokenizer'],ns)
        methods=functions(SOURCE,['_get_t5_prompt_embeds'],ns,cls='VA_Server')
        cp=Path(data['checkpoint_path'])
        model=loaders['load_text_encoder'](str(cp/'text_encoder'),torch.bfloat16,'cpu')
        model.eval().requires_grad_(False)
        tokenizer=loaders['load_tokenizer'](str(cp/'tokenizer'))

        class Trim:
            def __init__(self):self.valid=None
            def __call__(self,*a,**k):
                batch=tokenizer(*a,**k)
                if batch.input_ids.shape != (1,512):raise ValueError('original fixed padding changed')
                self.valid=int(batch.attention_mask.sum())
                batch['input_ids']=batch.input_ids[:,:self.valid].contiguous()
                batch['attention_mask']=batch.attention_mask[:,:self.valid].contiguous()
                return batch

        proxy=type('OriginalTextHelper',(),methods)()
        proxy.device,proxy.dtype=torch.device('cpu'),torch.bfloat16
        proxy.text_encoder,proxy.tokenizer=model,Trim()
        with torch.inference_mode():
            for prompt,reference in data['embeddings'].items():
                write(args.job/'status.json',dict(status='RUNNING',completed=len(rows),planned=11,
                    active_prompt=prompt,elapsed_sec=time.monotonic()-start,CUDA_initialized=False))
                rng=torch.random.get_rng_state().clone()
                begin=time.monotonic()
                actual=proxy._get_t5_prompt_embeds(prompt,max_sequence_length=512,device=proxy.device,dtype=proxy.dtype)
                elapsed=time.monotonic()-begin
                if actual.shape!=reference.shape or not torch.equal(rng,torch.random.get_rng_state()):
                    raise ValueError('shape or inference RNG differs')
                valid=proxy.tokenizer.valid
                error=(actual.float()-reference.float()).abs()
                rows.append(dict(prompt=prompt,valid_tokens=valid,padded_tokens=512,
                    shortened_encode_sec=elapsed,bitwise_equal=bool(torch.equal(actual,reference)),
                    different_elements=int((actual!=reference).sum()),max_abs=float(error.max()),
                    valid_max_abs=float(error[:,:valid].max())))
                write(args.job/'ROWS.json',rows)
        exact=all(r['bitwise_equal'] for r in rows)
        assert not torch.cuda.is_initialized()
        final=dict(status='PASS_ALL_REAL_PROMPTS_BITWISE' if exact else 'FAIL_BITWISE_PADDING_PARITY',
            prompts=len(rows),bitwise_equal_prompts=sum(r['bitwise_equal'] for r in rows),
            max_abs=max(r['max_abs'] for r in rows),elapsed_sec=time.monotonic()-start,
            shortened_encoding_sec=sum(r['shortened_encode_sec'] for r in rows),
            source_bank=str((args.bank_job/'bank.pt').resolve()),CUDA_initialized=False,
            target_transformer_loaded=False,LIBERO_episodes=0,
            caveat='One CPU/runtime and exactly these eleven prompts. Not online reset or GPU/full-task equivalence. Padding change is not deployed; every mismatch retained.',rows=rows)
        write(args.job/'RESULT.json',final)
        write(args.job/'status.json',dict(status=final['status'],completed=len(rows),planned=11))
        print(json.dumps({k:v for k,v in final.items() if k!='rows'},ensure_ascii=False),flush=True)
    except BaseException as exc:
        write(args.job/'FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc(),completed=len(rows)))
        raise
    finally:
        signal.alarm(0)


if __name__=='__main__':main()
