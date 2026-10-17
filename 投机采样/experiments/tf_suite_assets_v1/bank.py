"""Produce suite instruction metadata and full512 BF16 T5 banks, CPU only."""
from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import signal
import time
import types

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'投机采样/wam-p0-worktree/wan_va/wan_va_server.py'
UTILS = SOURCE.parent/'modules/utils.py'
SUITES=('object','goal','spatial')


def write(path, data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.pending')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    os.replace(tmp,path)


def functions(path,names,namespace,cls=None):
    tree=ast.parse(path.read_text())
    body=tree.body if cls is None else next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls).body
    nodes=[n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):raise ValueError('original CPU text helper names changed')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return {k:namespace[k] for k in names}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--source-commit',required=True)
    ap.add_argument('--prompts-only',action='store_true')
    ap.add_argument('--prompts',type=Path)
    ap.add_argument('--shared-components',type=Path)
    ap.add_argument('--asset-root',type=Path)
    ap.add_argument('--threads',type=int,default=7)
    ap.add_argument('--wall-cap',type=int,default=3600)
    args=ap.parse_args();job=args.job.resolve()
    if job.exists() or os.getenv('CUDA_VISIBLE_DEVICES')!='':raise ValueError('fresh CPU-only asset job required')
    if not 1<=args.threads<=7 or len(os.sched_getaffinity(0))>args.threads:raise ValueError('bounded registered CPU affinity required')
    job.mkdir(parents=True)
    started=time.monotonic();birth=time.time()
    write(job/'REGISTRATION.json',dict(goal='public task manifests' if args.prompts_only else 'complete original512 BF16 text banks',
        source_commit=args.source_commit,CPU=args.threads,GPU=0,wall_cap_sec=args.wall_cap,
        birth_epoch=birth,deadline_epoch=birth+args.wall_cap,storage_cap_bytes=256*2**20,
        pid=os.getpid(),birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]),
        cleanup='single bounded CPU process, no simulator/model worker; preserve partial result'))
    def alarm(*_):raise TimeoutError('registered CPU artifact lifetime')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(args.wall_cap)
    complete=[]
    try:
        if args.prompts_only:
            os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
            from libero.libero import benchmark
            data={}
            for suite in SUITES:
                b=benchmark.get_benchmark_dict()['libero_'+suite]()
                prompts={str(i):b.get_task(i).language for i in range(10)}
                if len(set(prompts.values()))!=10:raise ValueError('unique ten-task prompt coverage differs')
                states={str(i):len(b.get_task_init_states(i)) for i in range(10)}
                if any(n<50 for n in states.values()):raise ValueError('fewer than50 benchmark initial states')
                data[suite]=dict(suite='libero_'+suite,public_task_prompts=prompts,initial_state_counts=states)
            write(job/'PROMPTS.json',data)
            write(job/'RESULT.json',dict(status='COMPLETE_PUBLIC_BENCHMARK_INPUTS',wall_sec=time.monotonic()-started,
                suites=list(data),GPU=0,simulator_instances=0,episodes=0))
            return 0
        if not args.prompts or not args.shared_components or not args.asset_root:raise ValueError('bank inputs required')
        import torch
        from transformers import UMT5EncoderModel,T5TokenizerFast
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
        if torch.cuda.is_initialized():raise ValueError('CPU artifact process initialized CUDA')
        ns=dict(torch=torch,UMT5EncoderModel=UMT5EncoderModel,T5TokenizerFast=T5TokenizerFast,prompt_clean=prompt_clean)
        loaders=functions(UTILS,('load_text_encoder','load_tokenizer'),ns)
        methods=functions(SOURCE,('_get_t5_prompt_embeds','encode_prompt'),ns,cls='VA_Server')
        base=args.shared_components.resolve(strict=True)
        encoder=loaders['load_text_encoder'](str(base/'text_encoder'),torch.bfloat16,'cpu').eval().requires_grad_(False)
        tokenizer=loaders['load_tokenizer'](str(base/'tokenizer'))
        proxy=type('OriginalSuiteTextHelpers',(),methods)()
        proxy.device,proxy.dtype=torch.device('cpu'),torch.bfloat16
        proxy.text_encoder,proxy.tokenizer=encoder,tokenizer
        public=json.loads(args.prompts.read_text())
        negative=None
        for suite in SUITES:
            folder=job/suite;folder.mkdir()
            prompts=public[suite]['public_task_prompts']
            embeddings={}
            timers=[]
            for name,prompt in [('negative','')]+list(prompts.items()):
                write(job/'status.json',dict(status='ENCODING_REAL_REUSABLE_PROMPT',suite=suite,prompt=name,
                    completed_suites=complete,wall_sec=time.monotonic()-started,CPU=args.threads,GPU=0))
                before=torch.random.get_rng_state().clone();tick=time.monotonic()
                if prompt=='' and negative is not None:
                    value=negative.clone()
                else:
                    with torch.inference_mode():value=proxy._get_t5_prompt_embeds(prompt,num_videos_per_prompt=1,
                        max_sequence_length=512,device=torch.device('cpu'),dtype=torch.bfloat16)
                if tuple(value.shape)!=(1,512,4096) or value.dtype!=torch.bfloat16 or not bool(torch.isfinite(value).all()):
                    raise ValueError('complete original text embedding contract differs')
                if not torch.equal(before,torch.random.get_rng_state()):raise ValueError('CPU prompt construction consumed RNG')
                embeddings[prompt]=value.detach().clone()
                if prompt=='':negative=value.detach().clone()
                timers.append(dict(prompt_key=name,sec=time.monotonic()-tick))
            payload=dict(schema='libero-suite-original-T5-bank/v1',suite='libero_'+suite,
                target_checkpoint_id='Wjjjh/lingbot-va-libero-'+suite,
                checkpoint_path=str(args.asset_root.resolve()/suite/'runtime'),
                sequence_length=512,dtype='bfloat16',output_shape=[1,512,4096],
                torch_version=str(torch.__version__),threads=args.threads,
                public_task_prompts=prompts,embeddings=embeddings,
                mathematical_source_path=str(SOURCE),source_commit=args.source_commit,
                source_functions=('_get_t5_prompt_embeds','encode_prompt'),no_digest=True)
            tmp=folder/'bank.pending.pt';torch.save(payload,tmp);os.replace(tmp,folder/'bank.pt')
            persisted=torch.load(folder/'bank.pt',map_location='cpu',weights_only=True)
            if set(persisted['embeddings'])!=set(embeddings) or any(not torch.equal(embeddings[k],persisted['embeddings'][k]) for k in embeddings):
                raise ValueError('bank persistence changed actual embedding bytes')
            write(folder/'RESULT.json',dict(status='COMPLETE_CPU_FULL512_BANK',suite='libero_'+suite,
                bank_bytes=(folder/'bank.pt').stat().st_size,prompts=11,CPU=args.threads,GPU=0,timers=timers,
                online_equivalence='UNKNOWN_UNTIL_ACTUAL_RESET_AUDIT',source_digest_executed=False))
            complete.append(suite)
        write(job/'RESULT.json',dict(status='COMPLETE_THREE_CPU_BANKS',suites=complete,
            wall_sec=time.monotonic()-started,CPU=args.threads,GPU=0,target_transformer_loaded=False,
            no_episodes_or_tests=True,source_commit=args.source_commit))
        write(job/'status.json',dict(status='COMPLETE_THREE_CPU_BANKS',wall_sec=time.monotonic()-started))
        return 0
    except BaseException as exc:
        failure=dict(status='FAILED_OR_PARTIAL_CPU_BANK_INPUT',error=repr(exc),completed_suites=complete,
            wall_sec=time.monotonic()-started,GPU=0)
        write(job/'FAILED.json',failure);write(job/'status.json',failure)
        raise
    finally:signal.alarm(0)


if __name__=='__main__':raise SystemExit(main())
