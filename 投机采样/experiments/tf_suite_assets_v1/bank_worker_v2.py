"""Produce selected full512 BF16 instruction banks with per-prompt receipts.

This independent CPU producer never imports the old bank runner or computes
digests. It executes only the original T5 helpers on explicitly local assets.
Completed prompt artifacts may be referenced by a fresh, explicitly requested
job; incomplete writes, failed jobs and original deadlines remain untouched.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
import traceback


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT/'投机采样/wam-p0-worktree/wan_va/wan_va_server.py'
UTILS = SOURCE.parent/'modules/utils.py'
SUITES = ('object', 'goal', 'spatial')
CAP = 256*2**20
RESERVE = 2*2**20
SCHEMA = 'tf-suite-selected-CPU-text-bank-producer/v2'
PROMPT_SCHEMA = 'tf-suite-original-T5-prompt/v2'
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stat(path):
    path = Path(path).resolve(strict=True)
    s = path.stat()
    return dict(path=str(path), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def read(path, cap=2**20):
    path = Path(path)
    require(path.stat().st_size <= cap, 'bounded producer metadata required')
    return json.loads(path.read_text(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def ticks(pid):
    try:
        return int(Path('/proc/'+str(pid)+'/stat').read_text().rpartition(') ')[2].split()[19])
    except FileNotFoundError:
        return None


def original_nodes(path, names, cls=None):
    tree = ast.parse(path.read_text(), filename=str(path))
    if cls is None:
        body = tree.body
    else:
        candidates = [n for n in tree.body if isinstance(n,ast.ClassDef) and n.name == cls]
        require(len(candidates) == 1, 'one original T5 owner class required')
        body = candidates[0].body
    nodes = [n for n in body if isinstance(n,ast.FunctionDef) and n.name in names]
    require({n.name for n in nodes} == set(names), 'original CPU T5 helper inventory differs')
    return nodes


class LimitedWriter:
    def __init__(self, stream, cap):
        self.stream, self.cap, self.written = stream, cap, 0

    def write(self, value):
        require(self.written+len(value) <= self.cap, 'global producer storage cap reached')
        n = self.stream.write(value)
        self.written += n
        return n

    def flush(self):
        self.stream.flush()


class Output:
    def __init__(self, root):
        self.root = root

    def used(self):
        return sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())

    def json(self, path, value):
        path = Path(path)
        raw = (json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        require(len(raw) <= 2**20 and self.used()+len(raw) <= CAP, 'global metadata cap reached')
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_name(path.name+'.pending')
        require(not temp.exists(), 'retain interrupted metadata instead of overwriting')
        with temp.open('xb') as stream:
            stream.write(raw)
        os.replace(temp,path)

    def tensor(self, path, value, torch):
        path = Path(path)
        require(not path.exists(), 'immutable prompt/bank already exists')
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_name(path.name+'.pending')
        require(not temp.exists(), 'retain interrupted tensor instead of retrying')
        remaining = CAP-RESERVE-self.used()
        require(remaining > 0, 'global256MiB producer payload budget exhausted')
        with temp.open('xb') as stream:
            writer = LimitedWriter(stream,remaining)
            torch.save(value,writer)
            writer.flush()
        os.rename(temp,path)
        return stat(path)


def validate_tensor(value, torch):
    require(isinstance(value,torch.Tensor) and value.device.type == 'cpu'
        and value.dtype == torch.bfloat16 and tuple(value.shape) == (1,512,4096)
        and bool(torch.isfinite(value).all()), 'original full512 CPU BF16 embedding required')


def same_bytes(a, b, torch):
    return (a.shape == b.shape and a.dtype == b.dtype and a.device.type == b.device.type == 'cpu'
        and torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--job',type=Path,required=True)
    ap.add_argument('--source-commit',required=True)
    ap.add_argument('--suites',nargs='+',choices=SUITES,required=True)
    ap.add_argument('--prompts',type=Path,required=True)
    ap.add_argument('--shared-components',type=Path,required=True)
    ap.add_argument('--asset-root',type=Path,required=True)
    ap.add_argument('--negative-bank',type=Path,action='append',default=[])
    ap.add_argument('--reuse-from',type=Path,action='append',default=[])
    ap.add_argument('--threads',type=int,default=7)
    ap.add_argument('--wall-cap',type=int,default=7200)
    args = ap.parse_args()
    job, base, assets = args.job.resolve(), args.shared_components.resolve(strict=True), args.asset_root.resolve(strict=True)
    require(not job.exists() and os.getenv('CUDA_VISIBLE_DEVICES') == '', 'fresh explicitly CPU-only namespace required')
    require(1 <= len(args.suites) <= 2 and len(set(args.suites)) == len(args.suites),
        'one or two explicit suites fit the256MiB global artifact budget')
    require(args.threads == 7 and 1 <= len(os.sched_getaffinity(0)) <= 7, 'original seven Torch threads and bounded CPU affinity required')
    require(60 <= args.wall_cap <= 7200 and bool(args.source_commit), 'explicit bounded producer lifetime/source required')
    public = read(args.prompts)
    for suite in args.suites:
        prompts = public[suite]['public_task_prompts']
        require(public[suite]['suite'] == 'libero_'+suite and set(prompts) == {str(i) for i in range(10)}
            and all(type(v) is str and v for v in prompts.values()) and len(set(prompts.values())) == 10,
            'ten unique actual suite instructions required')
        for part in ('text_encoder','tokenizer'):
            require((base/part).is_dir() and (assets/suite/'runtime'/part).resolve(strict=True) == (base/part).resolve(strict=True),
                'suite runtime must reference the same shared original T5/tokenizer source')
    nodes = original_nodes(UTILS,('load_text_encoder','load_tokenizer'))
    methods = original_nodes(SOURCE,('_get_t5_prompt_embeds','encode_prompt'),cls='VA_Server')
    math_proof = dict(source=stat(SOURCE),utils=stat(UTILS),
        functions={n.name:ast.dump(n,include_attributes=False) for n in nodes+methods},
        source_identity='raw original helper AST and input paths; no cryptographic digest')
    input_identity = dict(shared_components=str(base),text_encoder=str((base/'text_encoder').resolve(strict=True)),
        tokenizer=str((base/'tokenizer').resolve(strict=True)))
    job.mkdir(parents=True,exist_ok=False)
    output = Output(job)
    started, birth = time.monotonic(),time.time()
    deadline = birth+args.wall_cap
    reg = dict(schema=SCHEMA,phase='PRODUCE_SELECTED_ORIGINAL_T5_BANKS_CPU_ONLY',goal='complete selected full512 BF16 banks with recoverable prompt references',
        source_commit=args.source_commit,suites=args.suites,inputs=input_identity,prompts=stat(args.prompts),asset_root=str(assets),
        negative_bank_candidates=[str(p.resolve()) for p in args.negative_bank],reuse_from=[str(p.resolve()) for p in args.reuse_from],
        torch_threads=7,CPU_affinity=sorted(os.sched_getaffinity(0)),GPU=0,wall_cap_sec=args.wall_cap,
        birth_epoch=birth,deadline_epoch=deadline,global_storage_cap_bytes=CAP,metadata_reserve_bytes=RESERVE,
        pid=os.getpid(),birth_ticks=ticks(os.getpid()),no_SHA=True,no_tests=True,no_automatic_retry=True,permissions=PERMISSIONS,
        cleanup='single CPU producer exits; no child/model worker or simulator; retain every completed and partial artifact')
    output.json(job/'REGISTRATION.json',reg)
    output.json(job/'MATHEMATICAL-SOURCE.json',math_proof)
    completed, prompts_done, negative_admissions = [],[],[]
    old_handlers = {s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM)}
    def alarm(signum,*_):
        raise TimeoutError('registered selected-bank producer lifetime or owned termination: '+str(signum))
    for sig in old_handlers:
        signal.signal(sig,alarm)
    signal.setitimer(signal.ITIMER_REAL,max(.001,deadline-time.time()))
    try:
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
        import torch
        from transformers import UMT5EncoderModel,T5TokenizerFast
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
        torch.set_num_threads(7)
        torch.set_num_interop_threads(1)
        require(not torch.cuda.is_initialized(), 'CPU producer cannot initialize CUDA')
        namespace = dict(torch=torch,UMT5EncoderModel=UMT5EncoderModel,T5TokenizerFast=T5TokenizerFast,prompt_clean=prompt_clean)
        exec(compile(ast.Module(body=nodes+methods,type_ignores=[]),str(SOURCE)+':original-T5-only','exec'),namespace)
        previous = []
        for supplied in args.reuse_from:
            old = supplied.resolve(strict=True)
            require(old != job, 'reuse must reference a distinct previous namespace')
            prior = read(old/'REGISTRATION.json')
            require(prior['schema'] == SCHEMA and prior['inputs'] == input_identity
                and (old/'RESULT.json').exists() != (old/'FAILED.json').exists(), 'closed compatible predecessor required')
            require(ticks(prior['pid']) != prior['birth_ticks'], 'never reuse a still-active producer')
            require(read(old/'MATHEMATICAL-SOURCE.json')['functions'] == math_proof['functions'], 'reused original T5 mathematics differs')
            previous.append(old)
        negative = None
        negative_origin = None
        for candidate in args.negative_bank:
            try:
                path = candidate.resolve(strict=True)
                require(path.stat().st_size <= 64*2**20, 'bounded complete source bank required')
                result_path = path.parent/'RESULT.json'
                receipt = read(result_path)
                require(receipt['status'] == 'COMPLETE_CPU_FULL512_BANK', 'only a complete original suite bank supplies negative')
                data = torch.load(path,map_location='cpu',weights_only=True)
                require(data['schema'] == 'libero-suite-original-T5-bank/v1' and data['sequence_length'] == 512
                    and data['dtype'] == 'bfloat16' and data['output_shape'] == [1,512,4096]
                    and data['torch_version'] == str(torch.__version__) and data['suite'] in ['libero_'+s for s in SUITES]
                    and set(data['public_task_prompts']) == {str(i) for i in range(10)}
                    and len(data['embeddings']) == 11
                    and set(data['embeddings']) == set(data['public_task_prompts'].values())|{''}
                    and tuple(data['source_functions']) == ('_get_t5_prompt_embeds','encode_prompt')
                    and Path(data['mathematical_source_path']).resolve(strict=True) == SOURCE.resolve(strict=True),
                    'complete full512 source/runtime/schema differs')
                source_runtime = Path(data['checkpoint_path']).resolve(strict=True)
                require(all((source_runtime/p).resolve(strict=True) == (base/p).resolve(strict=True)
                    for p in ('text_encoder','tokenizer')), 'negative source T5/tokenizer base differs')
                for v in data['embeddings'].values():
                    validate_tensor(v,torch)
                negative = data['embeddings'][''].detach().clone()
                negative_origin = dict(kind='REFERENCE_COMPLETE_SUITE_BANK_NEGATIVE',file=stat(path),receipt=stat(result_path),
                    suite=data['suite'],source_commit=data['source_commit'],shared_inputs=input_identity,
                    input_file_cryptographic_identity='UNKNOWN_NOT_HASHED',historical_math_bytes='UNKNOWN_NOT_STORED_BY_V1_BANK')
                negative_admissions.append(dict(file=str(path),status='ACCEPTED_COMPLETE_SAME_BASE_BANK'))
                del data
                break
            except TimeoutError:
                raise
            except (OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                negative_admissions.append(dict(file=str(candidate),status='NOT_REUSED_COMPUTE_ONCE_IF_NEEDED',error=repr(exc)))
        proxy = None
        def encode(prompt):
            nonlocal proxy
            if proxy is None:
                encoder = namespace['load_text_encoder'](str(base/'text_encoder'),torch.bfloat16,'cpu').eval().requires_grad_(False)
                tokenizer = namespace['load_tokenizer'](str(base/'tokenizer'))
                proxy = type('OriginalSuiteTextHelpers',(),{n.name:namespace[n.name] for n in methods})()
                proxy.device,proxy.dtype = torch.device('cpu'),torch.bfloat16
                proxy.text_encoder,proxy.tokenizer = encoder,tokenizer
            before = torch.random.get_rng_state().clone()
            with torch.inference_mode():
                value = proxy._get_t5_prompt_embeds(prompt,num_videos_per_prompt=1,max_sequence_length=512,
                    device=torch.device('cpu'),dtype=torch.bfloat16)
            validate_tensor(value,torch)
            require(torch.equal(before,torch.random.get_rng_state()), 'original T5 prompt forward consumed CPU RNG')
            return value.detach().clone()
        def reuse_prompt(suite,key,prompt):
            for old in previous:
                record_path = old/suite/'prompts'/(key+'.json')
                if not record_path.is_file():
                    continue
                r = read(record_path)
                require(r['schema'] == PROMPT_SCHEMA and r['status'] == 'COMPLETE_ORIGINAL_PROMPT_ARTIFACT'
                    and r['suite'] == 'libero_'+suite and r['prompt_key'] == key and r['prompt'] == prompt
                    and r['torch_version'] == str(torch.__version__) and r['shared_inputs'] == input_identity,
                    'completed referenced prompt receipt identity differs')
                file = Path(r['tensor']['path']).resolve(strict=True)
                require(stat(file) == r['tensor'] and file.stat().st_size <= 8*2**20,
                    'referenced prompt file changed')
                value = torch.load(file,map_location='cpu',weights_only=True)
                validate_tensor(value,torch)
                return value,dict(kind='REFERENCE_PREVIOUS_COMPLETE_PROMPT',file=stat(file),receipt=stat(record_path),predecessor=str(old))
            return None,None
        for suite in args.suites:
            folder = job/suite
            folder.mkdir(exist_ok=False)
            prompts = public[suite]['public_task_prompts']
            embeddings, timers, references = {},[],[]
            for key,prompt in [('negative','')]+list(prompts.items()):
                require(time.time() < deadline, 'producer lifetime expired before next original prompt')
                output.json(job/'status.json',dict(schema=SCHEMA,status='PRODUCING_ORIGINAL_CPU_T5_ARTIFACT',suite=suite,prompt_key=key,
                    completed_suites=completed,completed_prompts=prompts_done,wall_sec=time.monotonic()-started,GPU=0,
                    negative_admissions=negative_admissions,global_bytes=output.used()))
                tick = time.monotonic()
                value,origin = reuse_prompt(suite,key,prompt)
                if value is None and prompt == '' and negative is not None:
                    value,origin = negative.clone(),negative_origin
                if value is None:
                    value = encode(prompt)
                    origin = dict(kind='ACTUAL_ORIGINAL_CPU_T5_FORWARD',shared_inputs=input_identity,
                        mathematical_source=str(job/'MATHEMATICAL-SOURCE.json'),source_commit=args.source_commit)
                validate_tensor(value,torch)
                if prompt == '':
                    negative,negative_origin = value.detach().clone(),origin
                # Reference previous immutable prompt tensors; otherwise persist
                # the real output before advancing to the next expensive prompt.
                if origin['kind'] == 'REFERENCE_PREVIOUS_COMPLETE_PROMPT':
                    tensor_info = origin['file']
                else:
                    tensor_info = output.tensor(folder/'prompts'/(key+'.pt'),value,torch)
                    actual = torch.load(tensor_info['path'],map_location='cpu',weights_only=True)
                    require(same_bytes(value,actual,torch), 'saved actual prompt bytes changed')
                record = dict(schema=PROMPT_SCHEMA,status='COMPLETE_ORIGINAL_PROMPT_ARTIFACT',suite='libero_'+suite,
                    prompt_key=key,prompt=prompt,tensor=tensor_info,origin=origin,shared_inputs=input_identity,
                    torch_version=str(torch.__version__),sequence_length=512,dtype='bfloat16',output_shape=[1,512,4096],
                    source_commit=args.source_commit,mathematical_source=str(job/'MATHEMATICAL-SOURCE.json'),
                    wall_sec=time.monotonic()-tick,no_SHA=True,not_an_online_equivalence_test=True)
                output.json(folder/'prompts'/(key+'.json'),record)
                embeddings[prompt] = value
                references.append(str(folder/'prompts'/(key+'.json')))
                timers.append(dict(prompt_key=key,sec=record['wall_sec'],origin=origin['kind']))
                prompts_done.append(suite+'/'+key)
            payload = dict(schema='libero-suite-original-T5-bank/v1',suite='libero_'+suite,
                target_checkpoint_id='Wjjjh/lingbot-va-libero-'+suite,checkpoint_path=str(assets/suite/'runtime'),
                sequence_length=512,dtype='bfloat16',output_shape=[1,512,4096],torch_version=str(torch.__version__),
                threads=7,public_task_prompts=prompts,embeddings=embeddings,mathematical_source_path=str(SOURCE),
                source_commit=args.source_commit,source_functions=('_get_t5_prompt_embeds','encode_prompt'),no_digest=True,
                shared_inputs=input_identity,producer_schema=SCHEMA,prompt_references=references)
            bank = output.tensor(folder/'bank.pt',payload,torch)
            persisted = torch.load(bank['path'],map_location='cpu',weights_only=True)
            require(set(persisted['embeddings']) == set(embeddings)
                and all(same_bytes(embeddings[p],persisted['embeddings'][p],torch) for p in embeddings),
                'complete suite bank persistence changed original embedding bytes')
            output.json(folder/'RESULT.json',dict(status='COMPLETE_CPU_FULL512_BANK',suite='libero_'+suite,bank=bank,
                bank_bytes=bank['bytes'],prompts=11,CPU=7,CPU_affinity=sorted(os.sched_getaffinity(0)),GPU=0,timers=timers,
                online_equivalence='UNKNOWN_UNTIL_ACTUAL_RESET_AUDIT',source_digest_executed=False,permissions=PERMISSIONS))
            completed.append(suite)
        require(not torch.cuda.is_initialized(), 'CPU producer initialized CUDA')
        result = dict(schema=SCHEMA,status='COMPLETE_SELECTED_CPU_BANKS',suites=completed,completed_prompts=prompts_done,
            wall_sec=time.monotonic()-started,artifact_bytes=output.used(),global_storage_cap_bytes=CAP,
            CPU=7,CPU_affinity=sorted(os.sched_getaffinity(0)),GPU=0,target_transformer_loaded=False,
            no_episodes_or_tests=True,no_SHA=True,no_automatic_retry=True,source_commit=args.source_commit,
            online_equivalence='UNKNOWN_UNTIL_ACTUAL_RESET_AUDIT',negative_admissions=negative_admissions,permissions=PERMISSIONS)
        output.json(job/'RESULT.json',result)
        output.json(job/'status.json',result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        return 0
    except BaseException as exc:
        failure = dict(schema=SCHEMA,status='FAILED_OR_PARTIAL_SELECTED_CPU_BANKS',error=repr(exc),traceback=traceback.format_exc(),
            completed_suites=completed,completed_prompts=prompts_done,wall_sec=time.monotonic()-started,
            GPU=0,retain_completed_and_partial_artifacts=True,no_automatic_retry=True,permissions=PERMISSIONS)
        output.json(job/'FAILED.json',failure)
        output.json(job/'status.json',failure)
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in old_handlers.items():
            signal.signal(sig,handler)


if __name__ == '__main__':
    raise SystemExit(main())
