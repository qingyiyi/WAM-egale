"""Temporary5090 complete-policy server; frozen model/draft math, no SHA/smoke.

New observations are explicit and do not amend any previous quality certificate.
The unused flash-attn import and T5 construction are replaced only when the
configured engine is original Torch attention and all prompts use the saved bank.
"""
from pathlib import Path
from datetime import datetime
import argparse,ast,contextlib,copy,importlib.machinery,inspect,json,math,os,random,sys,textwrap,time,traceback,types
import numpy as np
import torch

P=Path('/root/autodl-tmp/cxz/spec-speed-20261025')
A=argparse.ArgumentParser();A.add_argument('--suite',required=True);A.add_argument('--port',type=int,default=19601);A.add_argument('--master-port',type=int,default=29601);A.add_argument('--attempt',default='01');v=A.parse_args()
OUT=P/'results'/v.suite/('server-'+v.attempt);OUT.mkdir(parents=True,exist_ok=False)
(OUT/'SERVER-SOURCE.py').write_text(Path(__file__).read_text())
state=dict(status='LOADING',start=time.time(),suite=v.suite,teacher_attempted=0,teacher_returned=0,policies=0,permissions=dict(engineering_quality_screen=False,independent_test_complete=False,noninferiority_established=False,formal_performance_permission=False))
cur={};context={};rt=None;proxy=None

def save():
    f=OUT/'STATUS.json';t=f.with_suffix('.tmp');t.write_text(json.dumps(state,indent=2)+'\n');t.replace(f)

def write(name,value):
    f=OUT/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

def seed(value):
    random.seed(value);np.random.seed(value);torch.manual_seed(value);torch.cuda.manual_seed_all(value)

def gpu_done():
    torch.cuda.synchronize()

def g_now():
    return cur.get('counters',{}).get('committed_scheduler_update_count',0) if cur.get('scheduled')=='S' else cur.get('scheduler',0)

@contextlib.contextmanager
def round_scope(_s):
    before=g_now();start=time.perf_counter()
    try:yield
    finally:cur.setdefault('rounds',[]).append(dict(host_seconds=time.perf_counter()-start,g=g_now()-before))

@contextlib.contextmanager
def action_scope(_s):
    gpu_done();start=time.perf_counter();begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);begin.record()
    try:yield
    finally:
        end.record();end.synchronize();cur['action']=dict(wall_seconds=time.perf_counter()-start,GPU_event_seconds=begin.elapsed_time(end)/1000)

def annotate(fn,tag,*,tree=False,target=False,robot_action=False):
    """Only bracket existing full loop bodies; mathematical statements unchanged."""
    original=inspect.unwrap(fn)
    try:raw=textwrap.dedent(inspect.getsource(original))
    except OSError:
        # Existing Runtime emits one standalone function but compiles its old
        # source line numbers. Read only that owned generated function file.
        file=Path(original.__code__.co_filename).resolve();assert file.is_relative_to(OUT)
        raw=file.read_text();defs=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef)]
        assert len(defs)==1 and defs[0].name==original.__name__
    mod=ast.parse(raw);count=0
    class Observe(ast.NodeTransformer):
        def visit_While(self,node):
            nonlocal count
            node=self.generic_visit(node)
            if tree and any(isinstance(x,ast.Name) and x.id in ('action_step','step_index','action_step_index') for x in ast.walk(node.test)):
                node.body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_bench_round_scope',ctx=ast.Load()),args=[],keywords=[]))],body=node.body)];count+=1
            return node
        def visit_For(self,node):
            nonlocal count
            node=self.generic_visit(node)
            if target and any(isinstance(x,ast.Name) and x.id=='action_timesteps' for x in ast.walk(node.iter)):
                node.body=[ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_bench_round_scope',ctx=ast.Load()),args=[],keywords=[]))],body=node.body)];count+=1
            return node
        def visit_If(self,node):
            nonlocal count
            node=self.generic_visit(node)
            if robot_action and ast.unparse(node.test)=='self._robotwin_spec':
                count+=1
                return ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='_bench_action_scope',ctx=ast.Load()),args=[],keywords=[]))],body=[node])
            return node
    mod=ast.fix_missing_locations(Observe().visit(mod))
    if count!=1:raise RuntimeError(f'Exactly one existing loop/action scope required for {tag}, found {count}')
    f=OUT/(tag+'.py');raw=ast.unparse(mod)+'\n';f.write_text(raw);ns=dict(original.__globals__);exec(compile(raw,str(f),'exec'),ns);return ns[original.__name__]

def unused_flash_import():
    # Original configured Torch path never calls this optional backend.
    def unavailable(*a,**k):raise RuntimeError('flash-attn backend unavailable; benchmark is original Torch backend')
    m=types.ModuleType('flash_attn');m.__spec__=importlib.machinery.ModuleSpec('flash_attn',loader=None);m.flash_attn_func=unavailable;m.flash_attn_with_kvcache=unavailable;sys.modules['flash_attn']=m

class BankOnlyEncoder(torch.nn.Module):
    def __init__(self):super().__init__();self.register_parameter('unused',torch.nn.Parameter(torch.empty(1,dtype=torch.bfloat16,device='cpu'),requires_grad=False))
    def forward(self,*a,**k):raise RuntimeError('Full512 saved bank required; unused T5 must never execute')

def make_predictor(plan,s):
    from draft_architecture_v1 import models
    from target_trace_draft_train_v1.adapter import ActionPredictor,NORMALIZATION
    path=P/'bundle/small-assets'/v.suite/'draft.pt';data=torch.load(path,map_location='cpu',weights_only=True)
    assert data['architecture']=='transformer64' and data['specification']==models.SPECS['transformer64']
    assert data['metadata']==plan['selected']['metadata']
    assert all(data['normalization'][k].shape==shape and data['normalization'][k].dtype==torch.float32 for k,shape in NORMALIZATION.items())
    with torch.random.fork_rng(devices=[]):model=models.make_model('transformer64').float()
    model.load_state_dict(data['model_state_dict'],strict=True)
    assert sum(p.numel() for p in model.parameters())==53070
    return ActionPredictor(model,data['normalization'],architecture='transformer64',checkpoint_path=path,checkpoint_sha256='NOT_COMPUTED_USER_NO_SHA',device=s.device)

def install_bank(s,plan):
    bank_path=P/'small-assets/robotwin-bank.pt' if v.suite=='robotwin' else P/'bundle/small-assets'/v.suite/'bank.pt'
    bank=torch.load(bank_path,map_location='cpu',weights_only=False)
    entries=bank['embeddings'];assert entries and '' in entries
    for x in entries.values():assert x.dtype==torch.bfloat16 and tuple(x.shape)==(1,512,4096)
    if v.suite!='robotwin':
        expected=plan.get('suite_info',{}).get('checkpoint_id',plan.get('target_checkpoint_id'))
        assert bank['target_checkpoint_id']==expected
    def lookup(_s,prompt=None,num_videos_per_prompt=1,max_sequence_length=512,device=None,dtype=None):
        if isinstance(prompt,list) and len(prompt)==1:prompt=prompt[0]
        assert prompt in entries and num_videos_per_prompt==1 and max_sequence_length==512
        assert dtype in (None,torch.bfloat16)
        state['bank_hits']=state.get('bank_hits',0)+1
        return entries[prompt].to(device=device or _s.device,dtype=torch.bfloat16).clone()
    s._get_t5_prompt_embeds=types.MethodType(lookup,s)
    write('BANK-DEPLOYMENT.json',dict(sequence_length=512,dtype='bfloat16',entries=len(entries),original_torch_version=bank.get('torch_version','NOT_RECORDED'),current_torch_version=str(torch.__version__),original_bank_metadata_changed=False,T5_loaded=False,T5_forward=False,no_hash=True,scope='Copied frozen BF16 embeddings; cross-CUDA reuse is not inherited CAL/quality qualification'))

try:
    save();torch.set_num_threads(2);torch.set_num_interop_threads(1)
    os.environ.update(RANK='0',LOCAL_RANK='0',WORLD_SIZE='1',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(v.master_port),LINGBOT_ATTN_MODE='torch',WAM_P0_ACTION_DRAFT_MODE='off')
    unused_flash_import()
    project=P/'bundle/project';exp=project/'投机采样/experiments'
    if v.suite=='robotwin':
        os.environ['LINGBOT_DISABLE_FSDP']='0'
        modelroot=P/'bundle/robotwin';worktree=modelroot
        for x in [modelroot,modelroot/'wan_va',exp,P/'bundle/robotwin/runtime',P/'bundle/robotwin/eval']:sys.path.insert(0,str(x))
    else:
        worktree=project/'投机采样/wam-p0-worktree'
        # Existing strongest single-worker LIBERO deployment disables FSDP.
        # RoboTwin keeps its own original FSDP2 path in the separate branch.
        os.environ['LINGBOT_DISABLE_FSDP']='1'
        for x in [worktree,worktree/'wan_va',exp]:sys.path.insert(0,str(x))
        info=json.loads((P/'bundle/manifests'/f'{v.suite}-original-plan.json').read_text())
        suite_id='libero_10' if v.suite=='long' else 'libero_'+v.suite
        checkpoint_id=info.get('suite_info',{}).get('checkpoint_id',info.get('target_checkpoint_id'))
        registry=P/'registry.json';reg=dict(suites={suite_id:dict(status='available',official_checkpoint=checkpoint_id,local_path=str(P/'assets'/v.suite))});registry.write_text(json.dumps(reg))
        os.environ.update(LINGBOT_VA_LIBERO_SUITE=suite_id,LINGBOT_VA_LIBERO_CHECKPOINT=str(P/'assets'/v.suite),LINGBOT_VA_LIBERO_CHECKPOINT_ID=checkpoint_id,LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=str(registry))
        # Relocate resource identities before any frozen helper clones globals.
        contract_path=exp/'single_worker_chc_gpu_v4/contract.py'
        spec=__import__('importlib.util',fromlist=['util']).spec_from_file_location('contract',contract_path)
        contract=__import__('importlib.util',fromlist=['util']).module_from_spec(spec);sys.modules['contract']=contract;spec.loader.exec_module(contract)
        contract.ROOT=project;contract.WORKTREE=worktree;contract.MODEL_SOURCE=worktree/'wan_va/modules/model.py'
        assert contract.MODEL_SOURCE.read_bytes()==(P/'bundle/small-assets'/v.suite/'target-source.bytes').read_bytes()
    import wan_va_server as source
    source.load_text_encoder=lambda *a,**k:BankOnlyEncoder()
    source.save_async=lambda *a,**k:None
    source.init_logger()
    plan=json.loads((P/'bundle/manifests'/f'{v.suite}-original-plan.json').read_text())
    cfg=copy.deepcopy(source.VA_CONFIGS['robotwin' if v.suite=='robotwin' else 'libero'])
    cfg.wan22_pretrained_model_name_or_path=str(P/'assets'/v.suite);cfg.save_root=str(OUT/'unused');cfg.rank=cfg.local_rank=0;cfg.world_size=1
    if v.suite!='robotwin':
        cfg.libero_eval_suite=suite_id;cfg.libero_checkpoint_id=checkpoint_id;cfg.libero_checkpoint_registry=str(P/'registry.json')
        if 'suite_info' in plan:cfg.norm_stat=copy.deepcopy(plan['suite_info']['norm_stat'])
    assert cfg.action_num_inference_steps==50 and cfg.num_inference_steps==(25 if v.suite=='robotwin' else 20)
    source.init_distributed(1,0,0);s=source.VA_Server(cfg)
    s.action_per_frame=int(cfg.action_per_frame)
    s._bench_round_scope=types.MethodType(round_scope,s);s._bench_action_scope=types.MethodType(action_scope,s)
    install_bank(s,plan)
    if v.suite=='robotwin':
        mesh=types.ModuleType('mesh_helper');mesh.get_mesh_id=source.get_mesh_id;sys.modules['mesh_helper']=mesh
        import server_math
        server_math._infer_action_verifier_k2_b2=annotate(server_math._infer_action_verifier_k2_b2,'robotwin-spec-rounds',tree=True)
        from eval_runtime import EvalRuntime
        rt=EvalRuntime(s,source,P/'bundle/small-assets/robotwin/draft.pt',plan['tau'],OUT/'adapter')
        s._infer=types.MethodType(annotate(s._infer,'robotwin-target-rounds',target=True),s)
        s._infer=types.MethodType(annotate(s._infer,'robotwin-action-span',robot_action=True),s)
    else:
        s._p0_action_draft_b2_predictor=make_predictor(plan,s)
        s._p0_action_draft_b2_checkpoint=s._p0_action_draft_b2_predictor.checkpoint_path
        s._p0_action_draft_threshold=plan['threshold'];s._p0_action_draft_mode=plan['mode']
        # Annotate the real tree loop before the unchanged O3 source generator.
        s._infer_action_verifier_k2_b2=types.MethodType(annotate(s._infer_action_verifier_k2_b2,'libero-spec-rounds',tree=True),s)
        from tf_fullsuite_eval_v2.server_entry import prefix_modules
        modules=prefix_modules();proxy=modules['selection'].install(s,plan['selector'])
        modules['buffers'].prepare(s._infer_action_verifier_k2_b2,s._load_p0_shadow_api())
        from tf_fullsuite_eval_v2.shared_prefix_no_digest import install_o3
        online=install_o3(s,original=modules['measured_online'],source_reference=P/'bundle/small-assets'/v.suite/'target-source.bytes')
        s._infer=types.MethodType(annotate(s._infer,'libero-target-rounds',target=True),s)
        oldstart,oldend=s._phase_profile_start,s._phase_profile_end
        def phase_start(name,*a,**k):
            if name!='action_total':return oldstart(name,*a,**k)
            ctx=action_scope(s);ctx.__enter__();return ('BENCH_ACTION',ctx)
        def phase_end(token,*a,**k):
            if isinstance(token,tuple) and len(token)==2 and token[0]=='BENCH_ACTION':return token[1].__exit__(None,None,None)
            return oldend(token,*a,**k)
        s._phase_profile_start=phase_start;s._phase_profile_end=phase_end
    original_counter=s._phase_profile_counter
    def counter(name,value=1):
        if cur:cur.setdefault('counters',{})[name]=cur.get('counters',{}).get(name,0)+value
        return original_counter(name,value)
    s._phase_profile_counter=counter
    original_step=s.action_scheduler.step
    def step(*a,**k):
        if cur:cur['scheduler']=cur.get('scheduler',0)+1
        return original_step(*a,**k)
    s.action_scheduler.step=step
    forward=s.transformer.forward
    def counted(*a,**k):
        if state['teacher_attempted']>=600000:raise RuntimeError('Registered overall teacher attempt cap')
        if context and state['teacher_attempted']-context.get('teacher_begin',state['teacher_attempted'])>=(8000 if v.suite=='robotwin' else 4000):raise RuntimeError('Registered complete-episode teacher attempt cap')
        state['teacher_attempted']+=1
        inp=a[0] if a else k.get('latent_dict');batch=int(inp['noisy_latents'].shape[0]);kind='action' if k.get('action_mode',False) else 'video'
        key=f'{kind}_B{batch}_update{k.get("update_cache",0)}'
        cur.setdefault('forwards',{}).setdefault(key,dict(attempted=0,returned=0));cur['forwards'][key]['attempted']+=1
        r=forward(*a,**k);state['teacher_returned']+=1;cur['forwards'][key]['returned']+=1
        if state['teacher_attempted']%50==0:save()
        return r
    # The unchanged lease controller binds this public signature to identify
    # candidate_cache_rows. A variadic observer must not hide those arguments.
    counted.__signature__=inspect.signature(forward)
    s.transformer.forward=counted
    canonical=s.infer
    def infer(_s,request):
        global cur,context
        request=dict(request);tag=request.pop('_benchmark',None)
        kind='reset' if request.get('reset') else 'cache' if request.get('compute_kv_cache') else 'policy'
        if kind=='reset':
            assert isinstance(tag,dict) and tag['arm'] in ('T','C','P')
            context=dict(tag,returned=0,teacher_begin=state['teacher_attempted']);seed(0 if v.suite=='robotwin' else plan['evaluation_seed'])
            directory=OUT/'reset-RNG'/context['case'];directory.mkdir(parents=True,exist_ok=True)
            torch.save(dict(python=random.getstate(),numpy=np.random.get_state(),torch_CPU=torch.get_rng_state(),torch_CUDA=torch.cuda.get_rng_state_all()),directory/(context['arm']+'.pt'))
        assert context
        scheduled=('T' if context['arm']=='T' or context['arm']=='P' and context['returned']%3==2 else 'S') if kind=='policy' else None
        cur=dict(case=context['case'],arm=context['arm'],kind=kind,scheduled=scheduled,ordinal=context['returned'],counters={},scheduler=0,forwards={},rounds=[])
        if proxy is not None:proxy.records.clear()
        gpu_done();t=time.perf_counter();route_start=t
        if rt is not None:rt.begin_chunk(kind=='policy' and scheduled=='S')
        else:s._p0_action_draft_mode=plan['mode'] if scheduled=='S' else 'off'
        route_seconds=time.perf_counter()-route_start
        try:
            result=canonical(request);gpu_done();elapsed=time.perf_counter()-t
        except BaseException as error:
            cur.update(complete=False,error=repr(error),traceback=traceback.format_exc(),request_seconds=time.perf_counter()-t,teacher_attempted_cumulative=state['teacher_attempted'],teacher_returned_cumulative=state['teacher_returned'])
            with (OUT/'failed_requests.jsonl').open('a') as f:f.write(json.dumps(cur,ensure_ascii=False)+'\n')
            state.update(status='REQUEST_FAILED',last_error=repr(error),active_case=context['case'],active_arm=context['arm']);save();raise
        audit=rt.end_chunk() if rt is not None and kind=='policy' else None
        if kind=='policy':
            g=int(cur['counters'].get('committed_scheduler_update_count',0)) if scheduled=='S' else cur['scheduler']
            assert g==50,(g,cur)
            assert cur['forwards'].get('action_B2_update1',{}).get('returned')==1,cur['forwards']
            if scheduled=='T':assert cur['forwards'].get('action_B2_update0',{}).get('returned')==50
            assert cur.get('action') and sum(x['g'] for x in cur['rounds'])==50,cur
            action_total=cur['action']['wall_seconds'];round_sum=sum(x['host_seconds'] for x in cur['rounds'])
            assert action_total>=round_sum
            cur.update(valid_commits_g=g,action_accounted_seconds=action_total+route_seconds,extra_action_setup_terminal_drain_seconds=action_total-round_sum,route_setup_seconds=route_seconds,policy_seconds=elapsed)
            context['returned']+=1;state['policies']+=1
            cur['trees']=audit['trees'] if audit else (list(proxy.records) if proxy is not None else [])
        else:cur.update(valid_commits_g=0,request_seconds=elapsed)
        cur.update(actual_route=scheduled,complete=True,suite=v.suite,trace_mode='off',no_CUDA_Graph=True,time_scope='Complete original policy/action; host round scopes plus measured action residual include terminal work; not robot episode by themselves')
        with (OUT/'requests.jsonl').open('a') as f:f.write(json.dumps(cur,ensure_ascii=False)+'\n')
        state.update(status='SERVING',active_case=context['case'],active_arm=context['arm'],active_kind=kind,active_policy_chunks=context['returned']);save()
        return result
    s.infer=types.MethodType(infer,s)
    state.update(status='READY',ready=time.time(),torch=str(torch.__version__),CUDA=torch.version.cuda,device_name=torch.cuda.get_device_name(),model_source=source.__file__,model_root=str(P/'assets'/v.suite),source_model_math_unchanged=True,benchmark_in_memory_scope_wrappers=True,T5_model_not_loaded=True,checkpoint_dtype='frozen original BF16; draft original FP32/decode BF16',single_worker_engine='canonical FSDP2' if v.suite=='robotwin' else 'existing best LIBERO noFSDP')
    state['old_quality_flags_unchanged']=True;save()
    source.run_async_server_mode(s,0,'127.0.0.1',v.port)
except BaseException as e:
    state.update(status='FAILED',error=repr(e),traceback=traceback.format_exc(),end=time.time());save();raise
