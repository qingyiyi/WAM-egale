"""Instance-only RoboTwin candidate migration. Canonical T branch is retained.

The three public forward functions are derived from the pinned original AST,
adding only an optional read-only candidate route. O2/O3 math/workspace helpers
are copied unchanged. This has its own validation, not the LIBERO certificate.
"""
from pathlib import Path
import ast,copy,types,time,contextlib,sys,json
import torch
B=Path(__file__).resolve().parent
import tree_api,server_math,buffers,prefix_selector
from draft import Transformer,Predictor,SPEC

class Runtime:
    def __init__(self, server, original, checkpoint, tau, evidence):
        self.s=server;self.out=Path(evidence);self.out.mkdir(exist_ok=True)
        self.counters={};self.phases={};self.rows=[];self.tree_records=[];self.history={};self.audits=[]
        self.shader='shared';self.shared_calls=0;self.chunk=0;self.ordinary_calls=0
        self.load_checkpoint(checkpoint)
        server._p0_action_draft_threshold=float(tau);server._p0_action_draft_b2_checkpoint=str(checkpoint)
        server._p0_action_draft_selector='prefix_then_cost';server._p0_state_teacher=None
        server._load_p0_action_draft_b2_predictor=lambda:self.predictor
        server._load_p0_shadow_api=lambda:self
        server._phase_profile_counter=self.counter;server._phase_profile_phase=self.phase
        for name in vars(server_math):
            if name.startswith(('_prepare_action','_repeat_action','_p0_action_','_infer_action_','_target_action_')):
                setattr(server,name,types.MethodType(getattr(server_math,name),server))
        self.install_model()
        self.install_infer(original)
        self.install_workspace()

    def load_checkpoint(self,path):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['schema']=='robotwin-tf64/v1' and payload['specification']==SPEC
        assert payload['channels']==list(self.s.job_config.used_action_channel_ids)
        model=Transformer().float().to(self.s.device);model.load_state_dict(payload['model_state_dict']);model.eval()
        self.predictor=Predictor(model,payload['normalization']);self.metadata=payload['metadata']

    def __getattr__(self,name):return getattr(tree_api,name)

    def counter(self,name,value=1):self.counters[name]=self.counters.get(name,0)+value

    @contextlib.contextmanager
    def phase(self,name):
        start=time.perf_counter()
        try:yield
        finally:self.phases[name]=self.phases.get(name,0.)+time.perf_counter()-start

    def select_k2_b2_route(self,scores,*,selector):
        assert selector=='prefix_then_cost'
        return prefix_selector.choose(tree_api,scores,self.s._p0_action_draft_threshold)

    def execute_k2_b2_route(self,tree,scores,route,**kwargs):
        result=tree_api.execute_k2_b2_route(tree,scores,route,**kwargs)
        step=self.counters.get('committed_scheduler_update_count',0)
        # Only reached root/selected parent labels, not every evaluated leaf.
        indices=[0]+([1+route.parent] if result.accepted_depth>=1 else [])
        for node in indices:
            local=0 if node==0 else 1
            previous=tree.root_previous_direction if node==0 else tree.first_directions[route.parent]
            x=tree.root_state if node==0 else tree.first_states[route.parent]
            self.rows.append(dict(state=x.detach().cpu().clone(),previous=previous.detach().cpu().clone(),
                target=scores.target_directions[node].detach().cpu().clone(),sigma=tree.sigma_path[local],
                next_sigma=tree.sigma_path[local+1],timestep=tree.target_timesteps[local],step=step+local,
                node=node,depth=result.accepted_depth,source='actual_reached_root_or_selected_parent'))
        self.tree_records.append(dict(step=step,depth=result.accepted_depth,first=list(scores.first_costs),
            second=[list(x) for x in scores.second_costs],parent=route.parent,child=route.child))
        return result

    def install_workspace(self):
        s=self.s;original=s._infer_action_verifier_k2_b2
        buffers.prepare(original,self)
        fn=types.MethodType(buffers.action_function(original),s)
        original_scatter=s._p0_action_scatter_compact_batch
        def action(_s,*args,**kwargs):
            assert buffers.ACTIVE.get() is None
            ws=buffers.Workspace();token=buffers.ACTIVE.set(ws)
            try:return fn(*args,**kwargs)
            finally:
                self.workspace_counts=dict(ws.counts);ws.close();buffers.ACTIVE.reset(token);self.history.clear()
        def scatter(_s,*args,**kwargs):
            ws=buffers.ACTIVE.get()
            return original_scatter(*args,**kwargs) if ws is None else ws.scatter(_s,*args,**kwargs)
        s._infer_action_verifier_k2_b2=types.MethodType(action,s)
        s._p0_action_scatter_compact_batch=types.MethodType(scatter,s)

    def attention(self,a,q,k,v,cache,rows,update):
        assert update==0 and q.shape==(14,32,24,128)
        assert tuple(rows.tolist())==(0,)*7+(1,)*7
        key=id(a);entry=self.history.get(key)
        identity=tuple((id(cache[n]),cache[n].data_ptr(),cache[n]._version) for n in ('k','v','mask','id','is_pred'))
        if entry is None:
            slots=cache['mask'].nonzero().squeeze(-1).contiguous();self.history[key]=(identity,slots)
        else:
            assert identity==entry[0],'history mutated inside candidate lease';slots=entry[1]
        assert slots.numel()>0 and cache['k'].shape[0]==2
        if self.shader=='native':
            pk=cache['k'][:,slots].index_select(0,rows);pv=cache['v'][:,slots].index_select(0,rows)
            return a.attn_op(q,torch.cat((pk,k),1),torch.cat((pv,v),1))
        import _robotwin_shared_q32 as ext
        out=ext.shared(q,cache['k'],cache['v'],k,v,slots,rows)
        self.shared_calls+=1
        # Real-input inline numerical audit on first 16 observed layer calls.
        if len(self.audits)<16:
            pk=cache['k'][:,slots].index_select(0,rows);pv=cache['v'][:,slots].index_select(0,rows)
            expected=a.attn_op(q,torch.cat((pk,k),1),torch.cat((pv,v),1))
            diff=(out.float()-expected.float()).abs()
            row=dict(call=self.shared_calls,Q=32,history_slots=int(slots.numel()),exact=torch.equal(out,expected),max_abs=float(diff.max()))
            self.audits.append(row);(self.out/'Q32-ATTENTION-AUDIT.json').write_text(json.dumps(self.audits,indent=2))
            if not row['exact']:raise RuntimeError('Q32 shared path differs from original native candidate attention')
        return out

    def install_model(self):
        model=self.s.transformer
        # Preserve both outer and per-block FSDP forward machinery. Patch only
        # their wrapped mathematical modules; never bypass all-gather/hooks.
        def unwrap(value):
            while hasattr(value, '_fsdp_wrapped_module'):
                value=value._fsdp_wrapped_module
            return value
        core=unwrap(model)
        # 官方当前用composable FSDP2：动态子类在MRO中保留数学类，
        # 不一定有FSDP1的_fsdp_wrapped_module。实例__call__及其FSDP hooks保持原样。
        math_class=next((c for c in type(core).__mro__ if c.__name__=='WanTransformer3DModel'),None)
        assert math_class is not None, type(core).__mro__
        module=sys.modules[math_class.__module__]
        tree=ast.parse(Path(module.__file__).read_text())
        compiled={}
        for cname in ['WanAttention','WanTransformerBlock','WanTransformer3DModel']:
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cname)
            fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward'))
            fn.args.args.append(ast.arg(arg='candidate_cache_rows'));fn.args.defaults.append(ast.Constant(None))
            if cname=='WanAttention':
                start=next(i for i,n in enumerate(fn.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='slots' for t in n.targets))
                end=next(i for i,n in enumerate(fn.body) if i>start and isinstance(n,ast.Assign) and ast.unparse(n.value)=='hidden_states.flatten(2, 3)')
                normal=fn.body[start:end]
                branch=ast.parse('if candidate_cache_rows is not None:\n hidden_states=self._robotwin_shared(query,key,value,kv_cache,candidate_cache_rows,update_cache)\nelse:\n pass').body[0]
                branch.orelse=normal;fn.body[start:end]=[branch]
            else:
                calls=[]
                for node in ast.walk(fn):
                    if isinstance(node,ast.Call) and ((cname=='WanTransformerBlock' and ast.unparse(node.func)=='self.attn1') or (cname=='WanTransformer3DModel' and ast.unparse(node.func)=='block')):
                        calls.append(node);node.keywords.append(ast.keyword(arg='candidate_cache_rows',value=ast.Name(id='candidate_cache_rows',ctx=ast.Load())))
                assert len(calls)==1
            mod=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));path=self.out/(cname+'-forward.py');path.write_text(ast.unparse(mod)+'\n')
            ns=dict(vars(module));exec(compile(mod,str(path),'exec'),ns);compiled[cname]=ns['forward']
        for wrapped_block in core.blocks:
            block=unwrap(wrapped_block)
            assert any(c.__name__=='WanTransformerBlock' for c in type(block).__mro__),type(block).__mro__
            block.forward=types.MethodType(compiled['WanTransformerBlock'],block)
            a=block.attn1;a.forward=types.MethodType(compiled['WanAttention'],a)
            a._robotwin_shared=lambda q,k,v,cache,rows,update,a=a:self.attention(a,q,k,v,cache,rows,update)
        core.forward=types.MethodType(compiled['WanTransformer3DModel'],core)
        canonical=model.forward
        def forward(_model,*args,**kwargs):
            if kwargs.get('candidate_cache_rows') is None:self.history.clear();self.ordinary_calls+=1
            return canonical(*args,**kwargs)
        model.forward=types.MethodType(forward,model)

    def install_infer(self,original):
        tree=ast.parse(Path(original.__file__).read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='VA_Server')
        fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_infer'))
        loops=[n for n in ast.walk(fn) if isinstance(n,ast.For) and ast.unparse(n.iter)=='enumerate(tqdm(action_timesteps))'];assert len(loops)==1
        loop=loops[0]
        parent=next(n for n in ast.walk(fn) if hasattr(n,'body') and loop in n.body)
        pre=ast.parse('if self._robotwin_spec:\n actions=self._infer_action_verifier_k2_b2(actions,action_timesteps,frame_st_id,action_inference_step,"",None,bonus_target_step=True)').body[0]
        # 复用的K2B2 helper已执行50次合法推进和一次terminal cache forward。
        # S/T整段互斥；不能在helper返回后再执行原循环的terminal一次。
        pre.orelse=[loop]
        parent.body[parent.body.index(loop)]=pre
        mod=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));path=self.out/'canonical-infer-with-spec-route.py';path.write_text(ast.unparse(mod)+'\n')
        ns=dict(vars(original));exec(compile(mod,str(path),'exec'),ns)
        self.s._infer=types.MethodType(ns['_infer'],self.s);self.s._robotwin_spec=False

    def begin_chunk(self,spec):
        self.s._robotwin_spec=bool(spec);self.counters={};self.phases={};self.rows=[];self.tree_records=[];self.history.clear()

    def end_chunk(self):
        if self.s._robotwin_spec:assert self.counters.get('committed_scheduler_update_count')==50
        self.chunk+=1;self.history.clear()
        return dict(counters=self.counters,phases_host_unsynchronized=self.phases,trees=self.tree_records,
                    shader=self.shader,shared_calls=self.shared_calls,chunk=self.chunk)
