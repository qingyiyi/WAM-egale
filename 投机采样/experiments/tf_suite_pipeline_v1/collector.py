"""Bounded same-query supervision from complete Target or O3 policy chunks.

Only observation wrappers are installed. No model/teacher call, scheduler call,
RNG restore, trace flag, digest, simulator, or training operation is added.
The O3 methods are AST-extracted unchanged from the qualified earlier observer;
its source-digest admission and its dataset writer are deliberately not loaded.
Importing this module does not import torch or initialize CUDA.
"""
from __future__ import annotations

import ast
import copy
from datetime import datetime, timezone
import fcntl
import inspect
import json
import math
import os
from pathlib import Path
import sys
import textwrap
import types


SCHEMA = 'tf-suite-draft-chunk-columns/v1'
BUDGET_SCHEMA = 'tf-suite-compact-data-budget/v1'
CAP = 2**30
METADATA_RESERVE = 4*2**20
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
STAGES = {'T_POOL':'TARGET200', 'PARENT_DATA':'PARENT30', 'CURRENT_DATA':'CURRENT30',
    'TARGET200':'TARGET200', 'PARENT30':'PARENT30', 'CURRENT30':'CURRENT30'}
SOURCE_ENUM = {'canonical_target_scheduler_CFG_after':0,
    'O3_same_query_root_CFG_after':1, 'O3_same_query_parent_CFG_after':2}
ROLE_ENUM = {'FIT':0, 'CAL':1}
DTYPE_ENUM = {'torch.bfloat16':0, 'torch.float32':1}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def _frozen_observation_methods():
    """Reuse only observation bodies, never import/execute the old module."""
    path = Path(__file__).resolve().parent.parent/'tf_effect_collector_v1/collector.py'
    raw = path.read_text()
    tree = ast.parse(raw)
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='Collector']
    require(len(classes)==1, 'one earlier O3 observer class required')
    names = {'_check_mode','_workspace','_caller','_cpu','_same_bytes','install','_capture'}
    methods = [copy.deepcopy(n) for n in classes[0].body
        if isinstance(n, ast.FunctionDef) and n.name in names]
    require({n.name for n in methods}==names, 'qualified observation method inventory differs')
    require(not any(isinstance(n,ast.Call) and (
        isinstance(n.func,ast.Name) and n.func.id in ('digest','fingerprint') or
        isinstance(n.func,ast.Attribute) and n.func.attr in ('sha256','sha1','md5'))
        for method in methods for n in ast.walk(method)), 'unexpected digest operation in observer methods')
    definition = ast.ClassDef(name='_FrozenO3Methods', bases=[], keywords=[], body=methods,
        decorator_list=[],type_params=[])
    unit = ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[]))
    namespace = dict(globals())
    exec(compile(unit,str(path)+':observation-bodies-only-no-digest','exec'),namespace)
    return namespace['_FrozenO3Methods'], str(path)


_FrozenO3Methods, O3_OBSERVER_SOURCE = _frozen_observation_methods()


class _O3Core(_FrozenO3Methods):
    def __init__(self, server, proxy, torch):
        self.s, self.proxy, self.torch = server, proxy, torch
        self.installed = self.closed = False
        self.active = self.pending = None
        self.errors = []
        self.chunk_index = 0

    def _error(self, where, exc):
        item = dict(where=where,error=repr(exc),chunk_index=self.chunk_index)
        if len(self.errors)<32:
            self.errors.append(item)
        if self.active is not None:
            self.active['errors'].append(item)

    def _validate_source(self):
        raw = inspect.unwrap(type(self.s)._infer_action_verifier_k2_b2)
        tree = ast.parse(textwrap.dedent(inspect.getsource(raw)))
        calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute)]
        for name in ('_target_action_candidate_directions','execute_k2_b2_route'):
            require(sum(n.func.attr==name for n in calls)==1,
                'one native O3 observation callsite required: '+name)
        assigned = {n.id for n in ast.walk(tree) if isinstance(n,ast.Name) and isinstance(n.ctx,ast.Store)}
        require({'step_index','tree','scores','route','frame_st_id'} <= assigned|set(raw.__code__.co_varnames),
            'native O3 caller local names differ')
        self.source_proof = dict(function=raw.__name__,file=raw.__code__.co_filename,
            first_line=raw.__code__.co_firstlineno,admission='AST callsite/local inventory; no digest',
            observation_method_source=O3_OBSERVER_SOURCE,old_observation_bodies_AST_reused=True,
            digest_admission_executed=False)

    def close(self):
        if self.closed:
            return
        require(self.active is None and self.pending is None, 'cannot detach active O3 observer')
        if self.installed:
            require(self.s._target_action_candidate_directions==self.bound_candidate
                and self.proxy.execute_k2_b2_route==self.bound_execute, 'O3 observer ownership changed')
            if self.had_candidate:
                self.s._target_action_candidate_directions=self.original_candidate
            else:
                del self.s._target_action_candidate_directions
            if self.had_execute:
                self.proxy.execute_k2_b2_route=self.original_execute
            else:
                del self.proxy.execute_k2_b2_route
        self.closed=True


class _BoundedWriter:
    def __init__(self, stream, limit):
        self.stream,self.limit,self.written=stream,int(limit),0

    def write(self, value):
        require(self.written+len(value)<=self.limit, 'suite compact data cap reached')
        n=self.stream.write(value)
        self.written+=n
        return n

    def flush(self):
        self.stream.flush()


class Collector:
    """Bracket each existing complete policy with begin_chunk/end_chunk.

    Metadata requires suite/suite_short, phase or source_stage, and the common
    suite_storage_root shared by all stages/workers. The original context is
    retained separately from canonical row/task/init/role identity. Training
    columns contain reached nodes only; counterfactual rows are a separate dict.
    end_chunk never turns a data error into a robot mathematical exception.
    """
    def __init__(self, s, proxy, output_dir, metadata):
        import torch
        self.torch,self.s,self.proxy=torch,s,proxy
        self.metadata=copy.deepcopy(dict(metadata))
        short=str(self.metadata.get('suite_short',self.metadata.get('suite',''))).removeprefix('libero_')
        require(short in ('object','goal','spatial'), 'one registered small-suite identity required')
        self.suite_short,self.suite=short,'libero_'+short
        stage=self.metadata.get('source_stage',self.metadata.get('phase'))
        require(stage in STAGES, 'registered source stage required')
        self.stage=STAGES[stage]
        self.kind='T' if self.stage=='TARGET200' else 'O3'
        self.output=Path(output_dir).resolve()
        root=self.metadata.get('suite_storage_root')
        require(root is not None, 'explicit shared suite compact-data budget root required')
        self.root=Path(root).resolve()
        require(self.output.is_relative_to(self.root), 'collector output must be within shared suite compact root')
        self.root.mkdir(parents=True,exist_ok=True)
        self.output.mkdir(parents=True,exist_ok=True)
        self.cap=int(self.metadata.get('storage_cap_bytes',CAP))
        require(0<self.cap<=CAP, 'suite compact data cap must not exceed one GiB')
        require(self.cap>METADATA_RESERVE, 'positive compact data allowance required')
        self.active=self.core=None
        self.errors=[]
        self.chunk_index=self.total_rows=self.total_trees=0
        self.installed=self.closed=self.disabled=False
        self.scheduler=s.action_scheduler
        self._target_original=self._target_wrapper=None
        self._target_had_step=False
        self._target_old_value=None
        raw=inspect.unwrap(type(s)._infer)
        self.native_target_code=raw.__code__
        native_ast=ast.parse(textwrap.dedent(inspect.getsource(raw)))
        calls=[n for n in ast.walk(native_ast) if isinstance(n,ast.Call)
            and isinstance(n.func,ast.Attribute) and n.func.attr=='step'
            and isinstance(n.func.value,ast.Attribute) and n.func.value.attr=='action_scheduler']
        require(len(calls)==1, 'one native target action scheduler callsite required')
        self.target_source_proof=dict(file=raw.__code__.co_filename,
            first_line=raw.__code__.co_firstlineno,function=raw.__name__,
            admission='exact running native code object plus one action scheduler callsite; no digest')

    def install(self):
        require(not self.installed and not self.closed, 'one collector installation required')
        require(self.s._p0_state_teacher is None and self.s._p0_trace is None,
            'native teacher/trace must remain disabled')
        # No draft, O3 workspace or model is loaded by Target-only installation.
        self.installed=True
        return self

    def _cpu(self,value):
        if isinstance(value,self.torch.Tensor):
            return value.detach().cpu().contiguous().clone()
        if isinstance(value,dict):
            return {k:self._cpu(v) for k,v in value.items()}
        if isinstance(value,list):
            return [self._cpu(v) for v in value]
        if isinstance(value,tuple):
            return tuple(self._cpu(v) for v in value)
        if value is None or type(value) in (str,bool,int,float):
            return value
        raise TypeError('unsupported compact collector value: '+type(value).__name__)

    def _error(self,where,exc):
        item=dict(where=where,error=repr(exc),chunk_index=self.chunk_index)
        if len(self.errors)<32:
            self.errors.append(item)
        if self.active is not None:
            self.active['errors'].append(item)

    def _context(self,context):
        c=copy.deepcopy(dict(context))
        for key in ('row_id','case_id','task','init','arm','out','rollout_chunk','role','original_trace'):
            require(key in c,'missing actual request identity: '+key)
        require(all(type(c[k]) is int for k in ('row_id','case_id','task','init','rollout_chunk')),
            'integer actual request/task/init identity required')
        require(0<=c['task']<10 and 0<=c['init']<26 and c['rollout_chunk']>=0
            and c['case_id']==c['init']*10+c['task'], 'registered task/init/case identity differs')
        expected={'TARGET200':({*range(18)},{18,19}),
            'PARENT30':({20,21},{22}),'CURRENT30':({23,24},{25})}[self.stage]
        role='FIT' if c['init'] in expected[0] else 'CAL' if c['init'] in expected[1] else None
        require(role is not None and c['role']==role, 'episode-level FIT/CAL assignment differs')
        require(c['arm']==('T' if self.kind=='T' else 'C'), 'training collection arm differs')
        trace=c['original_trace']
        require(isinstance(trace,dict), 'original trace context required')
        for key,actual in (('task_index',c['task']),('episode_index',c['init']),('rollout_chunk',c['rollout_chunk'])):
            require(trace.get(key)==actual,'original native trace identity differs: '+key)
        json.dumps(c,ensure_ascii=False,allow_nan=False)
        return c

    def begin_chunk(self,context):
        require(self.installed and not self.closed and self.active is None, 'idle installed collector required')
        c=self._context(context)
        self.active=dict(context=c,records=[],errors=[],calls=0,previous=None,
            disabled_at_begin=self.disabled)
        if self.disabled:
            self.active['errors'].append(dict(where='begin',error='collector data capture disabled'))
            return
        try:
            require(self.s._p0_state_teacher is None and self.s._p0_trace is None,
                'do not change production path through native teacher/trace')
            if self.kind=='T':
                self._target_original=self.scheduler.step
                self._target_had_step='step' in vars(self.scheduler)
                self._target_old_value=vars(self.scheduler).get('step')
                self._target_wrapper=self._observe_target_step
                self.scheduler.step=self._target_wrapper
            else:
                require(self.proxy is not None, 'O3 collection requires the actual installed selection proxy')
                if self.core is None:
                    self.core=_O3Core(self.s,self.proxy,self.torch)
                    self.core.install()
                    self.core.bound_candidate=self.s._target_action_candidate_directions
                    self.core.bound_execute=self.proxy.execute_k2_b2_route
                require(not self.core.closed and self.core.pending is None, 'O3 core not idle')
                self.core.active=dict(context=copy.deepcopy(c['original_trace']),records=[],
                    candidate_calls=0,execute_calls=0,errors=[])
        except Exception as exc:
            self._error('begin_observation',exc)

    def _native_target_frame(self):
        frame=inspect.currentframe().f_back
        try:
            for _ in range(12):
                if frame is None:
                    break
                if frame.f_code is self.native_target_code and frame.f_locals.get('self') is self.s:
                    return frame
                frame=frame.f_back
        finally:
            # The returned caller reference is promptly released by its owner.
            pass
        raise ValueError('exact native target action frame not present')

    def _observe_target_step(self,*args,**kwargs):
        # Copy small consumed operands synchronously BEFORE the native step or
        # any later caller-side in-place prefix clamp. A failed native step
        # makes the enclosing chunk invalid; it is never admitted as complete.
        try:
            self._record_target(args,kwargs)
        except Exception as exc:
            self._error('target_scheduler_observation',exc)
        # Exactly one original call; observation failure never replaces output.
        return self._target_original(*args,**kwargs)

    def _record_target(self,args,kwargs):
        require(self.active is not None and self.kind=='T', 'Target step outside active declared chunk')
        t=self.torch
        model_output=args[0] if len(args)>0 else kwargs['model_output']
        timestep=args[1] if len(args)>1 else kwargs['timestep']
        sample=args[2] if len(args)>2 else kwargs['sample']
        frame=self._native_target_frame()
        try:
            local=frame.f_locals
            i=local['i']
            require(type(i) is int and 0<=i<50 and i==self.active['calls'], 'Target indices must cover0..49 in order')
            require(local['did_action_scheduler_update'] is True and local['last_step'] is False
                and local['model_source']=='target' and self.s._p0_action_draft_mode=='off',
                'not a canonical Target scheduler update')
            require(sample is local['actions'] and model_output is local['action_d_cfg']
                and timestep is local['t'], 'scheduler operands differ from actual CFG-after native caller')
            require(int(local['frame_st_id'])==int(self.s.frame_st_id), 'native frame index changed')
            require(str(sample.dtype) in DTYPE_ENUM and str(model_output.dtype) in DTYPE_ENUM,
                'record actual BF16/FP32 production operands; do not change inference dtype')
            require(sample.device==model_output.device==self.s.device,
                'actual Target action device differs')
            require(len(args)<=3 and not kwargs.get('to_final',False),
                'canonical update cannot force scheduler final')
            # Match the scheduler's actual nearest-table index; metadata only.
            value=timestep.detach().cpu() if isinstance(timestep,t.Tensor) else timestep
            index=int(t.argmin((self.scheduler.timesteps-value).abs()).item())
            require(index==i and len(self.scheduler.sigmas)==50, 'actual scheduler table/index differs')
            sigma=float(self.scheduler.sigmas[index])
            if index+1>=len(self.scheduler.timesteps):
                next_sigma=1.0 if self.scheduler.inverse_timesteps or self.scheduler.reverse_sigmas else 0.0
            else:
                next_sigma=float(self.scheduler.sigmas[index+1])
            current=self._cpu(self.s._p0_action_compact(sample[0]))
            target=self._cpu(self.s._p0_action_compact(model_output[0]))
            require(current.shape==target.shape==(112,), 'fixed seven-channel sixteen-token row required')
            require(bool(t.isfinite(current).all() and t.isfinite(target).all()), 'nonfinite actual Target row')
            previous=self.active['previous']
            if i>0:
                require(previous is not None and previous.shape==(112,), 'previous true target direction missing')
                self.active['records'].append(dict(state=current.float(),previous=previous.float(),
                    target=target.float(),sigma=sigma,next_sigma=next_sigma,timestep=float(value),
                    step=i,frame=int(local['frame_st_id']),node=-1,
                    source_dtype=dict(state=str(current.dtype),previous=str(previous.dtype),target=str(target.dtype))))
            self.active['previous']=target
            self.active['calls']+=1
        finally:
            del frame

    def _restore_target_observer(self):
        if self._target_wrapper is None:
            return
        require(self.scheduler.step==self._target_wrapper, 'Target observer ownership not restored by outer wrappers')
        if self._target_had_step:
            self.scheduler.step=self._target_old_value
        else:
            del self.scheduler.step
        self._target_wrapper=self._target_original=None

    def _common_columns(self,n,*,steps,nodes,trees,sources):
        t=self.torch;c=self.active['context'];episode=c['task']*100+c['init']
        ids=dict(task=t.full((n,),c['task'],dtype=t.int64),init=t.full((n,),c['init'],dtype=t.int64),
            episode=t.full((n,),episode,dtype=t.int64),
            rollout_chunk=t.full((n,),c['rollout_chunk'],dtype=t.int64),
            request_row_id=t.full((n,),c['row_id'],dtype=t.int64),
            step_index=steps,node=nodes,tree_index=trees,source=sources,
            role=t.full((n,),ROLE_ENUM[c['role']],dtype=t.int8))
        # Episode sets are disjoint across stages; suite names separate suites.
        local=t.arange(1,n+1,dtype=t.int64)
        ids['query']=episode*1000000+c['rollout_chunk']*10000+local
        return ids

    def _target_columns(self):
        t=self.torch;r=self.active['records'];n=len(r)
        require(self.active['calls']==50 and n==49, 'complete Target fifty updates/forty-nine rows required')
        steps=t.tensor([v['step'] for v in r],dtype=t.int64)
        require(steps.tolist()==list(range(1,50)), 'Target row coverage differs')
        cols=dict(state=t.stack([v['state'] for v in r]),previous=t.stack([v['previous'] for v in r]),
            target=t.stack([v['target'] for v in r]),
            scalars=t.tensor([[v['sigma'],v['next_sigma'],v['timestep']/1000.] for v in r],dtype=t.float32),
            timestep=t.tensor([v['timestep'] for v in r],dtype=t.float32),
            delta=t.tensor([abs(v['next_sigma']-v['sigma']) for v in r],dtype=t.float32),
            reached=t.ones(n,dtype=t.bool),state_reached=t.ones(n,dtype=t.bool),
            selected_branch=t.ones(n,dtype=t.bool),draft_edge_committed=t.zeros(n,dtype=t.bool),
            frame_st_id=t.tensor([v['frame'] for v in r],dtype=t.int64))
        cols.update(self._common_columns(n,steps=steps,nodes=t.full((n,),-1,dtype=t.int64),
            trees=t.full((n,),-1,dtype=t.int64),sources=t.zeros(n,dtype=t.int8)))
        for key in ('state','previous','target'):
            cols[key+'_source_dtype']=t.tensor([DTYPE_ENUM[v['source_dtype'][key]] for v in r],dtype=t.int8)
        return cols,{},dict(tree_count=0,raw_rows=49,counterfactual_rows=0,
            scheduler_calls_observed=50,first_step_retained_only_as_previous=True,
            teacher_path='canonical native action scheduler sample + CFG-after model_output',
            source_proof=self.target_source_proof,source_dtypes=r[0]['source_dtype'])

    def _o3_columns(self):
        t=self.torch;core=self.core;a=core.active
        require(a is not None and core.pending is None and not a['errors'] and not core.errors,
            'O3 observer errors or unconsumed candidate')
        r=a['records'];n=len(r)
        require(n>0 and a['candidate_calls']==a['execute_calls']==n, 'one-to-one O3 candidate/execute/tree coverage')
        for v in r:
            require(all(v['source_dtype'][k] in DTYPE_ENUM for k in ('state','previous','target','proposals')),
                'record actual BF16/FP32 O3 operands without changing production dtype')
        nodes=t.tensor([0,1,2],dtype=t.int64).repeat(n)
        parent=t.tensor([v['parent'] for v in r],dtype=t.int64).repeat_interleave(3)
        depths=t.tensor([v['depth'] for v in r],dtype=t.int64).repeat_interleave(3)
        selected=(nodes==0)|(nodes-1==parent)
        reached=(nodes==0)|(selected&(depths>=1))
        committed=((nodes==0)&(depths>=1))|((nodes>0)&selected&(depths==2))
        offsets=t.tensor([0,1,1],dtype=t.int64).repeat(n)
        steps=t.tensor([v['step'] for v in r],dtype=t.int64).repeat_interleave(3)+offsets
        cols=dict(state=t.cat([v['state'] for v in r]),previous=t.cat([v['previous'] for v in r]),
            target=t.cat([v['target'] for v in r]),proposals=t.cat([v['proposals'] for v in r]),
            scalars=t.tensor([[v['sigma'][o],v['sigma'][o+1],v['times'][o]/1000.]
                for v in r for o in (0,1,1)],dtype=t.float32),
            timestep=t.tensor([v['times'][o] for v in r for o in (0,1,1)],dtype=t.float32),
            delta=t.tensor([abs(v['sigma'][o+1]-v['sigma'][o]) for v in r for o in (0,1,1)],dtype=t.float32),
            reached=reached,state_reached=reached,selected_branch=selected,draft_edge_committed=committed,
            accepted_depth=depths,frame_st_id=t.tensor([v['frame'] for v in r],dtype=t.int64).repeat_interleave(3))
        cols.update(self._common_columns(3*n,steps=steps,nodes=nodes,
            trees=t.arange(n,dtype=t.int64).repeat_interleave(3),sources=t.where(nodes==0,1,2).to(t.int8)))
        for key in ('state','previous','target','proposals'):
            cols[key+'_source_dtype']=t.tensor([DTYPE_ENUM[v['source_dtype'][key]] for v in r],dtype=t.int8).repeat_interleave(3)
        factual={k:v[reached].contiguous() for k,v in cols.items()}
        counterfactual={k:v[~reached].contiguous() for k,v in cols.items()}
        return factual,counterfactual,dict(tree_count=n,raw_rows=3*n,
            counterfactual_rows=int((~reached).sum()),candidate_calls=a['candidate_calls'],execute_calls=a['execute_calls'],
            teacher_path='actual packed B14 target directions, after production CFG; scored tree root/parents',
            source_proof=core.source_proof,source_dtypes=r[0]['source_dtype'],
            trees=dict(sigma_path=t.tensor([v['sigma'] for v in r],dtype=t.float64),
                target_timesteps=t.tensor([v['times'] for v in r],dtype=t.float64),
                first_costs=t.tensor([v['first'] for v in r],dtype=t.float32),
                second_costs=t.tensor([v['second'] for v in r],dtype=t.float32),
                threshold=t.tensor([v['threshold'] for v in r],dtype=t.float64),
                selected_parent=t.tensor([v['parent'] for v in r],dtype=t.int64),
                selected_child=t.tensor([v['child'] for v in r],dtype=t.int64),
                accepted_depth=t.tensor([v['depth'] for v in r],dtype=t.int64),
                start_step=t.tensor([v['step'] for v in r],dtype=t.int64),
                O3_generation=t.tensor([v['generation'] for v in r],dtype=t.int64)))

    def _atomic_budget(self,path,data):
        raw=(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode()
        require(len(raw)<=256*2**10, 'bounded shared compact-data ledger')
        temp=path.with_suffix('.json.pending')
        with temp.open('wb') as f:
            f.write(raw)
        temp.replace(path)

    def _payload_size(self,value):
        if isinstance(value,self.torch.Tensor):
            return value.numel()*value.element_size()+1024
        if isinstance(value,dict):
            return sum(self._payload_size(v) for v in value.values())+1024
        if isinstance(value,(tuple,list)):
            return sum(self._payload_size(v) for v in value)+1024
        return len(str(value))*4+128

    def _save(self,payload):
        path=self.output/('chunk-%06d.pt'%self.chunk_index)
        partial=path.with_suffix('.pt.partial')
        ledger=self.root/'COMPACT-DATA-BUDGET.json'
        with (self.root/'.compact-data-budget.lock').open('a+b') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            if ledger.exists():
                require(ledger.stat().st_size<=256*2**10, 'oversized compact budget ledger')
                budget=json.loads(ledger.read_text())
                require(budget['schema']==BUDGET_SCHEMA and budget['suite']==self.suite
                    and budget['cap_bytes']==self.cap, 'shared suite budget identity differs')
            else:
                budget=dict(schema=BUDGET_SCHEMA,suite=self.suite,cap_bytes=self.cap,
                    committed_data_bytes=0,files=0,inflight=None,retained_interrupted_files=[])
            # The lock is held throughout writes. Any existing inflight owner
            # has therefore exited its writer; preserve its actual partial.
            prior=budget.get('inflight')
            if prior is not None:
                candidates=[Path(prior[k]) for k in ('partial','final')]
                require(all(p.is_relative_to(self.root) for p in candidates), 'prior compact writer escaped suite root')
                present=[p for p in candidates if p.is_file()]
                require(len(present)<=1, 'ambiguous prior compact writer files')
                size=sum(p.stat().st_size for p in present)
                budget['committed_data_bytes']+=size
                if len(budget['retained_interrupted_files'])<32:
                    budget['retained_interrupted_files'].append(dict(owner=prior,bytes=size,status='UNKNOWN_PRIOR_WRITER_INTERRUPTED'))
                budget['inflight']=None
            require(not path.exists() and not partial.exists(), 'immutable compact shard already exists')
            remaining=self.cap-METADATA_RESERVE-budget['committed_data_bytes']
            require(self._payload_size(payload)+128*2**10<=remaining, 'suite one-GiB compact cap: no shard admitted')
            budget['inflight']=dict(partial=str(partial),final=str(path),pid=os.getpid(),at=stamp())
            self._atomic_budget(ledger,budget)
            try:
                with partial.open('xb') as stream:
                    writer=_BoundedWriter(stream,remaining)
                    self.torch.save(payload,writer)
                    writer.flush()
                os.rename(partial,path)
                size=path.stat().st_size
                budget['committed_data_bytes']+=size
                budget['files']+=1
                budget['inflight']=None
                self._atomic_budget(ledger,budget)
                return path,size,budget['committed_data_bytes']
            except Exception:
                # Leave both ledger inflight and partial for the next locked
                # writer to reconcile. Neither is silently deleted or retried.
                raise

    def _invalid(self,exc):
        marker=self.output/'COLLECTOR-FAILED.json'
        receipt=dict(schema=SCHEMA,status='INVALID_COMPACT_CAPTURE',error=repr(exc),
            context=None if self.active is None else copy.deepcopy(self.active['context']),
            errors=self.errors,extra_teacher_forwards=0,robot_result_changed=False,
            training_admissible=False,performance_claim=False,permissions=PERMISSIONS)
        try:
            with marker.open('x') as f:
                json.dump(receipt,f,ensure_ascii=False,indent=2,allow_nan=False)
                f.write('\n')
        except FileExistsError:
            pass
        except Exception as write_error:
            receipt['failure_marker_write_error']=repr(write_error)
        receipt['path']=None
        receipt['failure_marker']=str(marker)
        return receipt

    def end_chunk(self):
        if self.active is None:
            return self._invalid(RuntimeError('no active compact collection chunk'))
        try:
            if self.kind=='T':
                self._restore_target_observer()
            require(not self.active['errors'] and not self.errors and not self.disabled,
                'compact capture errors invalidate data, not robot mathematics')
            columns,counterfactual,details=(self._target_columns() if self.kind=='T' else self._o3_columns())
            n=int(columns['state'].shape[0])
            require(n>0 and bool(columns['reached'].all()), 'main training columns must contain reached rows only')
            payload=dict(schema=SCHEMA,status='COMPLETE_POLICY_CHUNK_PENDING_EPISODE_AUDIT',
                suite=self.suite,suite_short=self.suite_short,source_stage=self.stage,
                phase=self.metadata.get('phase','UNKNOWN'),metadata=self.metadata,
                context=copy.deepcopy(self.active['context']),columns=columns,
                counterfactual_columns=counterfactual,details=details,
                column_enums=dict(source=SOURCE_ENUM,role=ROLE_ENUM,source_dtype=DTYPE_ENUM),
                rows=n,reached_rows=n,counterfactual_rows=details['counterfactual_rows'],
                teacher_path=details['teacher_path'],extra_teacher_forwards=0,
                original_O3_path_preserved=self.kind=='O3',native_trace_and_state_teacher_disabled=True,
                complete_episode_audited=False,training_admissible=False,
                default_training_rows='columns only; never counterfactual_columns',
                performance_claim=False,permissions=PERMISSIONS)
            path,size,total=self._save(payload)
            self.total_rows+=n;self.total_trees+=details['tree_count']
            return dict(path=str(path),bytes=size,status=payload['status'],rows=n,reached_rows=n,
                counterfactual_rows=details['counterfactual_rows'],trees=details['tree_count'],
                raw_rows=details['raw_rows'],suite=self.suite,source_stage=self.stage,
                context=copy.deepcopy(self.active['context']),suite_compact_data_bytes=total,
                extra_teacher_forwards=0,complete_episode_audited=False,training_admissible=False)
        except Exception as exc:
            self._error('end_chunk',exc)
            self.disabled=True
            return self._invalid(exc)
        finally:
            if self.core is not None:
                self.core.active=self.core.pending=None
                self.core.chunk_index+=1
                if self.disabled:
                    try:
                        self.core.close()
                    except Exception as exc:
                        self._error('detach_disabled_O3_observer',exc)
            self.active=None
            self.chunk_index+=1

    def abort_chunk(self,error):
        """Preserve a failed request without claiming a complete training shard."""
        if self.active is not None:
            self._error('original_policy_exception',error)
        return self.end_chunk()

    def close(self):
        if self.closed:
            return
        if self.active is not None:
            self.abort_chunk(RuntimeError('collector closed before complete policy return'))
        if self._target_wrapper is not None:
            try:
                self._restore_target_observer()
            except Exception as exc:
                self._error('close_target_observer',exc)
        if self.core is not None:
            try:
                self.core.close()
            except Exception as exc:
                self._error('close_O3_observer',exc)
        self.closed=True
