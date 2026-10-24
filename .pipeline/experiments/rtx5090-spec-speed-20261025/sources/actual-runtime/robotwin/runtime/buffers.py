"""O3 request-local seven rows; math and callbacks remain frozen O2.

All cached templates are unbound. Tensor storage is owned only by Workspace,
closed at action exit. Trace/teacher use original O2 instead of pooled views.
"""
import ast
import copy
import contextvars
import inspect
import threading
import time

import torch

import scalar
from template import clone_function

ACTIVE = contextvars.ContextVar('O3_action_workspace', default=None)
EVENTS = []
_lock = threading.RLock()
_builders = {}
_actions = {}
_counts = dict(action_templates=0, builder_templates=0, action_compile_sec=0., builder_compile_sec=0.)
_o2_action_counts = dict(action_templates=0,score_templates=0,template_compile_sec=0.,table_reads=0,error_readbacks=0,error_scalars=0)


def stats():
    return dict(_counts)


class Workspace:
    def __init__(self):
        self.closed=False
        self.generation=0
        self.key=None
        self.rows=self.directions=self.previous=None
        self.first_states=self.leaf_states=self.first_directions=self.second_directions=None
        self.full=self.channel_ids=None
        self.channel_key=self.full_key=None
        self.api=None
        self.counts=dict(storage_allocations=0,trees=0,verifier_view_reads=0,
                         batch_scatters=0,batch_compacts=0,channel_index_uploads=0)

    def prepare(self,root):
        if self.closed:raise RuntimeError('closed O3 workspace')
        key=(tuple(root.shape),root.dtype,root.device)
        if self.key!=key:
            self.rows=torch.empty((7,*root.shape),device=root.device,dtype=root.dtype)
            self.directions=torch.empty((6,*root.shape),device=root.device,dtype=root.dtype)
            self.previous=torch.empty_like(root)
            self.first_states=self.rows[1:3]
            self.leaf_states=self.rows[3:].view(2,2,*root.shape)
            self.first_directions=self.directions[:2]
            self.second_directions=self.directions[2:].view(2,2,*root.shape)
            self.key=key
            self.counts['storage_allocations']+=3
        self.generation+=1
        self.counts['trees']+=1

    def indices(self,server,device):
        channels=server._p0_action_channel_ids()
        key=(channels,device)
        if key!=self.channel_key:
            self.channel_ids=torch.tensor(channels,device=device)
            self.channel_key=key
            self.counts['storage_allocations']+=1
            self.counts['channel_index_uploads']+=1
        return channels,self.channel_ids

    def scatter(self,server,compact,template):
        if self.closed:raise RuntimeError('closed O3 workspace')
        if not isinstance(compact,torch.Tensor) or compact.ndim!=2:
            raise ValueError('compact_states must have shape [N,D]')
        if compact.shape[0]!=7:raise ValueError('O3 requires exactly seven candidate rows')
        expected=(1,int(server.job_config.action_dim),int(server.job_config.frame_chunk_size),int(server.action_per_frame),1)
        if not isinstance(template,torch.Tensor) or tuple(template.shape)!=expected:
            raise ValueError('O3 candidate template shape mismatch')
        channels,ids=self.indices(server,template.device)
        size=len(channels)*server.action_per_frame*server.job_config.frame_chunk_size
        if compact.shape[1]!=size:raise ValueError('O3 compact action dimension mismatch')
        shape=(7,*expected[1:]);key=(shape,template.dtype,template.device)
        if key!=self.full_key:
            self.full=torch.empty(shape,device=template.device,dtype=template.dtype)
            self.full_key=key;self.counts['storage_allocations']+=1
        self.full.zero_()
        self.full.index_copy_(1,ids,compact.reshape(7,len(channels),expected[2],expected[3],1).to(device=template.device,dtype=template.dtype))
        self.counts['batch_scatters']+=1
        return self.full

    def compact(self,server,full):
        if self.closed:raise RuntimeError('closed O3 workspace')
        expected=(7,int(server.job_config.action_dim),int(server.job_config.frame_chunk_size),int(server.action_per_frame),1)
        if not isinstance(full,torch.Tensor) or tuple(full.shape)!=expected:
            raise ValueError('O3 target direction shape mismatch')
        channels,ids=self.indices(server,full.device)
        result=full.index_select(1,ids).reshape(7,-1)
        self.counts['batch_compacts']+=1
        return result

    def close(self):
        self.closed=True
        self.rows=self.directions=self.previous=self.full=self.channel_ids=None
        self.first_states=self.leaf_states=self.first_directions=self.second_directions=None
        self.api=None


def _builder(original):
    key=original.build_k2_b2_tree
    with _lock:
        if key in _builders:return _builders[key]
        start=time.perf_counter()
        raw,tree=scalar._source(key);gold=copy.deepcopy(tree)
        changed=[]
        for node in tree.body[0].body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='first_states':
                changed.append((node,copy.deepcopy(node.value)))
                node.value.keywords.append(ast.keyword(arg='out',value=ast.parse('__o3_ws.first_states',mode='eval').body))
        ret=tree.body[0].body[-1]
        if not isinstance(ret,ast.Return) or not isinstance(ret.value,ast.Call):raise ValueError('original builder return changed')
        original_func=copy.deepcopy(ret.value.func)
        ret.value.func=ast.Name(id='__o3_tree',ctx=ast.Load())
        replacements={
            'root_state':'__o3_ws.rows[0].copy_(root_state.detach())',
            'root_previous_direction':'__o3_ws.previous.copy_(previous_direction.detach())',
            'first_directions':'__o3_ws.first_directions.copy_(first_directions.detach())',
            'first_states':'first_states.detach()',
            'second_directions':'torch.stack(second_rows, dim=0, out=__o3_ws.second_directions).detach()',
            'leaf_states':'torch.stack(leaf_rows, dim=0, out=__o3_ws.leaf_states).detach()'}
        original_kwargs={}
        for kw in ret.value.keywords:
            if kw.arg in replacements:
                original_kwargs[kw.arg]=copy.deepcopy(kw.value)
                kw.value=ast.parse(replacements[kw.arg],mode='eval').body
        if len(changed)!=1 or set(original_kwargs)!=set(replacements):raise ValueError('exact builder allocation sites required')
        restored=copy.deepcopy(tree)
        for node in restored.body[0].body:
            if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='first_states':
                node.value=changed[0][1]
        restored.body[0].body[-1].value.func=original_func
        for kw in restored.body[0].body[-1].value.keywords:
            if kw.arg in original_kwargs:kw.value=original_kwargs[kw.arg]
        if scalar._dump(restored)!=scalar._dump(gold):raise ValueError('builder reverse math/callback proof failed')
        # Compile an unbound code template, never a function holding a workspace.
        fn=scalar._compile(tree,raw,__o3_ws=None,__o3_tree=None)
        class PackedTree(original.K2B2Tree):
            def verifier_states(self):
                ws=self._workspace
                if ws.closed or self._generation!=ws.generation:
                    raise RuntimeError('O3 tree workspace was reused or closed')
                ws.counts['verifier_view_reads']+=1
                return ws.rows
        _builders[key]=(fn,PackedTree)
        _counts['builder_templates']+=1
        _counts['builder_compile_sec']+=time.perf_counter()-start
        return _builders[key]


class API:
    def __init__(self,original,ws):
        self.base=scalar.api(original)
        self.ws=ws
        fn,cls=_builder(original)
        def construct(**fields):
            obj=cls(**fields)
            object.__setattr__(obj,'_workspace',ws)
            object.__setattr__(obj,'_generation',ws.generation)
            return obj
        self.build=clone_function(fn,__o3_ws=ws,__o3_tree=construct)

    def build_k2_b2_tree(self,**kwargs):
        root=kwargs['root_state'];previous=kwargs['previous_direction']
        if not isinstance(root,torch.Tensor) or not isinstance(previous,torch.Tensor):
            raise TypeError('root_state and previous_direction must be tensors')
        if root.shape!=previous.shape or root.numel()==0:
            raise ValueError('root_state and previous_direction must have the same non-empty shape')
        self.ws.prepare(kwargs['root_state'])
        return self.build(**kwargs)

    def __getattr__(self,name):return getattr(self.base,name)


def api(original):
    ws=ACTIVE.get()
    if ws is None or ws.closed:raise RuntimeError('O3 API outside active request')
    if ws.api is None:ws.api=API(original,ws)
    return ws.api


def compact(server,full):
    ws=ACTIVE.get()
    if ws is None:raise RuntimeError('O3 target compact outside active request')
    return ws.compact(server,full)


def _compile_action(tree,raw,**overrides):
    gold=copy.deepcopy(tree);changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='target_compact':
            changed.append(copy.deepcopy(node.value));node.value=ast.parse('__o3_compact(self,target_full)',mode='eval').body
    if len(changed)!=1:raise ValueError('one original target compact site required')
    restored=copy.deepcopy(tree)
    for node in ast.walk(restored):
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='target_compact':node.value=changed[0]
    if scalar._dump(restored)!=scalar._dump(gold):raise ValueError('action compact reverse proof failed')
    return scalar._compile(tree,raw,**{**overrides,'__o2_api':api,'__o3_compact':compact})


_action_factory=clone_function(scalar.action_function,_actions=_actions,
                               _counts=_o2_action_counts,_compile=_compile_action)


def action_function(original):
    before=len(_actions);start=time.perf_counter()
    fn=_action_factory(original)
    if len(_actions)>before:
        _counts['action_templates']+=1;_counts['action_compile_sec']+=time.perf_counter()-start
    return fn


def prepare(original,original_api):
    scalar.action_function(original)
    scalar.api(original_api)
    action_function(original)
    _builder(original_api)
