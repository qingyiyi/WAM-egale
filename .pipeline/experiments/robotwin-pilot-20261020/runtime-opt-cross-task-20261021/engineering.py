"""Isolated evaluation engineering options. Original fixed source files are read-only."""
import ast
import hashlib
import inspect
import json
import random
import textwrap
import time
from pathlib import Path
import numpy as np
import torch


def digest(value):
    h = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            a=x.detach().cpu().contiguous(); h.update(str((str(a.dtype),tuple(a.shape))).encode());h.update(a.view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray):
            h.update(str((x.dtype.str,x.shape)).encode());h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=str): add(k);add(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for y in x:add(y)
        elif isinstance(x,np.generic):add(x.item())
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    add(value);return h.hexdigest()


def rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch_CPU=torch.get_rng_state(),torch_CUDA=torch.cuda.get_rng_state_all())


def used_observation(obs):
    # Only front is absent from the optimized return; compare every remaining field.
    return {**obs,'observation':{k:v for k,v in obs['observation'].items() if k!='front_camera'}}


def install(variant,out):
    info={'variant':variant,'front_omitted':False,'batch_warmups_deferred':0,'batch_warmups_materialized':0}
    if variant in ('FRONT_ONLY','COMBINED'):
        from envs.camera.camera import Camera
        camera_load=Camera.load_camera
        def load_camera(self,*args,**kwargs):
            result=camera_load(self,*args,**kwargs)
            assert self.static_camera_name.count('front_camera')==1
            ids=[i for i,n in enumerate(self.static_camera_name) if n!='front_camera']
            # Keep the created camera alive to preserve scene entity construction/RNG.
            self._unused_front_camera=[c for c,n in zip(self.static_camera_list,self.static_camera_name) if n=='front_camera']
            self.static_camera_list=[self.static_camera_list[i] for i in ids]
            self.static_camera_config=[self.static_camera_config[i] for i in ids]
            self.static_camera_name=[self.static_camera_name[i] for i in ids]
            self.static_camera_num=len(ids)
            self.head_camera_id=self.static_camera_name.index('head_camera')
            info['front_omitted']=True
            return result
        Camera.load_camera=load_camera
    if variant in ('LAZY_BATCH_ONLY','COMBINED'):
        from envs.robot import planner
        cls=planner.CuroboPlanner
        original=cls.__init__
        source=textwrap.dedent(inspect.getsource(original))
        tree=ast.parse(source); hits=[]
        class Defer(ast.NodeTransformer):
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id=='super' and not node.args:
                    node.args=[ast.Name(id='CuroboPlanner',ctx=ast.Load()),ast.Name(id='self',ctx=ast.Load())]
                return self.generic_visit(node)
            def visit_Expr(self,node):
                if isinstance(node.value,ast.Call) and ast.unparse(node.value.func)=='self.motion_gen_batch.warmup':
                    hits.append(ast.unparse(node.value))
                    return ast.parse('self._batch_warmup_pending = True').body[0]
                return self.generic_visit(node)
        tree=ast.fix_missing_locations(Defer().visit(tree));assert len(hits)==1
        transformed=ast.unparse(tree)+'\n';(out/'GENERATED-PLANNER-INIT.py').write_text(transformed)
        namespace={};exec(compile(tree,str(out/'GENERATED-PLANNER-INIT.py'),'exec'),original.__globals__,namespace)
        replacement=namespace['__init__'];batch=cls.plan_batch
        def initialize(self,*args,**kwargs):
            result=replacement(self,*args,**kwargs);info['batch_warmups_deferred']+=1;return result
        def plan_batch(self,*args,**kwargs):
            if self._batch_warmup_pending:
                self.motion_gen_batch.warmup(batch=planner.CONFIGS.ROTATE_NUM)
                self._batch_warmup_pending=False;info['batch_warmups_materialized']+=1
            return batch(self,*args,**kwargs)
        cls.__init__=initialize;cls.plan_batch=plan_batch
    return info
