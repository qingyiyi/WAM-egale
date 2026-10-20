"""Read cuBLASLt public heuristic metadata on actual saved block activations.

Constants/struct ABI: installed CUDA 12.4 cublasLt.h. No kernel disassembly,
algorithm sweep, or replacement of Torch settings. Heuristic waves != occupancy.
"""
import ctypes as C
from datetime import datetime
import json
import os
from pathlib import Path
import time

import torch
from tools.block_loader import FIXTURES, load_block
from tools.fixtures import BlockFixture, decode
from tools.replay import device_tree
from wam_ops.optimized.native import NativeBlock


class Algo(C.Structure):
    _fields_=[('data',C.c_uint64*8)]


class Heuristic(C.Structure):
    _fields_=[('algo',Algo),('workspaceSize',C.c_size_t),('state',C.c_int),
              ('wavesCount',C.c_float),('reserved',C.c_int*4)]


def inspect(x,w,bias):
    lib=C.CDLL('libcublasLt.so.12')
    torchlib=C.CDLL(str(Path(torch.__file__).parent/'lib/libtorch_cuda.so'))
    def torchexport(name,result):
        fn=getattr(torchlib,f'_ZN2at4cuda{len(name)}{name}Ev');fn.argtypes=[];fn.restype=result
        return fn()
    workspace=torchexport('getCUDABlasLtWorkspaceSize',C.c_size_t)
    handle=torchexport('getCurrentCUDABlasLtHandle',C.c_void_p)
    def f(name,args):
        fn=getattr(lib,name);fn.argtypes=args;fn.restype=C.c_int
        def invoke(*values):
            status=fn(*values)
            if status:raise RuntimeError(f'{name}: status {status}')
        return invoke
    vp,sz=C.c_void_p,C.c_size_t
    op,pref=vp(),vp();layouts=[]
    m,k=x.numel()//x.shape[-1],x.shape[-1];n=w.shape[0]
    out=torch.empty((m,n),dtype=x.dtype,device=x.device)
    create=f('cublasLtMatrixLayoutCreate',[C.POINTER(vp),C.c_int,C.c_uint64,C.c_uint64,C.c_int64])
    setop=f('cublasLtMatmulDescSetAttribute',[vp,C.c_int,vp,sz])
    setpref=f('cublasLtMatmulPreferenceSetAttribute',[vp,C.c_int,vp,sz])
    def attr(fn,obj,key,value):fn(obj,key,C.byref(value),C.sizeof(value))
    def alignment(t):
        a=256
        while t.data_ptr()%a:a//=2
        return a
    try:
        f('cublasLtMatmulDescCreate',[C.POINTER(vp),C.c_int,C.c_int])(C.byref(op),68,0)
        f('cublasLtMatmulPreferenceCreate',[C.POINTER(vp)])(C.byref(pref))
        attr(setop,op,3,C.c_int(1));attr(setop,op,4,C.c_int(0))
        attr(setop,op,7,C.c_int(4));attr(setop,op,8,vp(bias.data_ptr()))
        for r,c,ld in ((k,n,k),(k,m,k),(n,m,n)):
            v=vp();create(C.byref(v),14,r,c,ld);layouts.append(v)
        attr(setpref,pref,1,sz(workspace))
        for key,t in ((5,w),(6,x),(7,out),(8,bias)):attr(setpref,pref,key,C.c_uint32(alignment(t)))
        if not torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction:
            attr(setpref,pref,3,C.c_uint32(2))
        h=Heuristic();count=C.c_int()
        f('cublasLtMatmulAlgoGetHeuristic',[vp,vp,vp,vp,vp,vp,vp,C.c_int,C.POINTER(Heuristic),C.POINTER(C.c_int)])(
            handle,op,layouts[0],layouts[1],layouts[2],layouts[2],pref,1,C.byref(h),C.byref(count))
        if count.value!=1 or h.state:raise RuntimeError('no valid matching heuristic')
        attrs={}
        get=f('cublasLtMatmulAlgoConfigGetAttribute',[C.POINTER(Algo),C.c_int,vp,sz,C.POINTER(sz)])
        for i,name in enumerate(('id','tile_id','splitK','reduction_scheme','CTA_swizzle','custom','stages_id')):
            val=C.c_int32();written=sz();get(C.byref(h.algo),i,C.byref(val),C.sizeof(val),C.byref(written));attrs[name]=val.value
        lib.cublasLtGetVersion.restype=sz
        return dict(M=m,N=n,K=k,workspace_preference=workspace,workspace_required=h.workspaceSize,
            heuristic_waves_not_measured=h.wavesCount,configuration=attrs,
            reduced_precision_allowed=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            library_version=lib.cublasLtGetVersion(),new_GEMM_executions=0,
            interpretation='public API reconstruction of pinned native descriptor; not interception of old call')
    finally:
        for v in layouts:f('cublasLtMatrixLayoutDestroy',[vp])(v)
        if pref:f('cublasLtMatmulPreferenceDestroy',[vp])(pref)
        if op:f('cublasLtMatmulDescDestroy',[vp])(op)


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    os.sched_setaffinity(0,{4,5});torch.set_num_threads(2)
    if torch.version.git_version!='0fabc3ba44823f257e70ce397d989c8de5e362c1':raise RuntimeError('pinned interpreter required')
    args.output.mkdir(exist_ok=False)
    result=dict(status='RUNNING',rows=[],full_block_calls=0,heuristic_queries=0,full30=0,episodes=0)
    start=time.monotonic()
    try:
        block=load_block('cuda');native=NativeBlock(block)
        with torch.inference_mode():
            for batch in (2,14):
                fixture=BlockFixture.load(FIXTURES,'short',batch)
                fixture=BlockFixture(fixture.path,device_tree(fixture.data));fixture.install_cache(block)
                a,k=fixture.arguments();debug=native(*a,**k,debug=True);result['full_block_calls']+=1
                for name,x,layer in [('Q',debug['modulated1'],block.attn1.to_q),
                                     ('FFN1',debug['modulated3'],block.ffn.net[0].proj),
                                     ('FFN2',debug['gelu'],block.ffn.net[2]),
                                     ('crossK',a[1],block.attn2.to_k)]:
                    result['rows'].append(dict(batch=batch,name=name,**inspect(x,layer.weight,layer.bias)))
                    result['heuristic_queries']+=1
        torch.cuda.synchronize();result['status']='COMPLETE_PUBLIC_API_METADATA_READ'
    except BaseException as exc:
        import traceback
        result.update(status='FAILED_OR_PARTIAL',error=repr(exc),traceback=traceback.format_exc())
    finally:
        result.update(lifecycle_seconds=time.monotonic()-start,at=datetime.now().astimezone().isoformat())
        (args.output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
