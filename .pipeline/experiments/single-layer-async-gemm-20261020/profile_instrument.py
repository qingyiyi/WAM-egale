"""Generate two diagnostic-only copies; primary optimized source stays frozen.

One copy adds PTX globaltimer nanosecond stage marks. The other changes ONLY
GEMM shared leading dimension 32 -> 40 BF16 (valid WMMA ldm multiple-of-eight).
No parameter/threshold/math policy or scheduler change. No pad/grid sweep.
"""
import hashlib
import json
from pathlib import Path

BASE=Path('/root/cxz/ops/whole-layer-warp-20261020')
SOURCE=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized/csrc')
SHA='df8b386f302b9904ac4ef8ab08d171ef6d9e6f8143451831c3318f5e0835ef0e'


def replace(source,before,after,count=1):
    assert source.count(before)==count,(before,source.count(before),count)
    return source.replace(before,after)


def instrument(source):
    source=replace(source,'int *queues,*visits,*stats;','int *queues,*visits,*stats; int64_t* stage_ticks;')
    source=replace(source,'__device__ inline int warp(){',
        '__device__ inline int64_t diagnostic_ns(){ unsigned long long t; '
        'asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }\n'
        '__device__ inline int warp(){')
    source=replace(source,'    grid.sync();\n}',
        '    grid.sync(); if(grid.thread_rank()==0)p.stage_ticks[st*2]=diagnostic_ns();\n}')
    for name in ('linear','norm','point','insert_cache','attention'):
        start=source.index('__device__ void '+name+'(')
        brace=source.index('{',start);depth=1;end=brace+1
        while depth:
            if source[end]=='{':depth+=1
            elif source[end]=='}':depth-=1
            end+=1
        body=source[start:end]
        body=replace(body,'g.sync();','g.sync();if(g.thread_rank()==0)p.stage_ticks[st*2+1]=diagnostic_ns();')
        source=source[:start]+body+source[end:]
    source=replace(source,'    layout(p);g.sync();int st=0,m=p.b*p.q,mc=p.b*p.c;',
        '    for(int i=g.thread_rank();i<MAX_STAGES*2;i+=g.size())p.stage_ticks[i]=0;\n'
        '    layout(p);g.sync();int st=0,m=p.b*p.q,mc=p.b*p.c;')
    start=source.index('__global__ void whole_layer(');brace=source.index('{',start);depth=1;end=brace+1
    while depth:
        if source[end]=='{':depth+=1
        elif source[end]=='}':depth-=1
        end+=1
    body=source[start:end]
    body=replace(body,'    // Workspace audit initialization',
        '    if(g.thread_rank()==0)p.stage_ticks[MAX_STAGES*2]=diagnostic_ns();\n'
        '    // Workspace audit initialization')
    body=replace(body,'    g.sync();','    g.sync();if(g.thread_rank()==0)p.stage_ticks[MAX_STAGES*2+1]=diagnostic_ns();')
    source=source[:start]+body+source[end:]
    source=replace(source,'    p.queues=queues.data_ptr<int>();p.stats=stats.data_ptr<int>();p.visits=visits.data_ptr<int>();',
        '    p.queues=queues.data_ptr<int>();p.stats=stats.data_ptr<int>();p.visits=visits.data_ptr<int>();\n'
        '    auto ticks=at::empty({MAX_STAGES+1,2},lo);p.stage_ticks=ticks.data_ptr<int64_t>();')
    source=replace(source,'    result["tile_stats"]=stats;result["cache_info"]=info;',
        '    result["tile_stats"]=stats;result["cache_info"]=info;result["stage_ticks"]=ticks;')
    return source


def pad(source):
    source=replace(source,'B x[64*32],w[64*32];','B x[64*40],w[64*40];')
    source=replace(source,'((uint4*)s.x)[i]=','*(uint4*)(s.x+r*40+c)=')
    source=replace(source,'((uint4*)s.w)[i]=','*(uint4*)(s.w+r*40+c)=')
    source=replace(source,'s.x+(m0+j*16)*32+t*16,32','s.x+(m0+j*16)*40+t*16,40')
    source=replace(source,'s.w+(n0+j*16)*32+t*16,32','s.w+(n0+j*16)*40+t*16,40')
    return source


def main():
    raw=(SOURCE/'whole_layer.cu').read_bytes();assert hashlib.sha256(raw).hexdigest()==SHA
    root=BASE/'diagnostic';root.mkdir(exist_ok=False)
    norm=(SOURCE/'norm.cuh').read_text()
    (root/'LICENSE.pytorch').write_bytes((SOURCE/'LICENSE.pytorch').read_bytes())
    records=[]
    for label,padded in [('profile',False),('pad40',True)]:
        ns='wam_whole_layer_diagnostic_'+label
        value=instrument(raw.decode())
        if padded:value=pad(value)
        value=value.replace('wam_whole_layer_v1',ns).replace('wam_optimized_norm','wam_norm_diagnostic_'+label)
        value=value.replace('"norm.cuh"','"norm_'+label+'.cuh"')
        path=root/(label+'.cu');path.write_text(value)
        (root/('norm_'+label+'.cuh')).write_text(norm.replace('wam_optimized_norm','wam_norm_diagnostic_'+label))
        records.append(dict(label=label,namespace=ns,CUDA_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            source_SHA256=SHA,changes=['stage PTX globaltimer ns marks']+(['GEMM shared ldm32->40'] if padded else [])))
    (root/'SOURCE-MANIFEST.json').write_text(json.dumps(records,indent=2)+'\n')
    print('generated two diagnostic-only namespaces; primary source/DSO untouched')


if __name__=='__main__':main()
