"""生成测量入口；数学函数来自当前optimized，原版阶段来自冻结baseline。"""
from pathlib import Path
import shutil
import json
import hashlib

B = Path(__file__).resolve().parent
R = Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
N = R.parent / 'baseline'
for name in ['source-before', 'generated', 'build', 'tmp', 'cache']:
    (B/name).mkdir(exist_ok=True)
source_files=['whole_layer.py','build_whole_layer.py','csrc/whole_layer.cu',
              'csrc/cutlass_gemm.cuh','csrc/gemm_pipeline.cuh','csrc/fa2_stage.cuh',
              'csrc/ffn_pipeline.cuh']
manifest=[]
for name in source_files:
    p=B/'source-before'/name;p.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(R/name,p)
    manifest.append(dict(path=name,SHA256=hashlib.sha256(p.read_bytes()).hexdigest()))
for p in R.glob('wam_whole_fa2_cuda*.so'):
    shutil.copy2(p,B/'source-before'/p.name)
(B/'SOURCE-BEFORE.json').write_text(json.dumps(manifest,indent=2)+'\n')

stages=[
 ('table','point(p,s,g,0,TABLE,MOD,0,0,0,int64_t(m)*6*D);','modulation'),
 ('norm1','norm(p,s,g,1,p.x,p.a[N1],m,-1,-1,p.eps[0],false,true);','norm1'),
 ('modulate1','point(p,s,g,2,MODULATE,M1,N1,0,0,int64_t(m)*D);','modulated1'),
 ('self_Q','linear(p,s,g,3,bp(p,M1),3,bp(p,SQ0),m,D,D);','self_q_linear'),
 ('self_K','linear(p,s,g,4,bp(p,M1),5,bp(p,SK0),m,D,D);','self_k_linear'),
 ('self_V','linear(p,s,g,5,bp(p,M1),7,bp(p,SV),m,D,D);','self_v'),
 ('self_Q_RMS','norm(p,s,g,6,bp(p,SQ0),p.a[SQN],m,11,-1,p.eps[3],true,false);','self_q_norm'),
 ('self_K_RMS','norm(p,s,g,7,bp(p,SK0),p.a[SKN],m,12,-1,p.eps[4],true,false);','self_k_norm'),
 ('self_Q_RoPE','point(p,s,g,8,ROPE,SQR,SQN,0,0,int64_t(m)*D/2);','self_q_rope'),
 ('self_K_RoPE','point(p,s,g,9,ROPE,SKR,SKN,0,0,int64_t(m)*D/2);','self_k_rope'),
 ('cache_insert','insert_cache(p,s,g,10);',None),
 ('self_attention','attention(p,s,g,11,false);','self_attention'),
 ('self_out','linear(p,s,g,12,bp(p,SA),9,bp(p,SO),m,D,D);','self_out'),
 ('residual1','point(p,s,g,13,RESIDUAL,R1,-1,SO,2,int64_t(m)*D);','residual1'),
 ('norm2','norm(p,s,g,14,bp(p,R1),p.a[N2],m,1,2,p.eps[1],false,false);','norm2'),
 ('cross_Q','linear(p,s,g,15,bp(p,N2),13,bp(p,CQ0),m,D,D);','cross_q_linear'),
 ('cross_K','linear(p,s,g,16,p.context,15,bp(p,CK0),mc,D,D);','cross_k_linear'),
 ('cross_V','linear(p,s,g,17,p.context,17,bp(p,CV),mc,D,D);','cross_v'),
 ('cross_Q_RMS','norm(p,s,g,18,bp(p,CQ0),p.a[CQ],m,21,-1,p.eps[5],true,false);','cross_q'),
 ('cross_K_RMS','norm(p,s,g,19,bp(p,CK0),p.a[CK],mc,22,-1,p.eps[6],true,false);','cross_k'),
 ('cross_attention','attention(p,s,g,20,true);','cross_attention'),
 ('cross_out','linear(p,s,g,21,bp(p,CA),19,bp(p,CO),m,D,D);','cross_out'),
 ('residual2','point(p,s,g,22,RESIDUAL,R2,R1,CO,-1,int64_t(m)*D);','residual2'),
 ('norm3','norm(p,s,g,23,bp(p,R2),p.a[N3],m,-1,-1,p.eps[2],false,true);','norm3'),
 ('modulate3','point(p,s,g,24,MODULATE,M3,N3,0,3,int64_t(m)*D);','modulated3'),
 ('FFN1','linear(p,s,g,25,bp(p,M3),23,bp(p,F1),m,F,D);','ffn1'),
 ('GELU','point(p,s,g,26,GELU,ACT,F1,0,0,int64_t(m)*F);','gelu'),
 ('FFN2','linear(p,s,g,27,bp(p,ACT),25,bp(p,F2),m,D,F);','ffn2'),
 ('residual3','point(p,s,g,28,RESIDUAL,OUT,R2,F2,5,int64_t(m)*D);','output'),
 ('layout','layout(p);g.sync();',None),
]
(B/'STAGES.json').write_text(json.dumps([dict(id=i,name=n,output=k) for i,(n,_,k) in enumerate(stages)],indent=2)+'\n')
s=(R/'csrc/whole_layer.cu').read_text()
s=s.replace('int *queues,*visits,*stats,*ffn_counts;',
            'int *queues,*visits,*stats,*ffn_counts; int64_t* stage_ticks;')
start=s.index('// All threads read the same immutable launch descriptors.')
end=s.index('\nstd::map<std::string,Tensor> forward',start)
kernel=r'''
__device__ inline int64_t diagnostic_ns(){
    unsigned long long t;asm volatile("mov.u64 %0, %%globaltimer;":"=l"(t));return t;
}
template<int Stage> __device__ void execute_stage(const Params& p,Shared& s,cg::grid_group& g){
    const int m=p.b*p.q,mc=p.b*p.c;
    STAGE_BRANCHES
}
template<int Stage> __device__ void timed_stage(const Params& p,Shared& s,cg::grid_group& g){
    if(g.thread_rank()==0)p.stage_ticks[2*Stage]=diagnostic_ns();
    execute_stage<Stage>(p,s,g);
    if(g.thread_rank()==0)p.stage_ticks[2*Stage+1]=diagnostic_ns();
}
template<int Stage> __global__ void one_stage(const __grid_constant__ Params p){
    extern __shared__ __align__(32) unsigned char memory[];
    auto& s=*reinterpret_cast<Shared*>(memory);auto g=cg::this_grid();
    execute_stage<Stage>(p,s,g);
}
__global__ void whole_layer(const __grid_constant__ Params p){
    extern __shared__ __align__(32) unsigned char memory[];
    auto& s=*reinterpret_cast<Shared*>(memory);auto g=cg::this_grid();
    for(int i=g.thread_rank();i<MAX_STAGES*3;i+=g.size())p.stats[i]=0;
    timed_stage<29>(p,s,g);
    WHOLE_STAGES
    g.sync();
}
__global__ void marker_kernel(int64_t* times,int index){
    if(threadIdx.x==0)times[index]=diagnostic_ns();
}
void marker(Tensor times,int index){
    marker_kernel<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>(times.data_ptr<int64_t>(),index);
}
'''
kernel=kernel.replace('STAGE_BRANCHES','\n    '.join(
    f'if constexpr(Stage=={i}){{{stmt}}}' for i,(_,stmt,_) in enumerate(stages)))
kernel=kernel.replace('WHOLE_STAGES','\n    '.join(f'timed_stage<{i}>(p,s,g);' for i in range(29)))
s=s[:start]+kernel+s[end:]
s=s.replace('bool audit,bool bf16_splits,bool tile_pipeline){',
            'bool audit,bool bf16_splits,bool tile_pipeline,int selected_stage,const std::vector<Tensor>& preload,const std::vector<Tensor>& meta){')
s=s.replace('for(int i=0;i<NB;++i){int r=', 'for(int i=0;i<NB;++i){if(!preload.empty()){b[i]=preload[i];p.a[i]=b[i].data_ptr();continue;}int r=')
s=s.replace('    p.slots=slots.data_ptr<int64_t>();',
            '    if(!meta.empty()){slots=meta[0];info=meta[1];}\n    p.slots=slots.data_ptr<int64_t>();')
s=s.replace('    auto queues=at::empty({MAX_STAGES},io)',
            '    auto ticks=at::empty({30,2},lo);p.stage_ticks=ticks.data_ptr<int64_t>();\n    auto queues=at::empty({MAX_STAGES},io)')
launch_start=s.index('    C10_CUDA_CHECK(cudaFuncSetAttribute(whole_layer,')
launch_end=s.index('    C10_CUDA_KERNEL_LAUNCH_CHECK();',launch_start)
dispatch='void* kernel=(void*)whole_layer;\n    switch(selected_stage){'+''.join(f'case {i}:kernel=(void*)one_stage<{i}>;break;' for i in range(30))+'}\n'
s=s[:launch_start]+'    '+dispatch+r'''
    C10_CUDA_CHECK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,kernel,THREADS,sizeof(Shared)));
    TORCH_CHECK(active>0,"stage cannot reside");
    // Same 108 resident-CTA dispatch as the full kernel. This is not a tile/grid sweep.
    int blocks=sm;void* args[]={&p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel(kernel,dim3(blocks),dim3(THREADS),args,sizeof(Shared),at::cuda::getCurrentCUDAStream()));
'''+s[launch_end:]
s=s.replace('cudaFuncGetAttributes(&attr,whole_layer)','cudaFuncGetAttributes(&attr,kernel)')
s=s.replace('    result["tile_stats"]=stats;','    result["stage_ticks"]=ticks;result["tile_stats"]=stats;')
s=s.replace('m.def("forward",&wam_whole_layer_v1::forward);','m.def("forward",&wam_whole_layer_v1::forward);\n    m.def("marker",&wam_whole_layer_v1::marker);')
(B/'stage_owned.cu').write_text(s)

# 原版阶段的数学函数原样复制；只添加测量入口，不改baseline源码/二进制。
n=(N/'csrc/block.cu').read_text()
n=n[:n.index('PYBIND11_MODULE')]
n+=r'''
namespace wam_block {
Tensor stage(int id,const std::vector<Tensor>& w,const Tensor& x,const Tensor& context,
 const Tensor& temb,const Tensor& freq,const std::vector<Tensor>& a,
 const std::vector<Tensor>& cache,const Tensor& rows,const Tensor& slots,int valid,
 const std::vector<double>& eps){
 const int b=x.size(0),q=x.size(1),c=context.size(1);
 auto hidden=[&](const Tensor& t){return t.view({b,q,3072});};
 switch(id){
 case 0:return table(temb,w[0],q);
 case 1:return norm(fp32(x),{},{},eps[0],false);
 case 2:return modulate(a[1],a[0],0);
 case 3:return linear(a[2],w[3],w[4]);
 case 4:return linear(a[2],w[5],w[6]);
 case 5:return linear(a[2],w[7],w[8]);
 case 6:return norm(a[3],w[11],{},eps[3],true);
 case 7:return norm(a[4],w[12],{},eps[4],true);
 case 8:return rope(a[6],freq,b,q);
 case 9:return rope(a[7],freq,b,q);
 case 10:return x; // B14 readonly, no insert.
 case 11:{
   const int total=valid+q;
   auto k=at::empty({b,total,24,128},x.options()),v=at::empty_like(k);
   // Same native materialization as full forward; layout measured separately.
   pack_kernel<<<grid(k.numel()/8),256,0,stream()>>>(cache[0].data_ptr<BF>(),a[9].data_ptr<BF>(),k.data_ptr<BF>(),
     slots.data_ptr<int64_t>(),rows.data_ptr<int64_t>(),b,total,valid,q,cache[0].size(1),3072,true);
   pack_kernel<<<grid(v.numel()/8),256,0,stream()>>>(cache[1].data_ptr<BF>(),a[5].data_ptr<BF>(),v.data_ptr<BF>(),
     slots.data_ptr<int64_t>(),rows.data_ptr<int64_t>(),b,total,valid,q,cache[0].size(1),3072,true);
   return wam_sdpa::sdpa(a[8].view({b,q,24,128}),k,v,std::nullopt);}
 case 12:return linear(hidden(a[10]),w[9],w[10]);
 case 13:return residual(x,hidden(a[11]),a[0],2);
 case 14:return bf16(norm(fp32(a[12]),fp32(w[1]),fp32(w[2]),eps[1],false));
 case 15:return linear(a[13],w[13],w[14]);
 case 16:return linear(context,w[15],w[16]);
 case 17:return linear(context,w[17],w[18]);
 case 18:return norm(a[14],w[21],{},eps[5],true);
 case 19:return norm(a[15],w[22],{},eps[6],true);
 case 20:return wam_sdpa::sdpa(a[17].view({b,q,24,128}),a[18].view({b,c,24,128}),a[16].view({b,c,24,128}),std::nullopt);
 case 21:return linear(hidden(a[19]),w[19],w[20]);
 case 22:return residual(hidden(a[12]),hidden(a[20]),{},-1);
 case 23:return norm(fp32(a[21]),{},{},eps[2],false);
 case 24:return modulate(a[22],a[0],3);
 case 25:return linear(a[23],w[23],w[24]);
 case 26:return wam_block::gelu(a[24]);
 case 27:return linear(a[25],w[25],w[26]);
 case 28:return residual(hidden(a[21]),hidden(a[26]),a[0],5);
 case 29:{
   auto free=at::empty({q},slots.options()),out=at::empty_like(slots),info=at::empty({2},slots.options());
   layout_kernel<<<1,128,0,stream()>>>(cache[3].data_ptr<bool>(),cache[2].data_ptr<int64_t>(),
    cache[4].data_ptr<bool>(),out.data_ptr<int64_t>(),free.data_ptr<int64_t>(),info.data_ptr<int64_t>(),
    cache[0].size(1),q,0,true,rows.data_ptr<int64_t>(),b,cache[0].size(0));
   return out;}
 }
 TORCH_CHECK(false,"Unknown measured stage");
}
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("stage",&wam_block::stage);}
'''
(B/'stage_native.cu').write_text(n)
print('Prepared 30 stages; cache insert is N/A for B14.')
