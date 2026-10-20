"""整理当前核心源码；先输出候选文件，--apply 时才写入 optimized。"""
from pathlib import Path
import argparse,subprocess,json,hashlib,shutil

parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true');args=parser.parse_args()
B=Path(__file__).resolve().parent
R=Path('/rjs/cxz/论文探索/算子研究/wam_ops/optimized')
out=B/'simplified-source';(out/'csrc').mkdir(parents=True,exist_ok=True)
old=(R/'csrc/whole_layer.cu').read_text()
s=old.replace('// Research-only complete SINGLE WAM block in ONE cooperative persistent kernel.\n// SM80: no TMA/WGMMA, host GEMM, dynamic parallelism, or cross-CTA shared memory.\n// Numerical admission is intentionally separate: WMMA != cuBLASLt bitwise order.',
'''// 单层 Transformer：一次 cooperative launch，CTA 常驻并领取各阶段的 tile。
// 固定实验输入为 BF16 hidden/context，D=3072、24 heads、head_dim=128。
// 输入校验与结果核验在外部实验脚本；这里直接呈现计算和必要同步。''')
s=s.replace('#include <cuda/atomic>\n','')
s=s.replace('constexpr int MAX_STAGES=40;','constexpr int MAX_STAGES=32; // 29个顺序阶段，30/31用于attention split合并。')
s=s.replace('enum Point {TABLE,MODULATE,RESIDUAL,ROPE,GELU,INSERT,RESTORE};',
            'enum Point {TABLE,MODULATE,RESIDUAL,ROPE,GELU};')
s=s.replace('int *queues,*visits,*stats,*ffn_counts;','int *queues;')
s=s.replace('int b,q,c,l,fb,pool,pool_b,update,max_tasks;','int b,q,c,l,fb,pool,update;')
s=s.replace('bool temb_float,cache,candidate,affine,audit,bf16_splits,tile_pipeline;',
            'bool temb_float,cache,candidate,affine;')
a=s.index('struct AttnShared {');z=s.index('union alignas(32) TileShared',a);s=s[:a]+s[z:]
s=s.replace('AttnShared attn;float point[512];','')
a=s.index('// Coarse stage barriers');z=s.index('// Small-M:',a)
s=s[:a]+'''// 每个阶段有一个全局计数器；一个CTA一次领取一个tile，做完再领。
// 阶段开始/结束的grid.sync保证所有前置输出完成；目前还不是跨阶段ready队列。
__device__ void begin_stage(const Params& p,cg::grid_group& grid,int stage){
    if(grid.thread_rank()==0)p.queues[stage]=0;
    grid.sync();
}
__device__ int claim_tile(const Params& p,Shared& s,int stage){
    if(threadIdx.x==0)s.task=atomicAdd(p.queues+stage,1);
    __syncthreads(); // 将warp0领到的编号交给整个CTA。
    return s.task;
}
__device__ void finish_tile(){
    __syncthreads(); // 共享workspace不可在上一tile的读写结束前被复用。
}

'''+s[z:]
s=s.replace('begin(p,s,g,st,ntiles)','begin_stage(p,g,st)').replace('claim(p,s,st,ntiles)','claim_tile(p,s,st)')
s=s.replace('begin(p,s,g,st,rows)','begin_stage(p,g,st)').replace('claim(p,s,st,rows)','claim_tile(p,s,st)')
s=s.replace('begin(p,s,g,st,tiles)','begin_stage(p,g,st)').replace('claim(p,s,st,tiles)','claim_tile(p,s,st)')
s=s.replace('done(p,st)','finish_tile()').replace('p.bf16_splits','true')
s=s.replace('    if(p.pool>4096){p.info[1]=1;return;}\n','')
s=s.replace('    if(p.candidate){\n        for(int b=0;b<p.b;++b)if(p.rows[b]<0||p.rows[b]>=p.pool_b)p.info[1]=2;\n    }else{','    if(!p.candidate){')
s=s.replace('        if(need&&(p.pool!=2160||used<=128)){p.info[1]=3;return;}\n','')
s=s.replace('        if(n!=p.q){p.info[1]=4;return;}\n','')
s=s.replace('int free=0,used=0;for(int i=0;i<p.pool;++i){if(p.mask[i])++used;else++free;}',
            'int free=0;for(int i=0;i<p.pool;++i)if(!p.mask[i])++free;')
s=s.replace('p.cache&&!p.candidate&&p.info[1]==0','p.cache&&!p.candidate')
s=s.replace('#include "ffn_pipeline.cuh"\n','')
s=s.replace('    // Workspace audit initialization is inside this same computational launch.\n    for(int i=g.thread_rank();i<MAX_STAGES*p.max_tasks;i+=g.size())if(p.audit)p.visits[i]=0;\n    for(int i=g.thread_rank();i<MAX_STAGES*3;i+=g.size())p.stats[i]=0;\n','')
s=s.replace('    if(p.tile_pipeline){ffn_subgraph(p,s,g,st,m);st+=3;}\n    else{\n','')
s=s.replace('        linear(p,s,g,st++,bp(p,ACT),25,bp(p,F2),m,D,F);\n    }',
            '    linear(p,s,g,st++,bp(p,ACT),25,bp(p,F2),m,D,F);')
comments={
 '    layout(p);g.sync();int st=0,m=p.b*p.q,mc=p.b*p.c;':
 '    // 缓存只整理本次可见slot；B14不会把候选的private KV写入公共历史。\n    layout(p);g.sync();int st=0,m=p.b*p.q,mc=p.b*p.c;\n    // 1. 按token生成6组调制量，得到self-attention的输入。',
 '    linear(p,s,g,st++,bp(p,M1),3,bp(p,SQ0),m,D,D);':
 '    // 2. Self attention：Q/K/V投影→Q/K的RMSNorm与RoPE→FA2→输出投影。\n    linear(p,s,g,st++,bp(p,M1),3,bp(p,SQ0),m,D,D);',
 '    if(p.affine)norm(p,s,g,st++,bp(p,R1),p.a[N2],m,1,2,p.eps[1],false,false);':
 '    // 3. Cross attention：Q来自当前hidden；K/V只来自context。\n    // 当前依次执行；K/V尚未提前作为ready任务填入其他阶段的尾波。\n    if(p.affine)norm(p,s,g,st++,bp(p,R1),p.a[N2],m,1,2,p.eps[1],false,false);',
 '    norm(p,s,g,st++,bp(p,R2),p.a[N3],m,-1,-1,p.eps[2],false,true);':
 '    // 4. FFN：3072→14336→3072；保留GEMM和GELU之间原有BF16舍入。\n    norm(p,s,g,st++,bp(p,R2),p.a[N3],m,-1,-1,p.eps[2],false,true);',
 '    point(p,s,g,st++,RESIDUAL,OUT,R2,F2,5,int64_t(m)*D);':
 '    // 5. 门控残差与临时KV状态恢复，输出仍为[B,Q,3072]。\n    point(p,s,g,st++,RESIDUAL,OUT,R2,F2,5,int64_t(m)*D);'
}
for a,z in comments.items():
 assert a in s,a;s=s.replace(a,z,1)
s=s.replace('// Same vector-4 per-thread input order and Welford tree as the pinned Torch\n// helper. Logical compute coordinates are explicitly remapped to four warps;\n// its group barrier has exactly 128 participants, never a partial CTA barrier.',
'''// Norm的tile是一整行3072维：warp1搬入，warp3–6按原Torch顺序做128线程归约，
// warp7写回。shared中的FP32转换/归约顺序不能随意改，否则BF16输出可能翻转。''')
s=s.replace('    constexpr int tile_elements=2048;\n    int tiles=', '    // 一个任务覆盖2048元素；目前4个compute warp各自处理步长128的元素。\n    constexpr int tile_elements=2048;\n    int tiles=',1)
s=s.replace('// Warp 0 claims CTA tasks. Warps 3..6 compute; small-M GEMM uses warps 1/2\n// as loaders. Large-M GEMM loads within its compute group, then all eight\n// warps cooperate in the epilogue. Norm/small-M still use warp 7 to store.',
'''// CTA内8个warp的固定分工：0领取tile；3–6计算；Norm/小M由7写回。
// 小M的1/2分别加载A/W；大M的CUTLASS由计算warp自行加载，epilogue用全部8warp。
// a[]是中间buffer，w[]对应whole_layer.py绑定的原始参数，均不跨候选混用。''')
start=s.index('std::map<std::string,Tensor> forward(')
s=s[:start]+r'''
// Host只绑定指针、分配workspace并发射一次kernel，不执行GEMM/attention。
// 调用方提供固定实验中的合法输入；shape/dtype/RNG/KV检查放在外部脚本。
std::map<std::string,Tensor> forward(
    const std::vector<Tensor>& w,const Tensor& x,const Tensor& context,
    const Tensor& temb,const std::optional<Tensor>& freq,
    const std::vector<Tensor>& cache,const std::optional<Tensor>& rows,
    int update,const std::vector<double>& eps){
    c10::cuda::CUDAGuard guard(x.device());
    Params p{};
    p.b=x.size(0);p.q=x.size(1);p.c=context.size(1);p.l=temb.size(1);p.update=update;
    p.x=(const B*)x.data_ptr();p.context=(const B*)context.data_ptr();
    p.temb=temb.data_ptr();p.temb_float=temb.scalar_type()==at::kFloat;
    for(int i=0;i<27;++i)p.w[i]=w[i].data_ptr();
    for(int i=0;i<7;++i)p.eps[i]=eps[i];
    p.affine=w[1].numel()!=0;
    if(freq){p.freq=freq->data_ptr<c10::complex<float>>();p.fb=freq->size(0);}
    if(!cache.empty()){
        p.cache=true;
        p.pk=(B*)cache[0].data_ptr();p.pv=(B*)cache[1].data_ptr();
        p.ids=cache[2].data_ptr<int64_t>();p.mask=cache[3].data_ptr<bool>();
        p.pred=cache[4].data_ptr<bool>();p.pool=cache[0].size(1);
        p.candidate=rows.has_value();
        if(rows)p.rows=rows->data_ptr<int64_t>();
    }

    // 所有中间量按原BF16/FP32边界落地；当前版本没有把整层中间量全部留在shared。
    const auto bo=x.options(),fo=bo.dtype(at::kFloat),lo=bo.dtype(at::kLong),io=bo.dtype(at::kInt);
    std::vector<Tensor> b(NB);
    for(int i=0;i<NB;++i){
        const int rows=(i==CK0||i==CK||i==CV)?p.b*p.c:p.b*p.q;
        if(i==MOD)b[i]=at::empty({rows,6,D},fo);
        else b[i]=at::empty({rows,(i==F1||i==ACT)?F:D},(i==N1||i==N3)?fo:bo);
        p.a[i]=b[i].data_ptr();
    }
    if(!p.freq){b[SQR]=b[SQN];b[SKR]=b[SKN];p.a[SQR]=p.a[SQN];p.a[SKR]=p.a[SKN];}
    if(!p.affine){b[N2]=b[R1];p.a[N2]=p.a[R1];}

    // slot表和任务计数属于执行所需状态；逐tile visits/stats审计已移出核心。
    auto slots=at::empty({std::max(p.pool,1)},lo),free=at::empty({p.q},lo),info=at::empty({2},lo);
    auto queues=at::empty({MAX_STAGES},io);
    p.slots=slots.data_ptr<int64_t>();p.free=free.data_ptr<int64_t>();
    p.info=info.data_ptr<int64_t>();p.queues=queues.data_ptr<int>();
    p.sms=at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    p.attn_scale=float(1.0/std::sqrt(128.0));
    p.attn_scale_log2=float(double(p.attn_scale)*M_LOG2E);

    // FA2仅在小batch时需要split-KV；B14正常路径num_splits=1。
    const int max_splits=(p.b*H*((p.q+63)/64)>=0.8f*p.sms*2)?1:
        std::min(128,(std::max(p.pool+p.q,p.c)+127)/128);
    auto lse=at::empty({p.b,H,p.q},fo),partial_lse=at::empty({max_splits,p.b,H,p.q},fo);
    auto partial_out=at::empty({max_splits,p.b,H,p.q,HD},fo),fa_meta=at::empty({2,4},lo);
    p.fa_lse=lse.data_ptr<float>();p.fa_lse_partial=partial_lse.data_ptr<float>();
    p.fa_out_partial=partial_out.data_ptr<float>();p.fa_meta=fa_meta.data_ptr<int64_t>();

    // Cooperative grid必须全部驻留，否则grid.sync无法完成；按实际资源计算CTA数量。
    int active=0;
    C10_CUDA_CHECK(cudaFuncSetAttribute(whole_layer,cudaFuncAttributeMaxDynamicSharedMemorySize,sizeof(Shared)));
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,whole_layer,THREADS,sizeof(Shared)));
    const int blocks=p.sms*active;void* launch_args[]={&p};
    C10_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)whole_layer,dim3(blocks),dim3(THREADS),
        launch_args,sizeof(Shared),at::cuda::getCurrentCUDAStream()));

    // debug只暴露已经存在的中间量，外部脚本据此核验，不在kernel里重复检查。
    const char* names[NB]={"modulation","norm1","modulated1","self_q_linear","self_k_linear",
        "self_v","self_q_norm","self_k_norm","self_q_rope","self_k_rope","self_attention",
        "self_out","residual1","norm2","cross_q_linear","cross_k_linear","cross_v",
        "cross_q","cross_k","cross_attention","cross_out","residual2","norm3","modulated3",
        "ffn1","gelu","ffn2","output"};
    std::map<std::string,Tensor> result;
    for(int i=0;i<NB;++i)result[names[i]]=b[i];
    result["output"]=b[OUT].view(x.sizes());
    result["cache_info"]=info;result["logical_slots"]=slots;result["attention_meta"]=fa_meta;
    cudaFuncAttributes attr;C10_CUDA_CHECK(cudaFuncGetAttributes(&attr,whole_layer));
    result["resources"]=at::tensor({int64_t(blocks),int64_t(active),int64_t(sizeof(Shared)),
        int64_t(attr.numRegs),int64_t(attr.localSizeBytes),int64_t(THREADS)},at::TensorOptions().dtype(at::kLong));
    return result;
}
} // namespace wam_whole_layer_v1
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("forward",&wam_whole_layer_v1::forward);
    m.attr("production_admitted")=false;
    m.attr("computation_launches")=1;
    m.attr("context_tile_rows")=WAM_CONTEXT_TILE_M;
    m.attr("point_tile_elements")=2048;
    m.attr("cooperative_epilogue")=true;
}
'''
(out/'csrc/whole_layer.cu').write_text(s)
fa=(R/'csrc/fa2_stage.cuh').read_text()
fa=fa.replace('begin(p,s,g,st,count)','begin_stage(p,g,st)').replace('claim(p,s,st,count)','claim_tile(p,s,st)').replace('done(p,st)','finish_tile()')
fa=fa.replace('begin(p,s,g,combine_stage,combine_count)','begin_stage(p,g,combine_stage)').replace('claim(p,s,combine_stage,combine_count)','claim_tile(p,s,combine_stage)').replace('done(p,combine_stage)','finish_tile()')
fa=fa.replace('// Device FA2 stages inside the existing cooperative whole-layer kernel.\n// GEMM\'s loader/compute/store specialization is unchanged. During attention,\n// warps 3..6 execute the pinned four-warp FA2 load/MMA/softmax/store mainloop.',
'''// Self/cross attention共用FA2设备主循环，计算由warp3–6完成。
// 每个任务属于一个(batch,head,query tile)，B14仍是14个独立Q16实例。
// prefix按合法slot读取，private suffix只属于自己的candidate，不能跨分支注意。''')
(out/'csrc/fa2_stage.cuh').write_text(fa)
for name in ['cutlass_gemm.cuh','gemm_pipeline.cuh']:
 text=(R/'csrc'/name).read_text()
 if name=='cutlass_gemm.cuh':
  text=text.replace('// CUTLASS 3.5.1 SM80 device mainloop, embedded in the existing whole-layer launch.\n// Preserve the recorded split/bias arithmetic; no child kernel or host BLAS.',
'''// 一个任务计算C中的TM×128输出tile，沿K每次推进64，CUTLASS内部三级流水。
// warp3–6共同执行cp.async/MMA；结果经shared交给全部8个warp做成对写回。
// action用TM64，context用TM128；保留原cuBLASLt的分段BF16与bias加法顺序。''')
 else:
  pos=text.index('#pragma once')
  text='''// GEMM设备入口：大M交给CUTLASS；B2 action的小M32保留原来的64×64/K32路径。
// 小M路径的warp1/2加载A/W，warp3–6计算，warp7写回；双buffer交替复用。
// split归约对应已观测的原库算术顺序，不可作为一般的调参开关删除。
'''+text[pos:]
 (out/'csrc'/name).write_text(text)
wrapper='''"""单层Transformer的一次persistent launch。固定实验输入；请在inference_mode下调用。"""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(None)
def extension():
    path = next(Path(__file__).parent.glob("wam_whole_fa2_cuda*.so"))
    spec = importlib.util.spec_from_file_location("wam_whole_fa2_cuda", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WholeLayerBlock:
    """绑定原参数和KV；数学计算都在csrc/whole_layer.cu中。"""
    production_admitted = False

    def __init__(self, block):
        # 只复用参数列表的绑定，不调用NativeBlock.forward。
        from .native import NativeBlock
        binding = NativeBlock(block)
        self.weights, self.eps = binding.weights, binding.eps
        self.block = block

    def __call__(self, hidden_states, encoder_hidden_states, temb, rotary_emb,
                 update_cache=0, cache_name="pos", candidate_cache_rows=None,
                 debug=False):
        cache = self.block.attn1.attn_caches.get(cache_name)
        values = [] if cache is None else [
            cache[key] for key in ("k", "v", "id", "mask", "is_pred")
        ]
        result = extension().forward(
            self.weights, hidden_states, encoder_hidden_states, temb, rotary_emb,
            values, candidate_cache_rows, update_cache, self.eps,
        )
        # 中间量仅供外部核验；普通调用返回与hidden同形状的输出。
        return result if debug else result["output"]
'''
(out/'whole_layer.py').write_text(wrapper)
build=(R/'build_whole_layer.py').read_text()
build=build.replace("if context_rows not in (64,128):\n    raise ValueError('Only the two registered context tile shapes are supported')\n",'')
(out/'build_whole_layer.py').write_text(build)
for p in (out/'csrc').glob('*'):
 subprocess.run(['clang-format','-i','--style={BasedOnStyle: LLVM, SortIncludes: false, IndentWidth: 4, ColumnLimit: 100, AllowShortFunctionsOnASingleLine: Inline, AllowShortIfStatementsOnASingleLine: Never}',str(p)],check=True)
manifest=[]
for p in out.rglob('*'):
 if p.is_file():
  rel=p.relative_to(out);manifest.append(dict(path=str(rel),SHA256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size))
  if args.apply:shutil.copy2(p,R/rel)
if args.apply:
 # 只移除当前核心不再引用的旧负结果路径；历史源码和结果保留。
 old_ffn=R/'csrc/ffn_pipeline.cuh'
 shutil.copy2(old_ffn,B/'ffn_pipeline-before-removal.cuh')
 old_ffn.unlink()
(B/'SIMPLIFIED-SOURCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('Applied' if args.apply else 'Prepared',len(manifest),'files')
