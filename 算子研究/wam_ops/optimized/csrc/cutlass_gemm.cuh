// CUTLASS 3.5.1 SM80 device mainloop, embedded in the existing whole-layer launch.
// No device::Gemm invocation, child kernel, host BLAS call, or new arithmetic.
#pragma once

using CutlassB = cutlass::bfloat16_t;
using CutlassRow = cutlass::layout::RowMajor;
using CutlassCol = cutlass::layout::ColumnMajor;
using CutlassConfig = cutlass::gemm::device::Gemm<
    CutlassB, CutlassRow, CutlassB, CutlassCol, CutlassB, CutlassRow, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64,128,64>,
    cutlass::gemm::GemmShape<32,64,64>,
    cutlass::gemm::GemmShape<16,8,16>,
    cutlass::epilogue::thread::LinearCombination<CutlassB,8,float,float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3>;
using CutlassBaseMma = typename CutlassConfig::GemmKernel::Mma;
using CutlassMma = cutlass::gemm::threadblock::WamMmaMultistage<
    typename CutlassBaseMma::Shape, typename CutlassBaseMma::IteratorA,
    typename CutlassBaseMma::SmemIteratorA, CutlassBaseMma::kCacheOpA,
    typename CutlassBaseMma::IteratorB, typename CutlassBaseMma::SmemIteratorB,
    CutlassBaseMma::kCacheOpB, float, CutlassRow,
    typename CutlassBaseMma::Policy, 3>;
union CutlassGemmStorage {
    typename CutlassMma::SharedStorage mainloop;
    float result[64*128];
};

__device__ void cutlass_gemm_tile(CutlassGemmStorage& s,
    const __nv_bfloat16* a,const __nv_bfloat16* w,const __nv_bfloat16* bias,
    __nv_bfloat16* out,int m,int n,int k,int task,bool bf16_splits){
    const int rm=(task/((n+127)/128))*64,cn=(task%((n+127)/128))*128;
    int splits=1;
    if(bf16_splits && n==3072){
        if(k==14336)splits=m==224?2:m==256?4:1;
        else if(k==3072 && m==224)splits=2;
    }
    const int span=k/splits,physical_warp=threadIdx.x/32,lane=threadIdx.x%32;
    for(int partition=0;partition<splits;++partition){
        if(physical_warp>=3 && physical_warp<7){
            const int thread=int(threadIdx.x)-96,warp=physical_warp-3;
            typename CutlassMma::IteratorA::Params ap{CutlassRow(k)};
            typename CutlassMma::IteratorB::Params bp{CutlassCol(k)};
            typename CutlassMma::IteratorA ai(ap,
                reinterpret_cast<CutlassB*>(const_cast<__nv_bfloat16*>(a))+partition*span,
                {m,span},thread,{rm,0});
            typename CutlassMma::IteratorB bi(bp,
                reinterpret_cast<CutlassB*>(const_cast<__nv_bfloat16*>(w))+partition*span,
                {span,n},thread,{0,cn});
            CutlassMma mma(s.mainloop,thread,warp,lane);
            typename CutlassMma::FragmentC acc;
            acc.clear();
            mma(span/64,acc,ai,bi,acc);
            // All compute warps finish reading the input ring before aliasing it.
            asm volatile("bar.sync 2, 128;" ::: "memory");
            typename CutlassMma::Operator::IteratorC store({s.result,CutlassRow(128)},lane);
            store.add_tile_offset({warp%2,warp/2});
            store.store(acc);
        }
        __syncthreads();
        if(physical_warp==7)for(int i=lane;i<64*128;i+=32){
            const int r=rm+i/128,c=cn+i%128;
            if(r<m && c<n){
                float z=s.result[i];
                if(partition==splits-1)z=__fadd_rn(z,__bfloat162float(bias[c]));
                if(partition>0)z=__fadd_rn(z,__bfloat162float(out[r*n+c]));
                out[r*n+c]=__float2bfloat16_rn(z);
            }
        }
        __syncthreads();
    }
}
