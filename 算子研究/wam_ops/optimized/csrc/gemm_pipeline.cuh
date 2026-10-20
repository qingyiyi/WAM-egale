// SM80 two-buffer GEMM, included inside the generated whole-layer namespace.
// Same 64x64x32 math / increasing K / FP32 accumulators / BF16 epilogue as v1.
// Warp 1 loads A, warp 2 loads W, warps 3..6 compute, warp 7 publishes output.
#pragma once

__device__ __forceinline__ void copy16_async(B* dst,const B* src,bool valid){
    const unsigned smem=static_cast<unsigned>(__cvta_generic_to_shared(dst));
    const int bytes=valid?16:0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 ::"r"(smem),"l"(src),"r"(bytes):"memory");
}

__device__ __forceinline__ void stage_gemm_copy(GemmShared& s,const B* a,
    const B* w,int rm,int cn,int m,int n,int k,int z,int buffer){
    // Every loader lane owns eight non-overlapping 16-byte destinations.
    if(warp()==1||warp()==2){
        for(int i=lane();i<64*32/8;i+=32){
            const int row=i/4,col=(i%4)*8;
            const bool valid=(warp()==1?rm+row<m:cn+row<n);
            const B* src=(warp()==1?a:w);
            // Invalid padded rows use an in-bounds pointer and zero source size.
            if(valid)src+=(warp()==1?rm+row:cn+row)*k+z+col;
            B* dst=(warp()==1?s.x[buffer]:s.w[buffer])+row*40+col;
            copy16_async(dst,src,valid);
        }
        asm volatile("cp.async.commit_group;":::"memory");
    }
}

__device__ __forceinline__ void wait_gemm_copy(){
    if(warp()==1||warp()==2)
        asm volatile("cp.async.wait_group 0;":::"memory");
}

__device__ void gemm_tile(Shared& shared,const B* a,const B* w,const B* bias,
                         B* out,int m,int n,int k,int task){
    auto& s=shared.tile.gemm;
    const int cols=(n+63)/64,rm=(task/cols)*64,cn=(task%cols)*64;
    wm::fragment<wm::accumulator,16,16,16,float> acc[4];
    if(compute())for(int j=0;j<4;++j)wm::fill_fragment(acc[j],0.f);

    stage_gemm_copy(s,a,w,rm,cn,m,n,k,0,0);
    wait_gemm_copy();
    __syncthreads(); // First buffer ready for all consumer warps.
    for(int z=0;z<k;z+=32){
        const int current=(z/32)&1,next=current^1;
        // The other buffer is free: its consumers finished at the prior barrier.
        if(z+32<k)stage_gemm_copy(s,a,w,rm,cn,m,n,k,z+32,next);
        if(compute()){
            int cw=warp()-3,m0=(cw/2)*32,n0=(cw%2)*32;
            #pragma unroll
            for(int t=0;t<2;++t){
                wm::fragment<wm::matrix_a,16,16,16,B,wm::row_major> af[2];
                wm::fragment<wm::matrix_b,16,16,16,B,wm::col_major> wf[2];
                #pragma unroll
                for(int j=0;j<2;++j){
                    wm::load_matrix_sync(af[j],s.x[current]+(m0+j*16)*40+t*16,40);
                    wm::load_matrix_sync(wf[j],s.w[current]+(n0+j*16)*40+t*16,40);
                }
                #pragma unroll
                for(int i=0;i<2;++i)for(int j=0;j<2;++j)
                    wm::mma_sync(acc[i*2+j],af[i],wf[j],acc[i*2+j]);
            }
        }
        wait_gemm_copy();
        // Both obligations: next input ready AND current buffer no longer read.
        __syncthreads();
    }
    if(compute()){
        int cw=warp()-3,m0=(cw/2)*32,n0=(cw%2)*32;
        for(int i=0;i<2;++i)for(int j=0;j<2;++j)
            wm::store_matrix_sync(s.y+(m0+i*16)*64+n0+j*16,acc[i*2+j],64,wm::mem_row_major);
    }
    __syncthreads();
    if(warp()==7)for(int i=lane();i<4096;i+=32){
        int r=rm+i/64,c=cn+i%64;
        if(r<m&&c<n)out[r*n+c]=bf(__fadd_rn(s.y[i],value(bias[c])));
    }
    __syncthreads();
}
