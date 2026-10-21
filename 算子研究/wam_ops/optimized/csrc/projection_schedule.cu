// 诊断入口：同一个256×128投影tile，只改变CTA领取任务的方式。
// 不替换whole_layer；输入/输出与tile覆盖检查在外部worker中进行。
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cutlass/gemm/device/gemm.h>
#include <torch/extension.h>
#include "cutlass_gemm_mainloop.cuh"
#include "projection_tile.cuh"

using B = __nv_bfloat16;
using at::Tensor;
struct Shared {
    CutlassTile<256>::Storage tile;
    int task;
};

// mode0: 每tile一个CTA；mode1: 108CTA静态循环；mode2: 108CTA原子队列。
// 三者共用这个compiled kernel，寄存器和shared配置完全相同。
__global__ void projection(const B *a, const B *w, const B *bias, B *out,
                           int m, int n, int k, int *queue, int mode,
                           int *visits, int *owners) {
    extern __shared__ __align__(128) unsigned char raw[];
    auto &s = *reinterpret_cast<Shared *>(raw);
    const int tasks = ((m + 255) / 256) * ((n + 127) / 128);
    int next = blockIdx.x;
    while (true) {
        if (mode == 2) {
            if (threadIdx.x == 0) s.task = atomicAdd(queue, 1);
            __syncthreads();
            next = s.task;
        }
        if (next >= tasks) break;
        if (visits && threadIdx.x == 0) {
            atomicAdd(visits + next, 1);
            owners[next] = blockIdx.x;
        }
        cutlass_gemm_tile<256>(s.tile, a, w, bias, out, m, n, k, next, false);
        next += gridDim.x;
    }
}

void run(Tensor a, Tensor w, Tensor bias, Tensor out, Tensor queue, int mode,
         Tensor visits, Tensor owners) {
    const int k = w.size(1), n = w.size(0), m = a.numel() / k;
    const int tasks = ((m + 255) / 256) * ((n + 127) / 128);
    projection<<<mode == 0 ? tasks : 108, 256, sizeof(Shared),
                 at::cuda::getCurrentCUDAStream()>>>(
        (B *)a.data_ptr(), (B *)w.data_ptr(), (B *)bias.data_ptr(), (B *)out.data_ptr(),
        m, n, k, queue.data_ptr<int>(), mode,
        visits.numel() ? visits.data_ptr<int>() : nullptr,
        owners.numel() ? owners.data_ptr<int>() : nullptr);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::vector<int64_t> setup() {
    C10_CUDA_CHECK(cudaFuncSetAttribute(projection, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       sizeof(Shared)));
    cudaFuncAttributes a;
    C10_CUDA_CHECK(cudaFuncGetAttributes(&a, projection));
    int resident = 0;
    C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &resident, projection, 256, sizeof(Shared)));
    return {int64_t(sizeof(Shared)), a.numRegs, int64_t(a.localSizeBytes), resident, 256};
}

__global__ void mark(int64_t *ticks, int i) {
    if (threadIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        ticks[i] = t;
    }
}
void marker(Tensor ticks, int i) {
    mark<<<1,32,0,at::cuda::getCurrentCUDAStream()>>>(ticks.data_ptr<int64_t>(), i);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", &run);
    m.def("setup", &setup);
    m.def("marker", &marker);
}
