// SM80 stage-local persistent CTA scheduling. No global barriers or spin waits.
#pragma once

namespace wam_tiles {
// Each CTA owns one residue class modulo gridDim.x: no atomics are needed to
// claim work and every task is visited exactly once. All warps compute together.
struct PersistentTasks {
    int count;
    __device__ int first() const { return int(blockIdx.x); }
    __device__ int next(int task) const { return task + int(gridDim.x); }
};

__device__ __forceinline__ void finish_tile() {
    // The copied mainloops use cp.async. Drain outstanding copies before the
    // next tile aliases the same shared storage, including uniform early exits.
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
}

__device__ __forceinline__ void record_tile(int* audit, int task) {
    if (audit && threadIdx.x == 0) atomicAdd(audit + task, 1);
}

__device__ __forceinline__ void record_worker(int* audit, int count, int done) {
    if (audit && threadIdx.x == 0) audit[count + blockIdx.x] = done;
}
} // namespace wam_tiles
