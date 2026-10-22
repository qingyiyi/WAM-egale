// 就绪tile调度：只在启动时做一次grid.sync；节点之间使用完成计数发布数据。
// 第一版的普通节点等待完整输入节点，不宣称已实现逐行跨层流水。
#pragma once
#include "ready_graph.h"
static_assert(NB == 28 && CK0 == 15 && CV == 16 && F2 == 26 && OUT == 27);
using DeviceCounter = cuda::atomic_ref<unsigned, cuda::thread_scope_device>;
struct ReadyState {
    ReadyNode graph[NB];
    unsigned next[NB], finished[NB], completed;
    unsigned ffn2_first[48]; // 0未完成；1第一段BF16已写好；2第二段已领取。
};

__device__ void init_ready(const Params &p) {
    auto &r = *p.ready;
    const int prefix = int(p.info[0]);
    int group = (prefix % 16 == 0 && (prefix + 95) / 64 == (prefix + 79) / 64) ? 2 : 1;
    for (int cfg = 0; cfg < 2; ++cfg)
        for (int i = 0; i < 6; i += 2)
            if (p.rows[cfg * 7 + i] != p.rows[cfg * 7 + i + 1])
                group = 1;
    const int self_tiles = ((7 + group - 1) / group) * H;
    for (int n = 0; n < NB; ++n) {
        r.graph[n] = ready_node(n, p.b * p.c, WAM_CONTEXT_TILE_M, self_tiles,
                                p.freq != nullptr, p.affine);
        r.next[n] = r.finished[n] = 0;
    }
    r.completed = 0;
    for (int i = 0; i < 48; ++i)
        r.ffn2_first[i] = 0;
}

// 只有CTA的thread0扫描队列。领取失败便寻找别的就绪节点，不持有锁等待输入。
__device__ void claim_ready(const Params &p, Shared &s) {
    auto &r = *p.ready;
    const unsigned done = DeviceCounter(r.completed).load(cuda::memory_order_acquire);
    s.node = (done & needs(OUT)) ? -2 : -1;
    s.task = 0;
    if (s.node == -2)
        return;
    for (int priority = 0; priority < NB; ++priority) {
        const int n = ready_priority(priority);
        const auto spec = r.graph[n];
        if ((done & needs(n)) || (done & spec.dependencies) != spec.dependencies)
            continue;
        // FFN2第二段只领取已完成第一段的输出tile；不占着CTA等生产者。
        const unsigned first_count = n == F2 ? 48 : spec.tiles;
        auto next = DeviceCounter(r.next[n]);
        if (next.load(cuda::memory_order_relaxed) < first_count) {
            unsigned tile = next.fetch_add(1, cuda::memory_order_relaxed);
            if (tile < first_count) {
                s.node = n;
                s.task = tile;
                return;
            }
        }
        if (n == F2) {
            for (int tile = 0; tile < 48; ++tile) {
                unsigned expected = 1;
                if (DeviceCounter(r.ffn2_first[tile]).compare_exchange_strong(
                        expected, 2, cuda::memory_order_acquire, cuda::memory_order_relaxed)) {
                    s.node = n;
                    s.task = tile + 48;
                    return;
                }
            }
        }
    }
}

// 这里仅执行一个tile，禁止调用包含grid.sync的整阶段函数。
// 每种tile沿用原warp分工；不是给一个warp强行分配整个GEMM/attention。
__device__ void run_ready_tile(const Params &p, Shared &s, int node, int t) {
    constexpr int m = 224;
    constexpr int64_t values = int64_t(m) * D;
    switch (node) {
    case MOD: point_tile(p, TABLE, MOD, 0, 0, 0, values * 6, t); break;
    case N1: case N3:
        norm_tile(s, node == N1 ? p.x : bp(p, R2), p.a[node], nullptr, nullptr,
                    p.eps[node == N1 ? 0 : 2], false, true, t);
        break;
    case M1: case M3:
        point_tile(p, MODULATE, node, node == M1 ? N1 : N3, 0,
                     node == M1 ? 0 : 3, values, t);
        break;
    case SQ0: case SK0: case SV: case SO: case CQ0: case CO: {
        const int input = node == SO ? SA : node == CQ0 ? N2 : node == CO ? CA : M1;
        const int wi = node == SQ0 ? 3 : node == SK0 ? 5 : node == SV ? 7 :
                       node == SO ? 9 : node == CQ0 ? 13 : 19;
        cutlass_gemm_tile<64>(s.tile.cutlass_gemm.action, bp(p, input), (const B *)p.w[wi],
                               (const B *)p.w[wi + 1], bp(p, node), m, D, D, t, true);
        break;
    }
    case SQN: case SKN: case CQ: case CK: {
        const int input = node == SQN ? SQ0 : node == SKN ? SK0 : node == CQ ? CQ0 : CK0;
        const int wi = node == SQN ? 11 : node == SKN ? 12 : node == CQ ? 21 : 22;
        const int ei = node == SQN ? 3 : node == SKN ? 4 : node == CQ ? 5 : 6;
        rms_pair_tile(s, bp(p, input), bp(p, node), (const B *)p.w[wi], p.eps[ei],
                        2 * t, node == CK ? p.b * p.c : m);
        break;
    }
    case SQR: case SKR:
        if (p.freq)
            point_tile(p, ROPE, node, node == SQR ? SQN : SKN, 0, 0, values / 2, t);
        break;
    case SA: case CA: {
        const bool cross = node == CA;
        const auto layout = prepare_attention_dual(p, s, cross, t == 0);
        attention_dual_tile(p, s, cross, t, layout);
        break;
    }
    case R1: case R2: case OUT:
        point_tile(p, RESIDUAL, node, node == R1 ? -1 : node == R2 ? R1 : R2,
                     node == R1 ? SO : node == R2 ? CO : F2,
                     node == R1 ? 2 : node == R2 ? -1 : 5, values, t);
        break;
    case N2:
        if (p.affine)
            norm_tile(s, bp(p, R1), p.a[N2], p.w[1], (const float *)p.w[2],
                        p.eps[1], false, false, t);
        break;
    case CK0: case CV: {
        const int wi = node == CK0 ? 15 : 17;
        cutlass_gemm_tile<WAM_CONTEXT_TILE_M, 136>(s.tile.cutlass_gemm.context, p.context,
            (const B *)p.w[wi], (const B *)p.w[wi + 1], bp(p, node), p.b * p.c, D, D, t, false);
        break;
    }
    case F1: action_ffn1_tile(p, s, t); break;
    case ACT: point_tile(p, GELU, ACT, F1, 0, 0, int64_t(m) * F, t); break;
    case F2:
        ffn2_partition(s.tile.cutlass_gemm.ffn1, bp(p, ACT), (const B *)p.w[25], t % 48, t / 48);
        ffn2_commit(s.tile.cutlass_gemm.ffn1, (const B *)p.w[26], bp(p, F2), t % 48, t >= 48);
        break;
    }
}

__global__ void whole_layer_ready(const __grid_constant__ Params p) {
    extern __shared__ __align__(32) unsigned char memory[];
    auto &s = *reinterpret_cast<Shared *>(memory);
    auto g = cg::this_grid();
    layout(p, s);
    if (blockIdx.x == 0 && threadIdx.x == 0)
        init_ready(p);
    g.sync(); // 唯一的全grid屏障：发布layout和本次调用清零的任务状态。
    for (;;) {
        if (threadIdx.x == 0)
            claim_ready(p, s);
        __syncthreads(); // 发布本CTA领取到的node/task以及acquire的输入可见性。
        const int node = s.node, task = s.task;
        if (node == -2)
            return;
        if (node < 0) {
            if (threadIdx.x == 0)
                __nanosleep(128);
            __syncthreads();
            continue;
        }
        run_ready_tile(p, s, node, task);
        // 所有实际写线程先发布全局结果，再让thread0发布完成计数；仅thread0
        // threadfence不够。acq_rel计数串起所有tile，最后一个发布节点完成位。
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) {
            auto &r = *p.ready;
            if (node == F2 && task < 48)
                DeviceCounter(r.ffn2_first[task]).store(1, cuda::memory_order_release);
            const unsigned finished = DeviceCounter(r.finished[node]).fetch_add(
                                          1, cuda::memory_order_acq_rel) + 1;
            if (finished == unsigned(r.graph[node].tiles))
                DeviceCounter(r.completed).fetch_or(needs(node), cuda::memory_order_release);
        }
        __syncthreads(); // 所有warp离开当前tile后，shared union才能复用。
    }
}
