// B14/Q16的实际依赖图。编号与whole_layer.cu的Buf一致；不改变tile数学体。
#pragma once
#include <cstdint>
#ifdef __CUDACC__
#define WAM_GRAPH_HD __host__ __device__
#else
#define WAM_GRAPH_HD
#endif

struct ReadyNode {
    uint32_t dependencies;
    int tiles;
};
WAM_GRAPH_HD constexpr uint32_t needs(int node) { return uint32_t(1) << node; }

// hidden链优先；只有可领任务耗尽时，才提前做独立context投影。
WAM_GRAPH_HD constexpr int ready_priority(int i) {
    return i < 15 ? i : i < 26 ? i + 2 : i == 26 ? 15 : 16;
}
WAM_GRAPH_HD inline ReadyNode ready_node(int node, int context_rows, int context_tile_rows,
                                        int self_tiles, bool rotary, bool affine) {
    constexpr int rows = 224, width = 3072;
    constexpr int point = rows * width / 2048; // 336个point tile，与已验证路径一致。
    switch (node) {
    case 0: return {0, point * 6};                         // modulation table
    case 1: return {0, rows};                              // norm1
    case 2: return {needs(0) | needs(1), point};             // modulated1
    case 3: case 4: case 5: return {needs(2), 96};          // self Q/K/V
    case 6: return {needs(3), rows / 2};                    // self Q RMS
    case 7: return {needs(4), rows / 2};                    // self K RMS
    case 8: return {needs(6), rotary ? point / 2 : 1};       // Q RoPE / alias
    case 9: return {needs(7), rotary ? point / 2 : 1};       // K RoPE / alias
    case 10: return {needs(8) | needs(9) | needs(5), self_tiles};
    case 11: return {needs(10), 96};                        // self out
    case 12: return {needs(0) | needs(11), point};           // residual1
    case 13: return {needs(12), affine ? rows : 1};          // norm2 / alias
    case 14: return {needs(13), 96};                        // cross Q
    case 15: case 16: return {0, (context_rows + context_tile_rows - 1) /
                                       context_tile_rows * 24}; // context K/V
    case 17: return {needs(14), rows / 2};                  // cross Q RMS
    case 18: return {needs(15), (context_rows + 1) / 2};     // cross K RMS
    case 19: return {needs(17) | needs(18) | needs(16), 168}; // cross attention
    case 20: return {needs(19), 96};                        // cross out
    case 21: return {needs(12) | needs(20), point};          // residual2
    case 22: return {needs(21), rows};                      // norm3
    case 23: return {needs(0) | needs(22), point};           // modulated3
    case 24: return {needs(23), 112};                       // FFN1
    case 25: return {needs(24), rows * 14336 / 2048};        // GELU
    case 26: return {needs(25), 96};                        // FFN2: 48 tiles × 2 parts
    case 27: return {needs(0) | needs(21) | needs(26), point};
    }
    return {0, 0};
}
#undef WAM_GRAPH_HD
