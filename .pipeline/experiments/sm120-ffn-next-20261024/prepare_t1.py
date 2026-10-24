"""只从冻结A生成独立诊断DSO；不改正常A或当前采用入口。"""
from pathlib import Path
import json
import sys

P = Path(__file__).resolve().parent
OLD = P.parent / 'rtx5090-block-port-20261023/ffn-local-handoff-20261024'
T = P / (sys.argv[1] if len(sys.argv)>1 else 'T1')
T.mkdir(exist_ok=True)
s = (P / 'A-source/csrc/whole_layer.cu').read_text()


def once(text, old, new):
    assert text.count(old) == 1, old[:100]
    return text.replace(old, new, 1)


s = once(s, 'struct Params {', 'struct Params {\n    int64_t *ffn_cost;\n    float *ffn_numeric;')
# debug独立记录每段原FP32 partial及已舍入running值；正常诊断不写大tensor。
start = s.index('__device__ __forceinline__ void ffn2_commit64(')
end = s.index('using DeviceCounter', start)
a = s[start:end]
a = once(a, '            float4 z = *reinterpret_cast<const float4 *>(s.result + (i / 128) * 136 + col_in_tile);',
'''            float4 z = *reinterpret_cast<const float4 *>(s.result + (i / 128) * 136 + col_in_tile);
            if (p.ffn_numeric)
                *reinterpret_cast<float4 *>(p.ffn_numeric + (partition * 2) * 224 * D + row * D + col) = z;''')
a = once(a, '            const B4 rounded = {__floats2bfloat162_rn(z.x, z.y), __floats2bfloat162_rn(z.z, z.w)};',
'''            const B4 rounded = {__floats2bfloat162_rn(z.x, z.y), __floats2bfloat162_rn(z.z, z.w)};
            if (p.ffn_numeric) {
                const float2 nl = __bfloat1622float2(rounded.lo), nh = __bfloat1622float2(rounded.hi);
                *reinterpret_cast<float4 *>(p.ffn_numeric + (partition * 2 + 1) * 224 * D + row * D + col) = {nl.x,nl.y,nh.x,nh.y};
            }''')
s = s[:start]+a+s[end:]
start = s.index('__device__ __forceinline__ void action_ffn2_splitk(')
end = s.index('// FFN局部生产/消费流水', start)
a = s[start:end]
a = once(a, '    while (true) {', '    while (true) {\n        int64_t clock0 = threadIdx.x == 0 ? ffn_timestamp() : 0;')
a = once(a, '        const int task = s.task;',
'''        if (threadIdx.x == 0) p.ffn_cost[blockIdx.x*7] += ffn_timestamp()-clock0;
        const int task = s.task;''')
a = once(a, '        if (task < 0) { __nanosleep(128); __syncthreads(); continue; }',
'''        if (task < 0) {
            if (threadIdx.x == 0) ++p.ffn_cost[blockIdx.x*7+6];
            __nanosleep(128); __syncthreads(); continue;
        }''')
a = once(a, '        ffn2_partition64(workspace, bp(p, ACT), (const B *)p.w[25], tile, partition);',
'''        if (threadIdx.x == 0) clock0 = ffn_timestamp();
        ffn2_partition64(workspace, bp(p, ACT), (const B *)p.w[25], tile, partition);
        if (threadIdx.x == 0) {
            p.ffn_cost[blockIdx.x*7+1] += ffn_timestamp()-clock0;
            clock0=ffn_timestamp();
        }''')
a = once(a, '        ffn2_commit64(p, workspace, tile, partition, partitions);',
'''        if (threadIdx.x == 0) {
            p.ffn_cost[blockIdx.x*7+2] += ffn_timestamp()-clock0;
            clock0=ffn_timestamp();
        }
        ffn2_commit64(p, workspace, tile, partition, partitions);
        if (threadIdx.x == 0) {
            p.ffn_cost[blockIdx.x*7+3] += ffn_timestamp()-clock0;
            clock0=ffn_timestamp();
        }''')
a = once(a, '        if (partition == partitions - 1)\n            ffn_publish(p, 8, row, columns);',
'''        if (threadIdx.x == 0) {
            p.ffn_cost[blockIdx.x*7+4] += ffn_timestamp()-clock0;
            ++p.ffn_cost[blockIdx.x*7+5];
        }
        if (partition == partitions - 1)
            ffn_publish(p, 8, row, columns);''')
s = s[:start]+a+s[end:]
start = s.index('__global__ void whole_layer(')
head, body = s[:start], s[start:]
body = once(body, '    auto g = cg::this_grid();',
'''    auto g = cg::this_grid();
    if (threadIdx.x == 0)
        for (int i=0;i<7;++i) p.ffn_cost[blockIdx.x*7+i]=0;''')
s = head+body
s = once(s, '    void *launch_args[] = {&p};',
'''    auto cost = at::empty({blocks,7}, lo);
    auto numeric = at::empty({debug ? 7 : 0,2,224,D}, fo);
    p.ffn_cost=cost.data_ptr<int64_t>();
    p.ffn_numeric=debug ? numeric.data_ptr<float>() : nullptr;
    void *launch_args[] = {&p};''')
s = once(s, '    if (!debug)\n        return result;',
'''    result["ffn_cost"] = cost;
    result["ffn_numeric"] = numeric;
    if (!debug)
        return result;''')
s = once(s, 'PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {',
'''PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ffn_storage_sizes", []() { using namespace wam_whole_layer_v1;
        return std::vector<int64_t>{sizeof(Shared),sizeof(TileShared),sizeof(CutlassTile<64>::Storage),
            sizeof(CutlassTile<64>::Mma::SharedStorage),sizeof(float)*64*136,
            sizeof(CutlassTile<64>::Storage)+sizeof(B)*64*128}; });''')
(T / 'whole_layer.cu').write_text(s)
build = (OLD / 'B-input/build_candidate.py').read_text().replace('wam_whole_ffn_input', 'wam_sm120_ffn_cost')
(T / 'build_candidate.py').write_text(build)
profile = (OLD / 'B-input/make_profile.py').read_text().replace('wam_whole_ffn_input_profile', 'wam_sm120_ffn_cost_profile')
profile = profile.replace("wam_whole_[^']+_cuda", "wam_sm120_[^']+_cuda")
(T / 'make_profile.py').write_text(profile)
print('T1 independent instrumentation prepared; original A unchanged')
