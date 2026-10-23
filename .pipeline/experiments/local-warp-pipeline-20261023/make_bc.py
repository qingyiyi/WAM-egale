from pathlib import Path
import shutil,difflib,json,hashlib
E=Path(__file__).resolve().parent;src=E/'worktree/算子研究/wam_ops/optimized';dst=E/'variants/BC/算子研究/wam_ops/optimized';shutil.copytree(src,dst)
p=dst/'csrc/whole_layer.cu';original=p.read_text();s=original
s=s.replace('bool temb_float, cache, candidate, affine, debug_modulation, context_cached;','bool temb_float, cache, candidate, affine, debug_modulation, context_cached;\n    bool ffn_prepare_overlap; // B/C share one DSO: only preparation ordering differs.')
pos=s.index('union alignas(32) TileShared')
s=s[:pos]+'''// Preparation belongs to this tile only. It does not alias active MMA storage.
struct alignas(8) PreparedB4 { __nv_bfloat162 lo, hi; };
struct FfnPrepare {
    float4 bias[32], table[32];
    float4 gate[256];             // First eight rows, not the complete tile.
    PreparedB4 residual[256];
};
struct FfnLocalTile {
    CutlassTile<128>::Storage math;
    FfnPrepare prepared;
};
static_assert(sizeof(FfnPrepare) == 7168);
static_assert(sizeof(FfnLocalTile) <= 112128, "Do not increase existing tile workspace");
'''+s[pos:]
s=s.replace('    GemmShared gemm;','    FfnLocalTile ffn_local;\n    GemmShared gemm;',1)
start=s.index('__device__ __forceinline__ void ffn2_partition');end=s.index('// 每线程四个相邻元素',start)
old=s[start:end];assert old.count('__syncthreads();')==1;s=s[:start]+old.replace('    __syncthreads();','    // Caller joins all eight warps after independent preparation.')+s[end:]
pos=s.index('// 每线程四个相邻元素')
s=s[:pos]+'''// W0/W1/W2/W7 prepare only inputs independent of the current accumulator.
// W3-W6 retain their original CUTLASS mainloop. No partial is read here.
__device__ __forceinline__ void ffn2_prepare(const Params &p, FfnPrepare &s,
                                             int tile, bool second) {
    const int pw = threadIdx.x / 32;
    if (!second || (pw >= 3 && pw < 7)) return;
    const int tid = pw == 7 ? int(threadIdx.x) - 128 : int(threadIdx.x);
    const int column = (tile % 24) * 128 + (tid % 32) * 4;
    const int cell = 5 * D + column;
    const auto b = *reinterpret_cast<const PreparedB4 *>((const B *)p.w[26] + column);
    const float2 bl = __bfloat1622float2(b.lo), bh = __bfloat1622float2(b.hi);
    const float4 table = *reinterpret_cast<const float4 *>((const float *)p.w[0] + cell);
    if (tid < 32) {
        s.bias[tid] = {bl.x, bl.y, bh.x, bh.y};
        s.table[tid] = table;
    }
    for (int vector = tid; vector < 256; vector += 128) {
        const int row = (tile / 24) * 128 + vector / 32;
        const int64_t src = (int64_t(row / p.q) * p.l + (p.l == 1 ? 0 : row % p.q)) * 6 * D + cell;
        float4 t;
        if (p.temb_float) t = *reinterpret_cast<const float4 *>((const float *)p.temb + src);
        else {
            const auto v = *reinterpret_cast<const PreparedB4 *>((const B *)p.temb + src);
            const float2 l = __bfloat1622float2(v.lo), h = __bfloat1622float2(v.hi);
            t = {l.x,l.y,h.x,h.y};
        }
        s.gate[vector] = {__fadd_rn(table.x,t.x), __fadd_rn(table.y,t.y),
                          __fadd_rn(table.z,t.z), __fadd_rn(table.w,t.w)};
        s.residual[vector] = *reinterpret_cast<const PreparedB4 *>(bp(p,R2) + row * D + column);
    }
}

'''+s[pos:]
s=s.replace('int tile, bool second) {\n    struct alignas(8) B4 { __nv_bfloat162 lo, hi; };','int tile, bool second, const FfnPrepare &prepared) {\n    using B4 = PreparedB4;')
a=s.index('    float4 bias{}, table{};',s.index('void ffn2_commit'));b=s.index('    for (int i = ',a)
s=s[:a]+'''    float4 bias{}, table{};
    if (second) {
        bias = prepared.bias[int(threadIdx.x) % 32];
        table = prepared.table[int(threadIdx.x) % 32];
    }
'''+s[b:]
a=s.index('                const int batch = row / p.q',s.index('void ffn2_commit'));b=s.index('                const float2 rlo =',a)
old=s[a:b];new=old.replace('                const int batch = row / p.q, token = row % p.q;','''                float4 gate;
                B4 residual;
                if (i < 8 * 128) {
                    gate = prepared.gate[i / 4];
                    residual = prepared.residual[i / 4];
                } else {
                const int batch = row / p.q, token = row % p.q;''')
new=new.replace('                const float4 gate =','                gate =')
new=new.replace('                const float2 lo = __bfloat1622float2(rounded.lo), hi = __bfloat1622float2(rounded.hi);\n','')
new=new.replace('                const B4 residual =','                residual =')
new+='''                }
                const float2 lo = __bfloat1622float2(rounded.lo), hi = __bfloat1622float2(rounded.hi);
'''
s=s[:a]+new+s[b:]
old='''        auto &workspace = s.tile.cutlass_gemm.ffn1;
        ffn2_partition(workspace, bp(p, ACT), (const B *)p.w[25], tile, group & 1);'''
new='''        auto &workspace = s.tile.ffn_local.math;
        auto &prepared = s.tile.ffn_local.prepared;
        if (!p.ffn_prepare_overlap) {
            ffn2_prepare(p, prepared, tile, group & 1);
            __syncthreads(); // B: explicitly serialize preparation before MMA.
        }
        if (compute())
            ffn2_partition(workspace, bp(p, ACT), (const B *)p.w[25], tile, group & 1);
        else if (p.ffn_prepare_overlap)
            ffn2_prepare(p, prepared, tile, group & 1);
        __syncthreads(); // Both accumulator and prepared inputs ready; buffer not reused.
'''
assert old in s;s=s.replace(old,new)
s=s.replace('ffn2_commit(p, workspace, tile, group & 1);','ffn2_commit(p, workspace, tile, group & 1, prepared);')
s=s.replace('const std::vector<Tensor> &context_kv) {','const std::vector<Tensor> &context_kv, bool ffn_prepare_overlap) {')
s=s.replace('    p.debug_modulation = debug;','    p.debug_modulation = debug;\n    p.ffn_prepare_overlap = ffn_prepare_overlap;')
s=s.replace('py::arg("context_kv") = std::vector<at::Tensor>{});','py::arg("context_kv") = std::vector<at::Tensor>{}, py::arg("ffn_prepare_overlap") = true);')
s=s.replace('    m.attr("production_admitted") = false;','    m.attr("ffn_prepare_bytes") = int(sizeof(wam_whole_layer_v1::FfnPrepare));\n    m.attr("ffn_local_tile_bytes") = int(sizeof(wam_whole_layer_v1::FfnLocalTile));\n    m.attr("production_admitted") = false;')
p.write_text(s);d=E/'T2-BC-build';d.mkdir();(d/'CHANGE.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='A/whole_layer.cu',tofile='BC/whole_layer.cu')))
R='/root/cxz/ops/local-warp-pipeline-20261023';reg=dict(goal='Build only FFN2 current tile preparation variant, shared B/C DSO',gpu=False,cap_seconds=600,command=['/root/cxz/envs/va/bin/python',f'{R}/variants/BC/算子研究/wam_ops/optimized/build_whole_layer.py','build_ext','--build-lib',f'{R}/T2-BC-build','--build-temp',f'{R}/T2-BC-build/build/obj'],source_sha256=hashlib.sha256(s.encode()).hexdigest());(d/'JOB.json').write_text(json.dumps(reg,indent=2)+'\n')
print(reg['source_sha256'])
