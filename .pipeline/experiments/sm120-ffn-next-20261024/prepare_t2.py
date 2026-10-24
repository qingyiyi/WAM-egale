"""唯一设计：相邻两段同owner，原七段算术不变；不修改采用A。"""
from pathlib import Path
import difflib
import json
import sys

P=Path(__file__).resolve().parent
OLD=P.parent/'rtx5090-block-port-20261023/ffn-local-handoff-20261024'
T=P/(sys.argv[1] if len(sys.argv)>1 else 'T2');T.mkdir(exist_ok=False)
s=(P/'A-source/csrc/whole_layer.cu').read_text();original=s


def once(text,old,new):
    assert text.count(old)==1,old[:100]
    return text.replace(old,new,1)


s=once(s,'union alignas(1024) TileShared {', '''// FFN2 compute的mainloop/result复用同一union；跨段BF16状态另存。
struct FfnPairStorage {
    CutlassTile<64>::Storage compute;
    B running[64*128];
};
static_assert(sizeof(FfnPairStorage) <= 98304, "keep the frozen A TileShared budget");
union alignas(1024) TileShared {
    FfnPairStorage ffn_pair;''')
start=s.index('__device__ __forceinline__ void ffn2_commit64(');end=s.index('using DeviceCounter',start)
a=s[start:end]
a=once(a,'int tile, int partition, int partitions = 2) {\n    const bool second = partition == partitions - 1;\n    struct alignas(8) B4',
'''int tile, int partition, int partitions = 2,
                                            const B *local_previous = nullptr, B *local_next = nullptr,
                                            bool publish_global = true) {
    const bool second = partition == partitions - 1;
    struct alignas(8) B4''')
a=once(a,'                const B4 prev = *reinterpret_cast<const B4 *>(out + row * D + col);',
'''                const B4 prev = *reinterpret_cast<const B4 *>(
                    local_previous ? local_previous + i : out + row * D + col);''')
a=once(a,'            if (!second || p.debug_modulation)\n                *reinterpret_cast<B4 *>(out + row * D + col) = rounded;',
'''            if (local_next)
                *reinterpret_cast<B4 *>(local_next + i) = rounded;
            if (publish_global && (!second || p.debug_modulation))
                *reinterpret_cast<B4 *>(out + row * D + col) = rounded;''')
s=s[:start]+a+s[end:]
start=s.index('__device__ __forceinline__ void action_ffn2_splitk(')
end=s.index('// FFN局部生产/消费流水',start)
old=s[start:end]
new=old.replace('partitions = 7, tasks_per_row = columns * partitions','partitions = 7, groups = 4, tasks_per_row = columns * groups')
begin=new.index('        const int partition = job / columns')
new=new[:begin]+'''        const int first_partition = 2 * (job / columns), tile = row * columns + job % columns;
        const int count = min(2, partitions-first_partition);
        auto &workspace = s.tile.ffn_pair.compute;
        B *running = s.tile.ffn_pair.running;
        for (int offset=0; offset<count; ++offset) {
            const int partition=first_partition+offset;
            // 每一段仍独立零初始化、执行32次K64 MMA，绝不跨段连续FP32累加。
            ffn2_partition64(workspace, bp(p, ACT), (const B *)p.w[25], tile, partition);
            if (offset==0) {
                if (threadIdx.x==0)
                    while (DeviceCounter(p.ffn_state[16+tile]).load(cuda::memory_order_acquire)
                           != first_partition) __nanosleep(128);
                __syncthreads();
            }
            // 第一次显式BF16舍入留shared；第二次按原顺序合并，再发布组边界。
            ffn2_commit64(p,workspace,tile,partition,partitions,
                offset ? running : nullptr,
                offset==0 && count==2 ? running : nullptr,
                offset==count-1);
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x==0)
            DeviceCounter(p.ffn_state[16+tile]).store(first_partition+count,cuda::memory_order_release);
        __syncthreads();
        if (first_partition+count==partitions)
            ffn_publish(p,8,row,columns);
    }
}

'''
s=s[:start]+new+s[end:]
s=once(s,'    result["ffn_pipeline_state"] = queues.narrow(0, MAX_STAGES, 16);',
'''    result["ffn_pipeline_state"] = queues.narrow(0, MAX_STAGES, 16);
    result["ffn_pair_schema"] = at::tensor({4,7,96,384}, at::TensorOptions().dtype(at::kInt));''')
# 说明旧代码注释的任务数已变化；不改变任何算术或其他阶段。
s=s.replace('// SM120 FFN2：96个输出tile × 7个K2048片段，仍使用原M64/N128/K64主循环。',
'''// SM120 PAIR2048x2：96输出tile×4 owner组；仍计算7个K2048，M64/N128/K64不变。''')
s=s.replace('// 每个行组按partition-major领取：前一片段一定先取得owner。所有cooperative',
            '// 每个行组按group-major领取：前一组一定先取得owner。所有cooperative')
s=s.replace('// SM120将七个K2048片段分派为独立任务；其他架构保留原两段执行路径。',
            '// SM120将七段分为[0,1]/[2,3]/[4,5]/[6]四组；其他架构原路径不变。')
(T/'whole_layer.cu').write_text(s)
(T/'CHANGE.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='frozen-A/whole_layer.cu',tofile='PAIR2048x2/whole_layer.cu')))
build=(OLD/'B-input/build_candidate.py').read_text().replace('wam_whole_ffn_input','wam_sm120_ffn_pair')
(T/'build_candidate.py').write_text(build)

# 独立profile额外导出七段数值，正常候选DSO没有此大tensor或写回。
q=once(s,'struct Params {','struct Params {\n    float *ffn_numeric;')
start=q.index('__device__ __forceinline__ void ffn2_commit64(');end=q.index('using DeviceCounter',start)
a=q[start:end]
a=once(a,'            float4 z = *reinterpret_cast<const float4 *>(s.result + (i / 128) * 136 + col_in_tile);',
'''            float4 z = *reinterpret_cast<const float4 *>(s.result + (i / 128) * 136 + col_in_tile);
            if (p.ffn_numeric)
                *reinterpret_cast<float4 *>(p.ffn_numeric + partition*2*224*D+row*D+col)=z;''')
a=once(a,'            const B4 rounded = {__floats2bfloat162_rn(z.x, z.y), __floats2bfloat162_rn(z.z, z.w)};',
'''            const B4 rounded = {__floats2bfloat162_rn(z.x, z.y), __floats2bfloat162_rn(z.z, z.w)};
            if (p.ffn_numeric) {
                const float2 lo=__bfloat1622float2(rounded.lo),hi=__bfloat1622float2(rounded.hi);
                *reinterpret_cast<float4 *>(p.ffn_numeric+(partition*2+1)*224*D+row*D+col)={lo.x,lo.y,hi.x,hi.y};
            }''')
q=q[:start]+a+q[end:]
q=once(q,'    void *launch_args[] = {&p};',
'''    auto numeric=at::empty({debug?7:0,2,224,D},fo);
    p.ffn_numeric=debug?numeric.data_ptr<float>():nullptr;
    void *launch_args[] = {&p};''')
q=once(q,'    if (!debug)\n        return result;',
'''    result["ffn_numeric"]=numeric;
    if (!debug)
        return result;''')
(T/'profile_base.cu').write_text(q)
profile=(OLD/'B-input/make_profile.py').read_text().replace('wam_whole_ffn_input_profile','wam_sm120_ffn_pair_profile')
profile=profile.replace("wam_whole_[^']+_cuda","wam_sm120_[^']+_cuda")
profile=profile.replace("P/'whole_layer.cu'","P/'profile_base.cu'")
(T/'make_profile.py').write_text(profile)

precision=(OLD/'B-input/precision_candidate.py').read_text()
precision=precision.replace("state[4:8] == [168] * 4","state[4:8] == [96] * 4")
precision=precision.replace("expected_splitk=args.expect_splitk == 'true', stream=args.stream,",
    "expected_splitk=args.expect_splitk == 'true', task_schema='PAIR2048x2-384-owner-tasks-v1', stream=args.stream,")
precision=precision.replace("row['ffn_splitk_commit'] = commits", "row['ffn_splitk_commit'] = commits\n                    row['ffn_pair_schema']=debug['ffn_pair_schema'].tolist()\n                    if row['ffn_pair_schema']!=[4,7,96,384]:raise RuntimeError('new pair schema mismatch')")
(T/'precision_candidate.py').write_text(precision)
measure=(OLD/'B-input/measure.py').read_text()
measure=measure.replace("result['cluster_capability'] = normal.cluster_info()", "result['cluster_capability'] = 'NOT_REQUESTED: no cluster/DSM change'")
measure=measure.replace("splitk_state[4:8]==[168]*4","splitk_state[4:8]==[96]*4")
measure=measure.replace("                tma_counts = debug['projection_tma_counts'].tolist()", "                if debug['ffn_pair_schema'].tolist()!=[4,7,96,384]:raise RuntimeError('PAIR2048x2 schema mismatch')\n                tma_counts = debug['projection_tma_counts'].tolist()")
marker="                    row['profile_resources'] = own(prof,True)['resources'].tolist()"
addition='''                    observed=own(prof,True)
                    trace=observed['ffn_numeric'].cpu()
                    previous=torch.load(P.parent/'T1-r1/data'/('numeric-'+length+'.pt'),map_location='cpu',weights_only=False)
                    row['seven_segments']=[{'s':s,'partial':compare(previous[s,0],trace[s,0]),'running':compare(previous[s,1],trace[s,1])} for s in range(7)]
                    if not all(z['partial']['bitwise_pass'] and z['running']['bitwise_pass'] for z in row['seven_segments']):raise RuntimeError('seven-segment numeric contract differs')
                    torch.save(trace,args.output/('numeric-candidate-'+length+'.pt'))
                    row['profile_resources']=observed['resources'].tolist()
                    del previous,trace,observed'''
measure=once(measure,marker,addition)
(T/'measure.py').write_text(measure)
(T/'SOURCE-ENTRY.json').write_text(json.dumps(dict(candidate='PAIR2048x2',base='SELF-QKV-TMA-4MMA',owner_tasks=384,mathematical_segments=672,normal_source='whole_layer.cu',independent_profile='profile_base.cu -> whole_layer_profile.cu',old_schema_not_rewritten=True,global_ACT_retained=True,FFN1_recompute_factor=1),indent=2))
print('one PAIR2048x2 design prepared; no old candidate repeated')
