"""CPU-only attribution; matching the failed prototype is not admission."""
import csv
from datetime import datetime
import json
from pathlib import Path
import statistics

B = Path(__file__).resolve().parent
RAW = B / 'remote-evidence/diagnostic'


def main():
    profile = json.loads((RAW / 'profile-1/RESULT.json').read_text())
    pad = json.loads((RAW / 'pad40-1/RESULT.json').read_text())
    queue = json.loads((RAW / 'QUEUE-STATUS.json').read_text())
    registration = json.loads((B / 'DIAGNOSTIC-REGISTRATION.json').read_text())
    rows, stages = [], []
    for original, challenge in zip(profile['rows'], pad['rows']):
        assert original['case'] == challenge['case']
        case = original['case']
        values = {
            mode: [v['stream_ms'] for v in pad['timing']
                   if v['case'] == case and v['name'] == mode]
            for mode in ('profile', 'pad40')
        }
        assert all(len(v) == 2 for v in values.values())
        before, after = [statistics.median(values[k]) for k in ('profile', 'pad40')]
        ns = original['profile_stage_ns']
        row = dict(
            case=case,
            profile_kernel_ms=original['profile_kernel_ns'] / 1e6,
            GEMM_fraction=original['profile_GEMM_ns'] / original['profile_kernel_ns'],
            cross_KV_fraction=(ns['cross_K_proj'] + ns['cross_V_proj']) / original['profile_kernel_ns'],
            self_attention_fraction=ns['self_attention'] / original['profile_kernel_ns'],
            unassigned_fraction=original['profile_unassigned_ns'] / original['profile_kernel_ns'],
            profile_stream_ms=before, pad40_stream_ms=after,
            pad40_over_profile_latency=after / before,
            diagnostic_latency_reduction=1 - after / before,
            profile_matches_v1=all(v['match_v1'] for v in original['checks'].values()),
            pad40_matches_v1=all(v['match_v1'] for v in challenge['checks'].values()),
            input_RNG_unchanged=all(r['RNG_unchanged'] and r['inputs_unchanged']
                                    for r in (original, challenge)),
            original_numerical_gate=False,
        )
        rows.append(row)
        for mode, source in [('profile', original), ('pad40', challenge)]:
            for label, duration in source[mode + '_stage_ns'].items():
                stages.append(dict(case=case, mode=mode, stage=label,
                                   nanoseconds=duration,
                                   fraction=duration / source[mode + '_kernel_ns']))
    calls = profile['complete_block_calls'] + pad['complete_block_calls']
    assert calls == registration['complete_block_forward_cap'] == 83
    assert all(r['profile_matches_v1'] and r['pad40_matches_v1'] and
               r['input_RNG_unchanged'] for r in rows)
    result = dict(
        at=datetime.now().astimezone().isoformat(),
        status='FINITE_DIAGNOSTIC_COMPLETE_ORIGINAL_NUMERICAL_AND_LATENCY_NO_GO',
        original_numerical_gate=False, production_admitted=False,
        formal_performance_permission=False,
        profile_complete_block_calls=profile['complete_block_calls'],
        pad40_complete_block_calls=pad['complete_block_calls'],
        total_M34_complete_block_calls=registration['already_completed_M34_complete_block_calls'] + calls,
        B14_GEMM_fraction=profile['B14_GEMM_kernel_fraction'],
        padding_trigger_pre_registered=profile['padding_trigger'],
        diagnostic_latency_reduction_range=[min(x['diagnostic_latency_reduction'] for x in rows),
                                            max(x['diagnostic_latency_reduction'] for x in rows)],
        rows=rows,
        measured_with='SM80 SASS PTX globaltimer; device stage elapsed snapshots, not NCU counters',
        counter_measurements=dict(L2_requests='UNKNOWN', DRAM_bytes='UNKNOWN',
                                  bank_conflicts='UNKNOWN', achieved_occupancy='UNKNOWN'),
        timed_challenger_scope='one process, two interleaved samples/case; instrumented profile vs pad40 only',
        original_reference_not_retimed_with_pad40=True,
        primary_source_and_DSO_unchanged=queue['original_primary_source_and_DSO_unchanged'],
        reservation_T0=queue['actual_T0'], closed_at=queue['finished_at'],
        reservation_seconds=(datetime.fromisoformat(queue['finished_at']) -
                             datetime.fromisoformat(queue['actual_T0'])).total_seconds(),
        all_owned_reaped=all(j['root_reaped'] for j in queue['jobs']),
        GPU_contexts_after=queue['remaining_NVML'],
        GPU_worker_lifecycle_seconds=sum(j['lifecycle_seconds'] for j in queue['jobs']
                                        if not j['name'].startswith('BUILD')),
        CPU_build_inside_reservation_seconds=sum(j['lifecycle_seconds'] for j in queue['jobs']
                                                if j['name'].startswith('BUILD')),
        CUDA_active_seconds='UNKNOWN', new_robot_episodes=0, full30_calls=0,
        diagnostic_function_resources={mode: pad['rows'][0]['checks'][mode]['resources']
                                       for mode in ('profile', 'pad40')},
    )
    (B / 'DIAGNOSTIC-RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, data in [('STAGES.csv', stages), ('LAYOUT-DIAGNOSTIC.csv', rows)]:
        with (B / name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    text = [
        '# 整层原型：阶段归因和唯一布局诊断', '',
        '原精度门仍为FAILED，原型未接入生产。此处“匹配”只指已失败v1的数学和状态隔离，不是匹配Torch。原RESULT.md/270计时样本保留。', '',
        '9真实输入的profile和pad40均逐位匹配v1中间量、最终hidden与只读有效KV，输入/RNG不变；未追加合法写序列重验，不能把旧写入失败改成通过。两种诊断扩展的C++ namespace、模块身份和单kernel trace不同且经过核验。', '',
        f'首轮设备计时：三B14合计GEMM占{100 * result["B14_GEMM_fraction"]:.2f}%，触发了事先登记的唯一ldm32→40 BF16共享布局实验。29阶段通过PTX globaltimer记录，包含阶段边界等待；不是指令级profile或实测硬件计数。', '',
        '| 输入 | GEMM占比 | cross K/V投影占比 | self attention占比 | profile ms | pad40 ms | 诊断延迟下降 |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        text.append(f'|{r["case"]}|{r["GEMM_fraction"]:.1%}|{r["cross_KV_fraction"]:.1%}|'
                    f'{r["self_attention_fraction"]:.1%}|{r["profile_stream_ms"]:.4f}|'
                    f'{r["pad40_stream_ms"]:.4f}|{r["diagnostic_latency_reduction"]:.1%}|')
    text += [
        '', 'profile/pad40列是一独立进程、每格两次交错样本的CUDA stream中位数，只能支持本次布局的局部方向信号。设备内阶段表来自独立审计调用，不能把其绝对时间和另一调用的stream时间直接相减。未锁时钟，跨case绝对时间不作单调性结论；新增诊断没有给出与native同轮重测的改善幅度。', '',
        '观察：B14大量时间花在cross K/V投影，长KV下self attention占比增加。已有静态prompt/cached路径就是强baseline的一部分；避免这些重复投影属于已有缓存收益，不能重新命名为MegaTransformer贡献。当前整层实现尚未接入该缓存协议。', '',
        '主原型172 registers/thread；加计时profile为175，pad40为198。诊断两者shared均59616B、LOCAL0、理论1 CTA/SM；编译器资源也随源码变化，不能把这当成只改变bank conflict的硬件隔离。WMMA数学/调度代码保持，9输入match-v1；改善与共享布局敏感性一致，但没有NCU bank-conflict计数，不能断言唯一根因。仍是同步加载、粗阶段barrier和全局中间量，没有跨阶段生产者消费者细流水。单kernel和warp分工本身没有保证速度。', '',
        f'本阶段实际83完整block调用，M34累计{result["total_M34_complete_block_calls"]}/600；0完整30层、0机器人episode。GPU预留生命周期{result["reservation_seconds"]:.3f}秒，其中两GPU worker生命周期{result["GPU_worker_lifecycle_seconds"]:.3f}秒、CPU编译{result["CPU_build_inside_reservation_seconds"]:.3f}秒；总CUDA active时间UNKNOWN。全部owned root已真实reap，GPU0 context为空。', '',
        '止损：不继续padding/寄存器/grid搜索，不晋级新kernel，不扩30层。本轮实现与诊断收口。后续若继续，先解决匹配原GEMM/attention数值契约的可行设备原语，再在强cached B14上验证tile流水；不能靠放宽旧门或把已有cache记为新贡献获得PASS。',
    ]
    (B / 'DIAGNOSTIC-RESULT.md').write_text('\n'.join(text) + '\n')
    print(result['status'], result['total_M34_complete_block_calls'], result['reservation_seconds'])


if __name__ == '__main__':
    main()
