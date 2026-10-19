"""Summarize complete first prototype; keep source/version and failed gates."""
import csv
import json
from pathlib import Path
import statistics

B=Path(__file__).resolve().parent


def main():
    runs={name:json.loads((B/'remote-evidence'/name/'RESULT.json').read_text())
          for name in ('validate-1','timing-1','timing-2')}
    samples=[];table=[]
    for name,run in runs.items():
        for x in run['timing']:samples.append(dict(process=name,**x))
    for ref in runs['validate-1']['rows']:
        case=ref['case'];r=dict(case=case,numeric_pass=ref['original_contract_passed'],
            max_abs=ref['original_vs_whole']['max_abs'],relative_L2=ref['original_vs_whole']['rel_l2'],
            first_saved_DAG_difference=ref['first_different_intermediate'],
            single_kernel_trace=ref['kernel_trace']['one_whole_layer_kernel'],
            readonly_valid_KV=ref['state']['pass_all'],tile_exact_once=ref['tile_exact_once'])
        for mode in ('Torch','native','whole'):
            values=[x['GPU_stream_ms'] for x in samples if x['case']==case and x['mode']==mode]
            r[mode+'_ms']=statistics.median(values)
            r[mode+'_samples']=len(values)
            for name in ('timing-1','timing-2'):
                r[mode+'_'+name+'_ms']=statistics.median([x['GPU_stream_ms'] for x in samples if
                    x['case']==case and x['mode']==mode and x['process']==name])
        r['whole_over_native_latency']=r['whole_ms']/r['native_ms']
        r['Torch_over_whole_speed_ratio']=r['Torch_ms']/r['whole_ms']
        table.append(r)
    queue=json.loads((B/'remote-evidence/GPU-QUEUE-STATUS.json').read_text())
    guard=json.loads((B/'remote-evidence/FROZEN-REMOTE.json').read_text())
    localguard=json.loads((B/'FROZEN-CPU.json').read_text())
    deployed=[r for r in guard['files'] if r['sha256'] is not None]
    absent=[r['path'] for r in guard['files'] if r['sha256'] is None]
    summary=dict(status='STRUCTURE_OBSERVED_NUMERIC_AND_LATENCY_NO_GO',
        source_CUDA_SHA256='df8b386f302b9904ac4ef8ab08d171ef6d9e6f8143451831c3318f5e0835ef0e',
        cases=table,real_inputs=9,three_worker_validation_not_new_inputs=True,
        single_kernel_trace_cases=sum(r['single_kernel_trace'] for r in table),
        original_bitwise_pass=sum(r['numeric_pass'] for r in table),
        resources=runs['validate-1']['rows'][0]['resources'],
        remote_block_attempts=sum(x['complete_block_attempts'] for x in runs.values()),
        remote_linear_diagnostic_calls=sum(x['primitive_diagnostic_calls'] for x in runs.values()),
        raw_timing_samples=len(samples),GPU_job_lifecycle_seconds=sum(j['lifecycle_seconds'] for j in queue['jobs']),
        all_GPU_roots_reaped=all(j['root_reaped'] and not j['remaining_NVML'] for j in queue['jobs']),
        source_guard=dict(remote_raw_all88=guard['all_unchanged'],remote_present=len(deployed),
            remote_deployed_unchanged=all(r['unchanged'] for r in deployed),
            remote_not_staged=absent,local_all88_unchanged=localguard['all_unchanged'],
            raw_missing_checks_retained=True,reason='protected local HW state/rules intentionally not staged into runtime mirror'),
        original_four_fields_unchanged=True,production_admitted=False,formal_performance_permission=False,
        original_precision_failed_not_relaxed=True,reference='uncached complete block eager; strongest cached/shared-prefix/Graph not yet compared',
        CUDA_active_total_seconds='UNKNOWN; measured region samples are not total CUDA active time',
        full30_calls=0,robot_episodes=0)
    (B/'RESULT.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name,rows in [('PERFORMANCE.csv',table),('RAW-TIMING.csv',samples)]:
        with (B/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# 整层一个persistent kernel：结构完成，精度与速度未验收','',
        '固定Long block0权重，9份真实完整block输入，B2/B14 action及B2 video×短中长KV。CUDA源码df8b386…；本机88保护项、远端86已部署项不变。远端未迁入state/rules，两Missing导致原始88项检查为false，原样保留，不创建缺失保护文件。0完整30层、0机器人episode。',
        '', '9条GPU trace各观察到一次整层计算launch；队列tile恰好一次、输入/RNG/参数flags/地址通过。self/cross attention、全部投影、FFN、Norm/RoPE/残差和KV管理都在内，8 warp按调度/加载/计算/写回分工。当前仍有global中间量及粗阶段屏障，无跨阶段细流水。',
        '', '| 输入 | Torch ms | 冻结native ms | 新整层 ms | 相对native延迟 | hidden输出相对L2 |',
        '|---|---:|---:|---:|---:|---:|']
    for r in table:text.append(f'|{r["case"]}|{r["Torch_ms"]:.4f}|{r["native_ms"]:.4f}|{r["whole_ms"]:.4f}|{r["whole_over_native_latency"]:.2f}×|{r["relative_L2"]:.6f}|')
    text += ['', '每格两独立进程、各5次交错样本，共10样本中位；270原始样本、分进程数据见CSV。profile与计时分离。不锁时钟，GPU1另卡任务及host条件留在原资源记录。这是未准入实现的成本诊断，不能称有效加速或论文正式性能。强cached/shared-prefix/Graph尚未比较；当前相对较弱native已明显更慢。',
        '', '原Torch输出/历史gold与冻结native有限对照通过；新实现9/9最终hidden输出逐位FAIL，相对L2约0.13%–0.56%，max abs见JSON。这不是物理动作误差或任务失败率。只读9有效KV通过；合法写序列18中9数值失败，metadata均过。新单kernel不得继承原接受路径、完整模型或闭环质量证书。',
        '', '相同输入Q/FFN2 Linear控制证实GEMM已有数值差异。公共cuBLASLt API重建读到：B2 FFN2 splitK7/inplace，B14 Q与FFN2 splitK2/inplace；新WMMA是单段FP32累加。配置与舍入契约不同，不能仅由metadata指定唯一根因；旧M26四公式失败不改、不重复无限扫参。',
        '', '实际函数172 registers/thread、59616B shared、LOCAL0，occupancy API理论1 CTA/SM、108 CTA。这是资源上限而非实测achieved occupancy。首版较小64×64×32 GEMM tile、同步加载、粗屏障和global workspace都有额外成本；精确阶段占比等待独立设备内计时，未将推断写成观察。',
        '', f'远端三GPU job共{summary["GPU_job_lifecycle_seconds"]:.3f}秒、{summary["remote_block_attempts"]}完整block尝试、{summary["remote_linear_diagnostic_calls"]}Linear控制；全部root真实reap并释放context。总CUDA active秒UNKNOWN。首次本机错误Python依赖失败0block和前两编译失败均保留。',
        '', '上述是首批v1的固定结果；后续有限profile/唯一布局诊断已收口，见DIAGNOSTIC-RESULT.md。精度边界见PRECISION-CUTS.md：first_saved_DAG_difference仅指已保存观测点，不补造主路径缺失的pre-norm投影。主源码/DSO冻结，不改baseline/投机侧，不集成，不扩30层，不以放宽精度门换PASS。']
    (B/'RESULT.md').write_text('\n'.join(text)+'\n')
    print(summary['status'],summary['remote_block_attempts'],len(samples),summary['source_guard'])


if __name__=='__main__':main()
