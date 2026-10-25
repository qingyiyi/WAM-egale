"""把本轮完整层与分段结果汇总；分段计时不相加冒充完整层。"""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import math
import statistics as stat

P = Path(__file__).resolve().parent
results = [json.loads((P/f'measure-{i}/RESULT.json').read_text()) for i in (1, 2)]
assert all(d['status'] == 'COMPLETE_EXACT_EAGER' for d in results)
public = json.loads((P/'PUBLIC-ENTRY-RESULT.json').read_text())
assert public['status'] == 'COMPLETE_PUBLIC_ENTRY_EXACT'
gm = lambda xs: math.exp(sum(math.log(x) for x in xs)/len(xs))

# 实际源码推导；attention的双组任务不等于跨候选自由注意。
definitions = [
 ('入口/KV布局', '1 CTA压紧slot；170 CTA同步', '有效history slot/映射；只读公共KV', '8 warp布局', 'grid出口'),
 ('LN1＋调制', '224', '每tile一行3072，归约后调制', '4归约；8写回', 'CTA＋grid'),
 ('Self QKV', '288', '64×128输出，K3072；Q/K/V各96', '1 TMA＋4 MMA＋3 E', '局部full/empty＋grid'),
 ('Q RMS＋RoPE', '112', '每tile两行3072，保留RMS舍入', '两组×4', '组/CTA＋grid'),
 ('K RMS＋RoPE', '112', '每tile两行3072，保留RMS舍入', '两组×4', '组/CTA＋grid'),
 ('Self attention', '短/中/长96/168/96双组任务', '每组一head、1或2候选各Q16；KV288/1152/2160', '两组×4', '组/CTA＋grid'),
 ('Self输出投影', '96', '64×128输出，K3072', '1 TMA＋4 MMA＋3 E', '局部full/empty＋grid'),
 ('残差1', '336', '2048元素，门控与残差', '8向量', 'grid'),
 ('LN2', '224', '每tile一行3072', '4归约；8写回', 'CTA＋grid'),
 ('Cross Q投影', '96', '64×128输出，K3072', '1 TMA＋4 MMA＋3 E', '局部full/empty＋grid'),
 ('Cross Q RMS', '112', '每tile两行3072', '两组×4', '组/CTA＋grid'),
 ('Cross attention', '168双组任务', '每组一个候选/head，Q16、context512、D128', '两组×4', '组/CTA＋grid'),
 ('Cross输出投影', '96', '64×128输出，K3072', '1 TMA＋4 MMA＋3 E', '局部full/empty＋grid'),
 ('残差2', '336', '2048元素，残差相加', '8向量', 'grid'),
 ('LN3＋调制', '224', '每tile一行3072，归约后调制', '4归约；8写回', 'CTA＋grid'),
 ('FFN1＋GELU', '108主体＋12尾块＝120', '主体128×256、尾32×256，K3072；GELU写ACT', '主体8/尾4 MMA；8后处理', 'CTA＋新增grid'),
 ('FFN2＋门控/残差3', '96输出×7段＝672', '64×128输出，每段K2048；最后一段做残差', 'W3–6 MMA；8后处理', 'CTA＋有序提交＋grid'),
 ('出口同步', '170 CTA', '完成整个block', '全CTA', 'grid'),
]
summary = dict(status='COMPLETE_ADOPT_S', generated_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               retained='S: FFN_STAGE_SPLIT', independent_A_preserved=True,
               complete_block_calls=sum(d['calls'] for d in results)+public['calls'],
               normal_scope='B14/Q16/block0; KV288/1152/2160; warm CrossKV; eager/noGraph; two independent processes',
               stage_scope='process1 embedded globaltimer incl barriers vs original Torch isolated eager regions; not additive',
               cases=[], formal_performance_permission=False)
stage_ratios = []
for i, case in enumerate(['short','middle','long']):
    rows = [d['rows'][i] for d in results]
    assert all(r['case'] == case for r in rows)
    for r in rows:
        assert len(r['precision']) == 27 and all(v['bitwise_pass'] for v in r['precision'].values())
        assert all(x['partial']['bitwise_pass'] and x['running']['bitwise_pass'] for x in r['seven_segments'])
        assert r['valid_KV']['pass_all'] and r['inputs_unchanged'] and r['RNG_unchanged']
        assert r['median_ms']['Own']/r['median_ms']['A'] <= 1.03
    r = rows[0]
    own = [stat.median(x[k] for x in r['regions_ms']) for k in range(18)]
    torch = {str(x['region']):x['median_ms'] for x in r['torch_stages']}
    ratios = [torch[str(k)]/own[k] if str(k) in torch else None for k in range(18)]
    stage_ratios.append(ratios)
    instrumentation = {a:stat.median(x['event_ms'] for x in r['instrumentation_samples'] if x['arm']==a) for a in ['normal','profile']}
    summary['cases'].append(dict(case=case, KV=r['total_KV'],
        medians_by_process=[x['median_ms'] for x in rows],
        torch_over_S=gm([x['median_ms']['Torch']/x['median_ms']['Own'] for x in rows]),
        guarded_over_S=gm([x['median_ms']['Guarded']/x['median_ms']['Own'] for x in rows]),
        S_latency_change_pct_vs_A=100*(gm([x['median_ms']['Own']/x['median_ms']['A'] for x in rows])-1),
        resources=r['resources'],profile_resources=r['profile_resources'],
        profile_overhead_pct=100*(instrumentation['profile']/instrumentation['normal']-1),
        own_stage_ms=own,torch_stage_ms=torch,stage_speedup=ratios,
        FFN_envelope_ms=stat.median(r['ffn_envelope_ms']),
        FFN_Torch_ms=torch['FFN_combined']))
(P/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')

table = '# 当前完整block阶段表：FFN两阶段同步版S\n\n'
table += '评估并采用S；独立原A保留。所有比值均为本轮新测 `T_Torch/T_S`，大于1更快。B14×Q16、hidden3072、FFN14336；RTX5090，170 CTA、256线程/8 warp。warm CrossKV、双方eager/noGraph。\n\n'
table += '|阶段|tile/任务数|每tile工作量|CTA内warp分工|短KV|中KV|长KV|同步|\n|---|---|---|---|---:|---:|---:|---|\n'
for i, (name,count,work,warps,sync) in enumerate(definitions):
    v = ['—' if stage_ratios[j][i] is None else f'{stage_ratios[j][i]:.3f}×' for j in range(3)]
    table += f'|{i+1} {name}|{count}|{work}|{warps}|'+ '|'.join(v)+f'|{sync}|\n'
table += '\n阶段比值来自观察器区间（含等待/屏障）与Torch独立eager片段，不可相加为完整层。入口/出口无独立同口径Torch片段，标—。Cross K/V投影与K RMS双方预计算，warm区间内不执行。E表示后处理warp；FFN后处理不代表与MMA已异步重叠。\n\n'
table += 'FFN1主体108块中，52块只有96/128有效行；最后四个N256列块另拆12个M32尾块。FFN2最后一个M64行块只有32有效行。Self attention双组数对应192/336/192个4warp逻辑任务，各候选保留私有可见性，不是一个Q112序列。\n\n'
table += '|无插桩完整层|短KV|中KV|长KV|\n|---|---:|---:|---:|\n'
table += '|原Torch/S|'+ '|'.join(f'{c["torch_over_S"]:.3f}×' for c in summary['cases'])+'|\n'
table += '|S相对原A延迟变化|'+ '|'.join(f'{c["S_latency_change_pct_vs_A"]:.2f}%' for c in summary['cases'])+'|\n'
table += '\n完整层汇总为两个独立进程各自中位数比值的几何平均；原始ms与每轮样本见两个measure目录。这里只验证三个冻结真实输入，不能外推所有输入或机器人E2E。\n'
(P/'STAGE-TABLE.md').write_text(table)
(P/'STAGE-TABLE-FINAL.md').write_text(table)

report = '''# FFN两阶段同步版：采用S，独立A保留

## OBSERVED

本轮仅在FFN1+bias/BF16/GELU写global ACT后增加grid同步，再执行原FFN2七段K2048与最终gate/残差。保留主体/尾tile、warp职责、七段FP32部分和及逐段BF16递推；没有新增DSM或统一warp比例。源码函数改名action_ffn_stages以反映实际阶段。

两独立进程×三真实KV fixture全部通过：27处输出/中间tensor逐位、七段partial/running逐位、有效KV/输入/RNG、非默认stream及profile输出。更新后的公开Python入口又逐一通过正常/debug输出及状态检查。有限fixture通过不等于任意输入证明，也不开放机器人质量/正式性能门。

完整层相对原A三个KV均改善，两个进程各自改善约1.74%–2.30%；满足事前≤3%回退筛选，并采用S作为下一轮易分析基线。170CTA、100352B shared/CTA、255regs/thread、784B静态local未变；profile静态local为792B，其整层扰动另列SUMMARY。

分开观察后FFN1+GELU只有原Torch速度的0.797–0.805倍（约慢24%–25%）；FFN2+gate/残差为1.027–1.033倍，已接近/略快于同边界原Torch。Self输出、Cross Q、Cross输出投影仍约0.84–0.88倍。完整18阶段表与原始时长见STAGE-TABLE.md、SUMMARY.json。

## 原因与UNKNOWN

这次受控修改说明当前按行组交接的FFN跨阶段重叠并未带来净优势；新增grid同步的S没有性能退化，不能简单把同步数量当瓶颈。源码同时改了函数名，编译器可改变布局；尚未把改善严格分离成某一种等待或缓存效应。

FFN1只有120个生产任务，少于170个常驻CTA；主块包含部分有效行，GEMM后FP32结果分两半经过shared、CTA屏障、bias/BF16/GELU再写ACT。这些是可核对的并行度/交接约束，但当前阶段区间尚未分离纯MMA、输入搬运、GELU、输出转换和屏障尾部，不能据此声称其中任一项就是全部慢因。无新增硬件计数器证据，不声称HBM或Tensor Core利用率变化。

## RECOMMENDATION

下一步只聚焦FFN1：先对照GEMM主体与GELU/写回成本，并区分主体和尾块、CTA实际任务分配；再根据结果决定是否改变分块或输出交接。保留FFN2现有七段实现及阶段全局同步，不同时引入DSM或完整DAG。三个小投影作为后续第二优先级。

## 固定入口

- 当前源码：../rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu（实际绝对路径见SOURCE-AND-RESULT-ENTRY.json）。
- 本轮冻结源码whole_layer.cu；旧A为A-whole_layer.cu及上级A.so，公开入口旧版A-whole_layer.py保留。
- 新二进制build/normal，独立观察器build/profile；make_profile.py可再生成18阶段观察器。
- 原始结果measure-1/RESULT.json、measure-2/RESULT.json；七段数值measure-1/numeric-candidate-*.pt；PUBLIC-ENTRY-RESULT.json核实际Python入口。
- JOB.json保留本轮起点/上限；原失败与旧窗口不修改。资源/清理另见RESOURCE-AND-STOP-REPORT.json。
'''
report = report.replace('../rtx5090-block-port-20261023/', '../../rtx5090-block-port-20261023/')
(P/'REVIEW_RETURN.md').write_text(report)
print(json.dumps({k:summary[k] for k in ['status','complete_block_calls','cases']},ensure_ascii=False))
