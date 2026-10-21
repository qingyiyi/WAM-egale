"""Refresh derived navigation only; protected state/cycle/rules are never opened."""
from pathlib import Path
import datetime
import json
import shutil

B = Path(__file__).resolve().parent
R = B.parents[3]
C = R/'.pipeline/cycles/wam-research-20261016'
assert R.name == '论文探索'
now = datetime.datetime.now().astimezone().isoformat()
resource = json.loads((B/'RESOURCE-AND-STOP-REPORT.json').read_text())
history = C/'history/overnight-before-20261022'
if not history.exists():
    history.mkdir(parents=True)
    for name in ('PLAN.md','PROGRESS.md','EXECUTION.md','DISCUSSION-SUMMARY.md'):
        shutil.copy2(C/name, history/name)
rel = '../../experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022'
robo = '../../experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021'
pending = resource['status'] != 'CLOSED'
ncu = '本机最终版NCU新窗口等待自然空闲，最迟08:47:30启动、09:00清理；前窗口外部干扰0kernel，原PARTIAL保留。' if pending else '本机NCU窗口均已终态；计数及是否有效见本轮RESOURCE报告，不将PARTIAL改成PASS。'
account = f"M39累计保守计费{resource['cumulative_charged_seconds']:.3f}/36000秒（含本机等待，非CUDA active）；远端夜间GPU阶段1427.785秒、CPU构建准备2460.965秒。"
common = '四项科学权限false；paused72、A_off、旧retention/FAILED、baseline/原tile/投机生产和state/cycle/rules不动。BAAI仅~/cxz；未新建论文初稿。'
(C/'PLAN.md').write_text(f'''---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '{now}'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

|ID|阶段|下一步|
|---|---|---|
|M17|LIBERO证据收口|6000局的曝光/首两S/长尾缺口保留|
|M39|单层先对齐|12候选终态，11通过有限验证；NCU补证，解释仍慢约11%–12.6%的原因后再决定调度|
|M40|RoboTwin迁移|跨任务4局已终态，原01:26截止已过；1800局和160GPU小时均未启动/未批准|

- [x] 夜间Self-Q、point/norm、写回、attention、FFN与同步对照；前瞻有界修订8→10→12保留，08b是正确性修复。
- [x] 最终9真实签名＋21结构变体有限逐位；B14三KV完整层五臂同轮两独立进程。
- [x] 3KV×28阶段两进程、原版前缀和资源控制；旧FAILED及退化候选保留。
- [x] 最终源0cc1944b、DSOac3f9e46，两机optimized安装同一验证二进制；旧2a8保留。
- [{' ' if pending else 'x'}] 本机最终版NCU有界窗口收口。{ncu}
- [ ] 完全对齐guarded强基线；当前仍有约11%–12.6%差距，不写完成。
- [ ] ready-DAG（尚未实现；先解释/处理原语与统一资源成本）。
- [x] RoboTwin两项＋修复组合、跨任务/长失败4局核验，owned清理。
- [ ] RoboTwin扩大资源/吞吐再评估；旧截止不延长，不自动扩大。

{account} 原累计10h、本轮30000秒/3GiB/09:00不增。M40累计54718.243980/86400秒、原10/22 01:26:04.706686截止已过。{common}

[夜间证据]({rel}/RESULT.md)、[之前完整计划](history/overnight-before-20261022/PLAN.md)。
''')
(C/'PROGRESS.md').write_text(f'''# 当前研究进度

更新{now}。

|ID|阶段|状态|证据|下一步|
|---|---|---|---|---|
|M17|LIBERO证据收口|in_progress|6000局齐，非独立TEST|曝光/首两S/长尾缺口|
|M39|单层先对齐|in_progress|最终完整层约2.296/2.453/2.476ms，有限逐位|NCU及剩余强基线差距，再DAG|
|M40|RoboTwin迁移|in_progress|跨任务4局终态、owned清理|原截止已过，扩大未获新资源|

## M39：夜间已验证版本

[结果]({rel}/RESULT.md)、[同轮完整层表]({rel}/WHOLE-LAYER-COMPARISON.csv)、[最终逐阶段表]({rel}/stage-control-final-r2/RESULT.md)。12项候选＋08b修复已终态。选11：静态point/norm分派，padded epilogue，嵌入已有guarded self，B14双4warp attention，FFN1 N256/K32半区写回，删除冗余静态入口同步。仍单层108CTA/256线程，阶段出口依赖保留；无ready-DAG，无30层融合。

B14短/中/长KV同轮两进程CUDA Graph约2.296/2.453/2.476ms，相对本夜开始2.510/2.718/2.977ms下降8.52%/9.75%/16.81%。同工作量uncached guarded强基线2.070/2.187/2.201ms，仍慢约10.94%/12.17%/12.50%；未完全对齐。Torch静态原数学路径2.315/3.101/4.336ms，不挑较慢dense路径宣称超过全部最佳实现。不是机器人E2E或新有效提交成本。

最终9真实签名（B2action/B14action/B2video×3KV）、81完整调用及21结构变体84调用通过有限逐位：原Torch输出、23共有中间量、普通/debug、有效KV更新、输入/RNG。08 short/drop1有2个中间元素差1.49e-8，原FAILED保留；08b非16尾G1修复后通过。不推论所有输入或在线策略质量。源0cc1944b、DSOac3f9e46，[两机部署身份]({rel}/SELECTED.json)。

中KV剩余显著项：self_attention约231 vs guarded157µs，Cross-K RMS116 vs70µs，FFN1 199 vs167µs，FFN2 138 vs117µs，GELU30 vs15µs。阶段独立运行也慢；不是全由全局同步。最小shared/更多CTA控制使RMS/GELU部分改善，但不普遍；完整层100096B shared、255regs/thread、1CTA/SM限制各阶段单独选资源。静态local936B不是动态spill字节。已有grid_constant参数标注，不重复计作新改动。

{ncu} {account} 本轮[资源/停止账本]({rel}/RESOURCE-AND-STOP-REPORT.json)持续显示等待状态，09:00硬截止。远端本轮全部owned已退出。

## M40：工程跨任务收口

[4局结果]({robo}/RESULT.md)：basket valid14原版/组合均成功，177.761→166.753秒（−6.19%）；microwave valid14均失败，921.867→887.386秒（−3.74%）。全程受用观察/物理/RNG/命令/RPC摘要与终态一致；固定顺序每例一次探索性成本，不是稳定新质量N。原组合valid16四臂均成功，组合−7.33%保留。全局工程默认未改。

累计54718.243980/86400秒，owned清理；原10/22 01:26:04.706686截止已过，不重启。1800局/160GPU小时未启动/未批准。本机允许最多四分之一、同case同卡、证据回传远端，当前没有本机RoboTwin worker。旧DEV90 T/C/P各30成功25/24/23，非独立TEST；提交调用与实际推进分列，旧字段/罚分不覆盖。

## 共同边界与日期

{common} LIBERO三方法各4×10×50=2000，共6000局；Long483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499，各500。不是每方法三seed重复。

10/22 go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包。质量和系统证据不足就缩小主张，不制造PASS。

[本轮前完整记录](history/overnight-before-20261022/PROGRESS.md)。
''')
(C/'EXECUTION.md').write_text(f'''# 最近Execution（{now}）

1. 沿用户持续至10/22 09:00、原M39累计10h授权，完成12候选、08b修复、两进程三KV整层及最终逐阶段；负结果和实施失败保留。[作业/证据索引]({rel}/RESULT.md)。
2. 本机/BAAI仅optimized安装最终源0cc1944b/DSOac3f9e46，旧二进制保留，所有部署有真实收据；未修改baseline或投机生产。
3. {ncu} {account} 剩余预算见[动态账本]({rel}/RESOURCE-AND-STOP-REPORT.json)，不得把等待记为GPU实际计算。
4. M40跨任务4局已终态、owned退出，累计54718.243980秒；01:26硬截止不延长，无后继1800队列。
5. 更新派生短入口和Cycle四文件，前版本存history；不运行全量sync，不覆盖保护文件。GitHub main同步以实际commit回执为准。

[前Execution](history/overnight-before-20261022/EXECUTION.md)。
''')
(C/'DISCUSSION-SUMMARY.md').write_text(f'''# 当前讨论与授权（{now}）

用户授权睡眠期间继续单层性能对齐至北京时间10/22 09:00。先对齐，无法完全对齐需给证据和合理解释；保留完整一层单kernel、tile任务和persistent CTA边界。优先register/shared，减少不必要local访存，但不以资源越大或warp越多当收益。

当前12候选已收口，最终比夜间开始快8.5%/9.7%/16.8%，仍慢于同工作量guarded约11%–12.6%。精度仅有限fixture逐位，不等于所有在线输入质量。统一整层资源限制已获得干预证据；调用边界动态代价仍待最新NCU，不把旧计数套到新版本。{ncu}

下一步先完成本轮证据归档和剩余原因判定，再考虑依赖就绪DAG补位。没有授权扩大为30层融合、改阈值/模型/质量门或启动新训练/TEST。

RoboTwin工程4局已完整终态，原01:26截止已经结束；用户愿意当前版本先跑和本机≤1/4不等于此前未批准的160GPU小时扩大预算。当前没有机器人作业。{common}

[前讨论](history/overnight-before-20261022/DISCUSSION-SUMMARY.md)。
''')
(R/'.pipeline/PROGRESS.compact.md').write_text(f'''# WAM短状态（{now}）

先读INDEX、CONTEXT-POLICY、memory约束，再按需读[Cycle四文件](cycles/wam-research-20261016/PROGRESS.md)；v15语义入口，不全量sync或全载旧索引。

## 当前主线

M39先对齐单层，再依据原语/资源证据做tile ready调度。只改算子研究/wam_ops/optimized，baseline/原tile/投机生产不动。当前不是30层大融合，不是闭环性能放行。

## 算子：已完成和未完成

[夜间结果](experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)：12实现候选＋08b修复已终态，最终选择11。B14短/中/长KV完整层约2.296/2.453/2.476ms，比夜间开始下降8.52%/9.75%/16.81%；同轮uncached guarded强对照2.070/2.187/2.201ms，仍慢10.94%/12.17%/12.50%，未完全对齐。2独立进程/轮换Graph，非profiler计时、非机器人E2E。

9真实签名与21结构变体通过有限逐位：原Torch输出、23共有中间量、有效KV、普通/debug、输入/RNG。08 short/drop1的2个中间元素差1.49e-8旧FAILED保留，08b非16尾G1修复。最终源0cc1944b/DSOac3f9e46两机同一验证二进制，旧2a8保留。仍108CTA/256线程单层；静态分派/padded写回/双4warp attention/FFN1 N256K32/移除冗余入口同步，真实出口依赖保留，ready-DAG未实现。

[逐阶段](experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/stage-control-final-r2/RESULT.md)：self-attention、Cross-K RMS、FFN/GELU为剩余显著项；独立阶段仍慢，不能全怪grid同步。资源控制支持统一shared/驻留限制是部分原因。静态local936B不是动态spill。{ncu}

{account} 09:00硬截止，本轮30000秒/新增3GiB不增。[资源状态](experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESOURCE-AND-STOP-REPORT.json)。远端GPU作业已退出；本机仅等待/诊断，不清理别人进程。

## RoboTwin与LIBERO

M40[工程跨任务4局](experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021/RESULT.md)已终态，basket14双方成功−6.19%、microwave14双方失败−3.74%，全程受用观察/物理/RNG/命令/RPC摘要同；固定顺序单次探索性，非质量N。旧组合valid16−7.33%保留。累计54718.243980/86400秒、原10/22 01:26:04.706686截止已过、owned空；1800局/160GPU小时未启动/未批准。本机最多1/4并回传远端，当前0worker。旧DEV90 T/C/P25/24/23各30局，非独立TEST；提交与真实推进分列。

M17 LIBERO6000局（每方法四套×500）齐；Long T/C/P483/484/473，Goal463/464/467，Spatial472/466/464，Object497/498/499。曝光/首两S/长尾缺口仍在，不是每方法三seed重复，不是独立TEST。

## 不变边界与日期

{common} TF64结构仅研究暂冻，旧TF质量NO-GO不因算子逐位通过而改变；PF2不是bonus。BAAI仅~/cxz。10/22 go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包；证据不足缩小claim。
''')
(R/'.pipeline/INDEX.md').write_text(f'''# WAM当前入口

更新{now}。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)、[短状态](PROGRESS.compact.md)与[Cycle](cycles/wam-research-20261016/PROGRESS.md)。固定v15，不全量sync。

- M39[夜间单层结果](experiments/single-layer-async-gemm-20261020/overnight-alignment-20261022/RESULT.md)：B14三KV下降8.5%/9.7%/16.8%，仍慢于同工作量guarded约11%–12.6%；9签名＋21变体有限逐位。两机安装最终版本，NCU状态看资源账本；硬截止10/22 09:00。
- M40[跨任务4局](experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021/RESULT.md)已终态/清理，原01:26截止已过，1800局未启动。
- [算子源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[带读prompt](../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)。四权限false、旧失败与保护文件保持。
''')
print('Updated derived Cycle and short navigation',len((R/'.pipeline/PROGRESS.compact.md').read_bytes()))
