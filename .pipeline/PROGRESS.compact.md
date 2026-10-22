---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-23T00:47:13.615406+08:00'
current: M39
next: 根据当前阶段表讨论六个小投影与LN余差；GPU1继续固定RoboTwin队列，GPU0本轮已清理。
---

# 当前研究进度（2026-10-23T00:47:13.615406+08:00）

M39已按批准顺序完成：外层action_ffn_pipeline强制内联，然后仅FFN1的12个M32尾tile K32→K64。每项9真实签名/36结构控制、225次有限完整调用逐位/状态通过；两进程同真实输入计时通过，两机采用。外层内联当轮完整层延迟降0.42%/0.54%/0.61%，尾K64当轮再降0.79%/0.42%/0.56%。不跨轮次相乘宣称E2E。

当前原Torch/自有完整单层warm Graph为 **1.2426/1.7780/2.8859×**（KV288/1152/2160）。双方同最终Cross K/RMS与V预计算，原Torch静态dense history；不是最强guarded新增收益、30层或完整机器人性能。当前源码9b5dec45…、DSO2f333d8d…，完整身份见SELECTED。108CTA×8warp，112384B shared/CTA、255寄存器/线程、静态local1016B，数学tile3408/3480/3408、grid同步27。

新增当前阶段表两进程测量：改良插桩资源相同、完整层扰动≤0.462%，所有记录的对应阶段输出与输入/KV/RNG有限门通过。原Torch热阶段对自有核内阶段诊断：六小投影约0.68–0.74×，LN3约0.59–0.60×，FFN四段合计约1.01/1.01/1.03×，Self attention约4.60/4.52/9.27×。FFN存在重叠，不拆分相加。缓存上下文不同，这不是严格单原语或可省墙钟分解；精确生产独立FFN仍UNKNOWN。首次测量副本布局扰动和两种独立FFN副本的矛盾结果保留，不挑有利数字。

用户确认远端不允许管理权限和GPU计数器：不再尝试远端NCU、提权或改驱动。远端只用CUDAEvent/软件时间线；本机当前被他人任务占用，NCU0启动。新增GPU workers238.919秒、CPU编译904.875秒；保守计准备/等待/清理2712.273秒，累计 **15315.883/36000秒**，剩余5.746小时。10/23 00:43清理完成；旧00:58截止/失败/预算保留。四科学权限和production_admitted仍false。

M40截至2026-10-23T00:45:25.973283+08:00：GPU1状态RUNNING_COMPLETE_DEV_EPISODE，完成15/1800，T/C/P各5/5成功；仅早期运行状态，不能外推总体SR。5完整配对初态/首两chunk审计通过，active open_microwave-valid000-P。GPU0固定保留算子，不借GPU0。600有效初态与优化reset bank已备齐，原固定队列自动接续，每30局shard归档；实施/配对失败停止不自动重试。新160GPU小时/最长10天、T0=10/22 22:03:22；旧24小时账本与截止不改。新面板为扩大DEV，不是独立TEST。

| ID | 阶段 | 状态 | 当前证据 | 下一步 |
|---|---|---|---|---|
| `M39` | 单层算子 | in_progress | 内联/尾K64采用，完整阶段表与资源闭合 | 先讨论六小投影及小阶段差距，勿直接恢复全层ready |
| `M17` | LIBERO证据 | blocked | 四套6000局齐；独立性/配对/长尾缺口保留 | 不恢复过期作业 |
| `M40` | RoboTwin扩大DEV | in_progress | GPU1自动队列15/1800 | 定期核验进程/日志/完整三臂与shard |

[本轮结果](experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/RESULT.md)、[阶段表](experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/STAGE-TABLE.md)、[资源](experiments/single-layer-async-gemm-20261020/ffn-pipeline-inline-20261022/RESOURCE-AND-STOP-REPORT.json)、[RoboTwin检查点](experiments/robotwin-pilot-20261020/eval100-20261022/CHECKPOINT-20261023-0045.json)。10/23冻结身份与claim、10/28主表、11/01 18:00中文证据包；证据不足收缩claim，不制造PASS。
