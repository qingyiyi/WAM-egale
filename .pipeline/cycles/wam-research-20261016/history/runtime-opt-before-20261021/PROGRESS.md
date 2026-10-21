---
kind: progress
cycle: wam-research-20261016
updated: '2026-10-21T22:14:16.314145+08:00'
plan: PLAN.md
current: M40
next: M40
---
# 当前研究进度

| ID | 阶段 | 状态 | 证据 | 下一步 |
| --- | --- | --- | --- | --- |
| `M17` | LIBERO证据收口 | in_progress | 6000局齐，非独立TEST | 曝光/首两S/长尾缺口保留 |
| `M39` | 简易单层先对齐 | in_progress | 入口再精简，9签名及普通/debug有限逐位；性能无稳定退化 | Self-Q编译边界→FFN→DAG |
| `M40` | RoboTwin迁移 | in_progress | 用户批准两项工程优化，GPU1完整任务对照运行中 | 逐项等价/耗时→组合；再修订100次预算 |

## M39：入口精简与代码带读

[本轮结果](../../experiments/single-layer-async-gemm-20261020/readability-20261021/RESULT.md)、[计时](../../experiments/single-layer-async-gemm-20261020/readability-20261021/WHOLE-LAYER-COMPARISON.csv)。Python直接绑定参数，移除两个无引用设备函数；普通forward不构造完整debug字典/资源统计。CUDA数学、tile、cache语义、sync及错误传播不变，baseline/投机/原tile不动。当前源043c7759、DSO2a8a250e，本机/远端同一验证二进制。

9真实签名（B2action/B14action/B2video×3KV），原Torch输出/23共有中间量/18有效KV转移及输入/RNG有限逐位通过；9组普通/debug返回和状态一致。两个进程三KV整层Graph变化−0.132%～+0.103%，无稳定退化，不称新增设备端加速。完整层约2.506/2.722–2.724/2.974–2.978ms；不是相对Torch新倍数或机器人E2E。

[前投影结果](../../experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/RESULT.md)的1.88%–2.08%改善、Self-Q退化与旧快独立参考保留。FFN原语差距未解决，ready-DAG未实现。本轮GPU55.563769秒、CPU构建133.949637秒；M39累计2538.679523/36000秒。owned已退出。[带读prompt](../../../算子研究/wam_ops/optimized/READ_CODE_PROMPT.md)按入口→整层→tile→warp→数据流/同步逐段讲。

## M40：官方复现与慢因已获得证据

旧DEV90六任务各5初态，T/C/P成功25/24/23，各30局；30组首两S配对通过，非独立TEST，不支持质量保留。用户已将扩大目标从50改100，六任务三臂1800局。用户对新增160GPU小时请求回复先诊断慢因，**没有批准新增预算，1800局未启动**。[具体方案](../../experiments/robotwin-pilot-20261020/microwave-reproduction-audit-20261021/EVAL100-PLAN.md)。

[原Target两局核查](../../experiments/robotwin-pilot-20261020/microwave-reproduction-audit-20261021/RESULT.md)：固定官方11文件一致，valid14失败/valid16成功均复现；原_infer无draft Runtime，所有命令、初态与保存的首次两policy/RNG均相同。失败门最多0.048515rad，判据0.942478。只支持所覆盖两例，非paper全部复现、非新质量N；full-episode uncached bank对照UNKNOWN。

[90局账本及完整profile](../../experiments/robotwin-pilot-20261020/runtime-profile-20261021/RESULT.md)：客户端平均262.81秒；RPC之外60.02%，模型reset90次合计1.115秒、仅0.0047%。18条失败占55.11%耗时；微波炉15局占48.36%。单成功局profile约253.6秒：RPC102.5，take_action72.5（其中双臂规划60.5），外部get_obs46.6，setup18.7，其余13.4。累计/嵌套口径分列；该局命令与未profile相同，插桩有约5.36%扰动，不作正式性能。

优化候选尚未实施：front_camera渲染/取回未被模型使用；planner每局初始化/预热17.2秒；CuRobo sync_cuda_time中的等待不能直接当39秒纯开销删掉。先保持三路受用图像/物理/RNG/命令等价核验，再谈吞吐。没有修改官方物理、成功判据、驱动或渲染质量。

计数修正：原actual_commands为提交调用数，终态后会无推进返回。T/C/P提交12224/13376/13408，环境真正推进11808/12927/12958；新派生语义表分列，旧数据/罚分不改，不能说物理越过horizon1500。

原两例GPU1191.369558秒，本次profile285.933544秒；M40累计51483.682640/86400秒，原10/22 01:26:04.706686硬截止不改。owned已清理，当前两GPU无本轮作业。新增160小时未授权，不自动扩大。

## 共同边界

四权限false、paused72、A_off、旧retention/FAILED和保护文件不动；BAAI仅~/cxz。LIBERO每方法4×10×50=2000局，三方法6000；不是论文每方法三seed重复。Long T/C/P483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499，各500。非独立TEST。

10/22阶段go/no-go、10/23冻结身份、10/28主表、11/01 18:00中文证据包；扩大不能完成时如实PARTIAL，不以进度为由放行质量门。[前完整记录](history/readability-and-runtime-before-20261021/PROGRESS.md)。

2026-10-21T22:38:33.151264+08:00：runtime-opt-20261021已在GPU1启动（guard793475）；最多4个完整episode/5400秒生命周期/1GiB，原累计与硬截止不变。此前CPU-only CuRobo导入因其类默认参数需CUDA不可用，已记录为CPU预检限制；不伪造import PASS，真实导入随完整episode核验。
