---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-24T00:45:50.961788+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前进度

FFN2有序split-K本轮完成并研究采用。M64×N128/K64、四MMA warp、原cp.async主循环和七段BF16提交不变，仅将96输出tile×7段变为672计算任务；170个常驻CTA领取，96计数器按序提交，新增384B全局计数。

三真实B14/Q16/block0的26中间量＋输出/有效KV/输入/RNG及public默认/非默认流有限逐位通过。两独立进程同轮原Torch/自有短中长1.1493/1.6245/2.0497×，相对冻结GRAIN完整层降5.404%/5.174%/4.474%；对Guarded短快1.54%、中慢0.22%、长慢2.56%。全eager无Graph，冷CrossKV另列。

独立FFN2＋bias＋门控残差近原Torch，较串行快约1.43×；纯Linear未对齐，完整融合FFN仍慢15%–16%。三个小投影仍约0.67–0.71×。完整17阶段本轮重测并在对话打印；插桩与正常完整层分开。TMA未启用，旧三个TMA流水负结果和GRAIN独立A均保留。

正常/profile/probe构建及七GPU进程均成功，无GPU失败/超时；7389完整block、5094独立FFN2。owned20身份全部退出，5090显存0；六份源码/二进制回本机byte一致。保守计30分钟，累计31486.413/36000秒、余额1.254小时，旧deadline不重置。四科学权限和production_admitted=false。

双A100 RoboTwin00:40只读见两卡模型/模拟上下文、CPU巡检正常；不改队列、不重算SR。下一步建议有界处理Self/Cross输出与Cross Q的有序K分段，再隔离TMA预取；本次不追加作业。

[结果](../../experiments/rtx5090-block-port-20261023/ffn-splitk/REVIEW_RETURN.md)、[完整表](../../experiments/rtx5090-block-port-20261023/ffn-splitk/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/ffn-splitk/SELECTED.json)。
