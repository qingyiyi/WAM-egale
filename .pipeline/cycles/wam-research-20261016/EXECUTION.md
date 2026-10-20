# 最近执行 checkpoint

2026-10-20T21:59:48+08:00，北京。current M39，parallel M40；[PLAN](PLAN.md)。

1. 最新请求明确继续RoboTwin并先分析算子，按新有限profile和未开始42清单登记；未重置旧账本/截止。
2. GPU0当前context128副本插桩12block、150原语，加native12完整block。输出/中间量/KV有限逐位通过；当前源码和研究二进制未变。
3. GEMM占49–57%，cross-K RMS和pointwise开销突出，长KV attention占23%；NCU硬件counter拒绝访问，未知项保留。约23.2秒GPU生命周期，owned结束。
4. 首六Robo监督、114固定初态及bank核对完成。GPU1 21:37:02启动剩余42完整Target自动队列，6小时帽。现场10/20 21:59:00北京：新增5/42完整终态、5成功，当前`open_microwave-valid01`；原首六5成功1失败保留。
5. 更新源码解释、HW和main公开小证据；本轮不新增算子数学改动或自动训练。

[算子结果](../../experiments/single-layer-async-gemm-20261020/context128-profile/RESULT.md)、[终态](../../experiments/single-layer-async-gemm-20261020/context128-profile/FINAL-AUDIT.json)、[Robo现场](../../experiments/robotwin-pilot-20261020/target-remaining42/LIVE-SNAPSHOT.json)、[上轮checkpoint](history/context128-profile-start-20261020/EXECUTION.md)。

6. 用户最新要求先对齐各阶段与原版性能差异：已记为guideline，下一步增加原版/自有独立/嵌入整层三路径同边界矩阵；尚未启动新测量。
