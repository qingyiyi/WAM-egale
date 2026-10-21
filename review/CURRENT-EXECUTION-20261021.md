# 当前执行证据（2026-10-21T13:45:55.363261+08:00）

算子FFN1尾块一次候选保留：完整单层两进程三KV改善0.19%–0.50%，仍慢14%–29%。9签名有限逐位通过，max_abs=0；跨阶段ready未实现。当前独立/嵌入同边界差约−1.23%至+0.36%，不能把剩余差额全归同步。动态spill/L2/DRAM/occupancy UNKNOWN。

- [完整层与四阶段表](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn1-tail-control-20261021/RESULT.md)
- [源](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)；SHA256 `df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a`
- [资源](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn1-tail-control-20261021/RESOURCE-AND-STOP-REPORT.json)、[旧登记字段说明](../.pipeline/experiments/single-layer-async-gemm-20261020/ffn1-tail-control-20261021/REGISTRATION-NOTES.md)

RoboTwin DEV90：3/90完整终态，当前`stack_blocks_two-valid14-C`；状态`RUNNING_COMPLETE_DEV_EPISODE`。GPU1单worker，固定30初态×T/C/P自动接续。前启动导入错误0teacher/0episode保留；用户明确允许新namespace补启，原24h累计及10/22 01:26:04截止不变。

- [本次补启授权/边界](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/RETRY-AUTHORIZATION.md)
- [固定计划](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/PLAN.json)、[状态快照](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/STATUS.json)、[真实导入核查](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/CPU-IMPORT-REVIEW.json)
- 初态、C/P首两S有效KV摘要/RNG/动作/路径与分支计数门；基础设施或隔离失败停止新局，不自动重试。

CURRENT18的13/18是CONT采集，不能改称新ON闭环SR。新权重部署CAL不是机器人质量门。DEV90和LIBERO既有面板非独立TEST；四项权限false。raw KV/权重/原始场景PT、profile trace不公开；不能凭仓库声称重算所有tensor。旧paused72在RESULTS.json保留。

下一步：核对首完整三臂及隔离门，后台接续；算子量化原语/等待后做真实DAG补位。main是动态进度镜像，审阅时固定本次commit。
