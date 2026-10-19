---
kind: discussion-summary
cycle: wam-research-20261016
updated: '2026-10-19T16:09:23.647800+08:00'
---
# 当前讨论与授权

用户最新要求整个block的baseline与optimized均为自有算子，并对齐原Torch性能/精度；已覆盖原“仅attention替换”和baseline不得改的阶段限制。先归档旧源，再完整顺序原生链、严格中间量/状态/30层、同口径性能。成熟cuBLASLt GEMM与Torch allocator/stream是公开依赖，不包装成自研GEMM或大型fusion。

只维护baseline/optimized，共用必要原语与工具。交付须有可复制完整绝对源码路径、真实源码讲解、实际替换与测量范围分开。参数容器不保留旧Torch数学fallback。投机生产和原tile不改，不用新有限通过覆盖旧FAILED。

用户要求等他人GPU自然退出，不反复协调、不杀他人/改时钟。已授权非性能共享；计时无争用，A/A、独立进程、原始值和中断全部保留。最终矩阵已齐，统一汇报/HW/main；不为获得PASS放宽原标准。

M26约3%为FFN1融合后的block延迟下降，不是自有实现慢3%；M27已分表纠正。M28是新完整native版本，不借旧attention/FFN1的完整模型或性能证书。新copy修正不是新的MegaTransformer创新。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

投机源码/配置/运行依赖与原tile只读；本轮按用户新授权可改算子baseline，旧源已归档。1875投机源与保护state/rules复核不变，不建缺失cycle.yaml。不杀他人/改服务/时钟；BAAI写入仅~/cxz；AutoDL不再SSH、关机UNKNOWN。仅main公开镜像。

远端既定质量继续自动接续、每5分钟CPU只读巡检；M17原10/21 16:56:31停新/17:11:31清理。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包不变。旧讨论见[归档](history/M28-before-final-native-result-20261019/DISCUSSION-SUMMARY.md)。

M28两份完整native/六actual30有限精度和双进程动态block性能已完成，GPU退出。当前尚无最强B14/Graph、完整新policy或E2E结果；不借旧证书。用户新增要求核验投机去噪论文：M29确认多项直接先例，候选感知系统机制仍需实证，不能称首次机器人投机/Transformer draft。M17保留同卡匹配T200约束；GPU0无合法后继时可以空闲，不通过重复Target或改协议占满。
