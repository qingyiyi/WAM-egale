---
kind: discussion-summary
cycle: wam-research-20261016
updated: '2026-10-19T14:49:18.383103+08:00'
---
# 当前讨论与授权

最新用户明确要求完整block的baseline与optimized均为原生算子，并对齐性能/精度。M28已新登记、旧baseline源码归档，当前可改baseline；投机与原tile仍保护。先顺序原生链及逐位门，再优化，不以ATen高层转调冒充。


用户优先定位短KV/B14自有实现相对Torch的慢点，允许直接核对Torch源码；要求可复制完整绝对源码路径，并确认测量是否覆盖Transformer。M27已有限核查收口：2.45–3.23%为可选FFN融合后的完整block耗时下降，不是默认复现回归。新增A/A两次资源中断PARTIAL、一次等待到期均保留；未改内核、未声称全矩阵或正式性能通过。

M28用户新范围优先：两份研究入口现接完整native block；27参数原位共享，数学无ATen高层转调。旧attention/FFN1结果仅历史，不借其30层证书。视频/驱逐/安装/30层与新性能分层验证。

本机等他人GPU自然退出，不反复协调、不杀他人/改时钟；本轮全部owned已退出，余量不自动接续。后续稳定计时、旧layer29/last缺输入异常需明确同源输入与新预算；不直接大融合或persistent。原M26失败、NCU偏差及M25严格筛查仍保留。

远端两卡每卡最多2worker，T200同卡配对约束保留，后继自动；CPU只读watch每5分钟。M17原10/21工作停16:56:31/清理17:11:31不变。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost、BF16/FP32冻结。PF2=SST完整policy，T50更新/51action调用含末尾；reset/cache不计周期。各臂512-token BF16优化reset，cold另列；actual796/800罚分分列。

投机源码/配置/运行依赖与原tile只读；M28按用户新授权可改算子baseline，旧源已归档。1875投机文件及保护state/rules不改，不建缺失cycle.yaml。不自动训练、120/独立TEST、新TF正式45chunk或第二环境。BAAI写入仅~/cxz，不杀他人/改服务/时钟；AutoDL不再SSH，关机UNKNOWN。仅main公开镜像；中文论文会话交飞书，不建初稿。

10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。更早讨论见[历史](history/M27-before-investigation-20261019/DISCUSSION-SUMMARY.md)。
