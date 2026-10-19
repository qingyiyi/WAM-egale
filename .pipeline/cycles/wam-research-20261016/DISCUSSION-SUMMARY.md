# 当前讨论与授权

最新用户要求算子只保留baseline与当前优化两份，代码命名和结构简洁；先做上一清单1（整模型复现）和2（性能基线），完成后一起讲解。随后明确投机侧代码禁止改动。因此新开发位于独立`算子研究`；只删确认无投机依赖的旧算子代码，原tile/shared-prefix等投机依赖只读保留。实际评估不因算子整理迁移/改入口。当前优化是原数学block加独立CUDA SDPA，不重新命名为新融合/调度创新。

M25是用户新任务的独立有界作业，CPU≤2/全作业3小时，10:14:26起至13:14:26，GPU生命周期≤1200秒、full30≤150、block≤500、新增≤1GiB；等空闲后两独立进程，不与他人争用/杀进程。输入来源明示：B14真实full30 fixture；B2 full30是同root CFG0/7派生、普通B2路径，非原始顺序轨迹；原B2/B14真实block输入另覆盖。旧M24起止/412调用/原AA失败均不重置。

用户已批准Object仅补固定14局/56000 forward，旧16有效/旧14失败和5876实际消耗保留，原M17起点/10月21截止不改。新14准入/ON已完成，匹配Target200只作baseline、未用于ON训练；全500继续，新health-v2每5分钟巡检。NVML/容器PID未证对应，额外槽不强行放行；不自动重试。BAAI写入永远仅~/cxz。AutoDL不再SSH，关机UNKNOWN。用户不再迁移开发，本机供算子；新机Codex安装不等于会话迁移。

TF64/TF_ON、normalizer、tau、K2B2、bonus/prefix_then_cost、BF16模型/FP32打分不变。PF2=SST完整policy，T50更新/51次action含末尾，非bonus或状态恢复；reset/cache不计policy周期。四套同等512-token BF16优化reset；cold另列，actual796/800罚分分列。Long首轮500已完成；三轮为偏好，后两轮seed/预算未冻结。T25/T17对照需另登记，video20同网格。

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。旧NO_GO/retention/FAILED/paused72/96与MLP0/22缺收据/default A_off不改。TF64/TF_ON、tau、K2B2/bonus/prefix_then_cost、BF16/FP32语义不变。PF2=SST完整policy，T50更新/51action调用含末尾。所有臂用512-token BF16优化reset，cold另列；actual796和失败罚分800分列。

投机代码/配置/服务/运行依赖不改。原tile是shared-prefix构建依赖，原位只读保留；保护state/cycle/rules与原tile HW不动。不自动新训练、120/独立TEST、被阻塞的新TF正式45chunk、架构搜索/第二环境/撤回融合。BAAI所有写入永远限~/cxz，不杀他人/改系统/时钟，AutoDL不再SSH。GitHub仅main当前镜像；中文论文会话交飞书，不建初稿。


GitHub仅main当前进度镜像，固定科学commit/原记录保持身份；中文论文逐段会话交飞书，不建初稿。10/22质量/系统go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。旧授权/暂停/已终止预算不能自动恢复。更早完整讨论见[history](history/M25-before-operator-consolidation-20261019/DISCUSSION-SUMMARY.md)。
