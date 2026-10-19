---
kind: discussion-summary
cycle: wam-research-20261016
updated: '2026-10-19T18:37:38.352270+08:00'
---
# 当前讨论与授权

用户明确研究tile化算子、persistent kernel基础上的tile调度，并建议参考MegaMoE；最新补充只需完成其思想，不要求复刻官方实现，要求综合HW提出设计。已用Maintain保存为当前算子目标/偏好；具体tile大小、队列、FFN路线和收益不是用户已确认事实。

用户另要求真实Transformer结构/输入输出以便协作者研究。交接区分WAM block与TF draft、M32/224与每候选Q16、pool与可见K、CFG共享历史与私有KV，提供完整绝对源码路径及原件缺口。M30仅CPU完成，未实现新persistent kernel。

此前用户要求完整block两份均为自有算子：M28已交付完整顺序原生链，cuBLASLt与Torch运行时依赖公开，有限精度/性能范围明确；从该交付继续优化optimized，baseline留作对照。旧attention-only/直接Torch复制阶段已被取代，旧FAILED保持。

用户进一步指出FFN-first整体拆法偏离Transformer本身，要求把self/cross-attention也纳入tile子图。已保存为设计范围反馈；当前建议按完整DAG定义任务，attention以输出query tile为首版任务单元，内部保留QK/在线softmax/PV，逐步接相邻投影/归约就绪。具体方案未运行，不扩展生产权限或原资源账本。原tile/投机源/保护state/rules不改；只baseline/optimized两份活跃实现。

用户要求远端完成三套先报成功率，已交付每臂500局：Long483/484/473、Goal463/464/467、Spatial472/466/464。连续TF接近Target但非全面占优，PF2不一致更好。描述性覆盖不解除严格配对失败、曝光/组件审计缺口，不推断非劣。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1。BAAI仅~/cxz，AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲，精度可共享；不杀他人/改服务/时钟。 用户选择等待他人GPU自然退出，已授权共享精度、计时无争用，不反复要求协调窗口。没有为填满GPU重复任务或改变同卡配对。

10/22 go/no-go、10/23候选/API/engine冻结、10/28主表、11/01 18:00中文证据包；中文正文会话交飞书，不新建本地论文稿。只main发布核对源码/HW，旧固定科学commit保留。

旧完整讨论见[归档](history/M30-before-handoff-20261019/DISCUSSION-SUMMARY.md)。
