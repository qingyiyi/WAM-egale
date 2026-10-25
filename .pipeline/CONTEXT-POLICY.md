# HW 读取与维护策略

## 默认读取

1. `INDEX.md` → `memory/INDEX.md` → `memory/CURRENT-CONSTRAINTS.md`。
2. 按任务确认当前 Session 的 Cycle；读取其 PLAN、完整的当前 PROGRESS、最近 EXECUTION、DISCUSSION-SUMMARY。本会话是 WAM；其他 active Cycle 不自动并入。
3. 只打开当前主题索引及适用约束原文，再按需读取实验计划/结果/源码。历史全文、旧六分类索引、全部memory或大JSON不默认展开。

现行约束短视图是派生入口，不替代用户指令、结构化规则、原授权和实验协议。执行前核对原作业预算/终态/资源；不能靠摘要放行。

## 体量与更新

- 短状态目标≤8KiB；四份Cycle文件合计目标≤16KiB；单主题入口≤4KiB；日常读取包目标≤24KiB。
- 只更新一份当前事实；PROGRESS记状态与证据，EXECUTION记少量检查点，DISCUSSION记已确认决定/待决问题，PLAN列依赖。不在四处复制整篇汇报。
- 短状态超过8KiB、最近事件超过15条或出现新旧状态冲突时，按语义重建。旧全文另存同目录历史文件并链接；不能只截前80行。
- 默认只读取一个相关主题索引。新增长期要求合并同主题Memory并标kind/level/scope/source；动态数字留在Experiment/Cycle。
- 历史归档不等于撤销规则或删除失败。每次重建核对四项false、默认/候选身份、失败、授权、截止、next和证据链接。

## 边界

不改受保护的state/cycle/rules、结构化规则、原tile、冻结实验源和运行作业。HW精简与大文件清理分开，不能以文档重复为由删实验数据。

账号维持固定v15语义入口；旧v12安装/legacy配置只供追溯。不升级、不重置、不运行旧全量sync，不修改Hook信任或系统配置。多Cycle时使用Session焦点，不能覆写全局保护状态来选择任务。
