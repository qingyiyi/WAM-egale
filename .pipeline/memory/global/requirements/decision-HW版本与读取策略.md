---
kind: decision
name: HW版本与读取策略
level: guideline
scope: project-tooling
status: active
updated: '2026-10-25T16:40:28.148883+08:00'
supersedes: []
sources:
- .pipeline/maintenance/hw-upgrade-20261016/GLOBAL-SWITCH.json
- .pipeline/cycles/hw-maintenance-20261016/DISCUSSION-SUMMARY.md
---

# HW版本与读取策略

用户已明确允许账号级升级：默认使用固定v15插件，旧40项技能按canonical文件路径停用；保留原v12物理安装及 `hw-codex --hw-legacy` 兼容入口。工具默认影响该账号，其他项目记录不自动迁移。

本项目先读INDEX、约束级记忆和聚焦Cycle；其他主题与原证据按需读。原228记录、六分类索引、长PROGRESS和tile原始HW保留，旧状态不作为当前计划。不要再自动重建六索引并全量注入。

原state/cycle/rules及结构化规则继续受保护。现有会话不强制重启；新会话读取新配置。Hook信任本轮未变更，未信任时依靠普通文件恢复，不将Hook程序检查说成宿主自动执行已通过。

用户2026-10-25再次明确“整理一下本机的HW，精简一下”。本项目默认只读短入口/现行约束/当前Cycle和一个相关主题；重复汇报归并为当前事实、阻塞、下一步与证据链接。超过体量目标或存在新旧状态冲突时按语义重建，原全文归档可追溯，不以精简为由删除失败/原始证据、修改保护源或干预运行实验。动态数字不复制到长期记忆，不再向四份Cycle文件重复追加整篇汇报。
