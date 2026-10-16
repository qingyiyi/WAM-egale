---
kind: progress
cycle: hw-maintenance-20261016
plan: PLAN.md
status: closed
updated: 2026-10-16T00:38:58+08:00
current: M3
next: return_to_wam_research
---
# HW维护已完成

| ID | 阶段 | 状态 | 当前结果 / 证据 | 下一步 |
|---|---|---|---|---|
| `M1` | 隔离安装新版 | `completed` | 官方固定commit安装；33/33上游CPU检查；旧安装保留 | 新会话读取新版 |
| `M2` | 主题记忆与恢复入口 | `completed` | 10份短主题记忆，228原记录与六索引保留；原证据可追溯 | 按主题维护 |
| `M3` | 默认加载与回退 | `completed` | 用户授权账号默认切换；普通启动旧40停用、新10+10映射；legacy40恢复与全局回退验证通过 | 返回研究入口 |

[完整维护报告](../../../review/OMISSIONS.md)。Hook信任未改，四个自动Hook仍untrusted；文件恢复已验证，不冒称宿主自动Hook执行PASS。

宿主项目级禁用无效曾导致重复；在说明账号级影响并获得用户明确同意后才切换。原state/cycle/rules、tile和研究冻结源不改，未新增GPU作业。
