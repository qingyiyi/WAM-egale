---
kind: decision
name: BAAI的Codex安装与启动入口
level: reference
scope: project:WAM-egale;host:ssh.platform-multi.baai.ac.cn:2222;workspace:dx-calc1
status: active
updated: '2026-10-18T22:39:41+08:00'
supersedes: []
sources:
  - 用户明确要求在新服务器安装Codex，并询问迁移的效率
  - .pipeline/experiments/remote-codex-setup-20261018/RESULT.md
---

# BAAI 的 Codex 启动方式

在该主机使用 `~/cxz/bin/codex`，原生发行目录在 `~/cxz/tools/codex/releases`。入口固定 `CODEX_HOME=~/cxz/.codex`，缓存、状态、配置与临时文件也在 `~/cxz`。遵守已有[全部工作仅在~/cxz](../rules/requirement-新BAAI服务器仅在用户主目录cxz内工作.md)约束，不依赖外部 `~/.codex` 或全局 PATH 修改。

初装版本0.162.1；实际版本与认证状态以后动态核对。需要登录时用 `~/cxz/bin/codex login --device-auth` 在用户浏览器完成；不在记忆或公开仓库存凭据。使用说明位于 `~/cxz/tools/codex/README.zh-CN.md`。

这是用户授权的安装交付，不把询问迁移效率记作“全部开发工作已迁移”或新实验授权。安装不自动迁移原会话；新机运行中的评测树不应被最新版源码直接覆盖。
