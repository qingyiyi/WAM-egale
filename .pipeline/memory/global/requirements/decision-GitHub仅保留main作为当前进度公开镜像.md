---
kind: decision
name: GitHub仅保留main作为当前进度公开镜像
level: constraint
scope: project:WAM-egale;publication:github.com/qingyiyi/WAM-egale
status: active
updated: '2026-10-18T23:29:19+08:00'
supersedes:
  - 按experiment或review发布分支分别维护公开阶段快照的安排
sources:
  - 用户明确要求合并所有分支且只保留main
  - 用户随后澄清：把本地的推到main分支就行，github只是给网页版看目前进度的中转
---

# GitHub 只维护 main 进度镜像

`qingyiyi/WAM-egale` 是本地代码、HW与研究进度供网页版阅读的公开镜像。后续把核对后的当前公开内容直接更新到 `main`，不再自动新建 experiment/review 等发布分支。历史阅读使用固定commit；删除分支标签前核验所有有效提交已由main保留，不改写旧科学快照身份或实验结果。

本地研究工作区、实际作业与原始证据仍是执行依据；不能将公开main的变化覆盖回运行中或冻结的实验目录。公开范围继续排除凭据、私钥、权重、大tensor和不必要的运行文件；该发布决定不等于新增实验、清理数据或迁移开发工作授权。

当前主发布工作树为 `review_exports/WAM-egale-libero-fullscale-20261017`，在main；另外两个旧导出worktree只保留原commit的detached状态和文件，不作为当前发布入口。
