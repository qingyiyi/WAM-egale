# GitHub 收拢到 main

用户明确授权所有发布分支合并到main，删除其他分支；GitHub仅用于网页版阅读当前本地进度。

- [x] 获取全部远端branch tips，核对提交关系与三个本地worktree状态。
- [x] 保留各tip可达性，将最新公开快照和本轮发布约定更新到main并推送。
- [x] 远端main核验后删除已保留的其余branch；本地旧worktree仅detach同一commit，不删目录或数据。
- [x] 核验远端仅main、默认分支main、旧固定commit仍可达，保存收据。

CPU≤1，GPU0；墙钟上限20分钟，新增文档/收据≤10MiB。无force-push、无科学代码/门限修改、无GPU作业或独立SHA；不重置原实验预算或停止评测。清理仅限已并入main的Git分支标签，不清理项目文件。
