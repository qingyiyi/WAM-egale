# GitHub仅保留main：完成记录

时间：2026-10-18T23:33:52+08:00。用户明确GitHub仅作为本地代码/HW与当前进度供网页端读取的镜像。

- 六个原发布分支（含main）的tips均被最新公开快照ca5f37e包含；远端四套件tip112e1e9和更晚本地tip e2b037b均为其祖先。
- main由b0dfac3快进到ca5f37e并实际推送，未force-push、未制造内容合并或改变科学结果。
- 核对远端main后删除其余5个远端分支，同时安全删除5个已合入本地分支。
- 两个旧worktree只detach原4f04b35/689a9d1，未更改其中任何工作文件或删除目录。
- 远端default HEAD仍指main，远端与本地均仅main分支。旧固定commit仍由main历史可达。
- 后续直接从主发布工作树更新main；本地代码/HW为依据，网页单次审查固定所读commit。

[事前tips及祖先核验](REFS-BEFORE.json)、[删除后实际核对](CONSOLIDATION-RESULT.json)。发布新README/入口/长期规则不改变原FAILED、阈值、retention、默认A_off或四项false。无GPU、独立SHA、模型作业、环境迁移或运行任务中断。
