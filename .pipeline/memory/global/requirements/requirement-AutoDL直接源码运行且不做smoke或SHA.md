---
kind: requirement
name: AutoDL直接源码运行且不做smoke或SHA
level: constraint
scope: cycle:wam-research-20261016
status: active
updated: '2026-10-17T10:52:00+08:00'
supersedes:
  - '网页PF2提案中在AutoDL执行独立CPU预检与12chunk GPU门检的建议'
sources:
  - '2026-10-17用户明确：AutoDL不要smoke、不要sha、不用做测试，直接在源码上运行和修改'
---

# AutoDL直接使用项目源码，不做smoke、测试脚本或SHA

本Cycle的AutoDL工作遵从用户最新要求：不运行smoke、独立预检或测试脚本，不计算SHA或整套模型hash；如进入获准的执行阶段，直接在独立研究目录使用和修改项目源码。必要路由、计数、初态与资源审计随正式完整任务记录，不把未执行的门检写成通过。

禁止范围包括启动loader、manifest或fingerprint helper内部隐式执行的SHA，不能只去掉命令行校验而保留这些调用。原件保持不变；独立入口的来源记录与未经远端复验的限制须如实说明。

这条要求替代网页版方案中在AutoDL先做独立CPU/GPU门检的建议；不删除历史失败、固定快照与既有本地源码身份清单，也不据此改变质量门。原READ_ONLY_PLAN阶段已由后续[12h执行批准](decision-PF2源码执行与AutoDL十二小时窗口.md)替代：本轮源码/部署/固定六pair GPU已批准；训练仍未批准，不能继承已到期的旧10:00窗口。不得保存SSH凭据。
