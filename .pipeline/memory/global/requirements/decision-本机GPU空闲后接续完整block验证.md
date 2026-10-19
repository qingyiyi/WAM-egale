---
kind: decision
name: 本机GPU空闲后接续完整block验证
level: constraint
scope: cycle:wam-research-20261016;experiment:wam-block-baseline-20261019;phase:authorized_gpu_followup
status: active
updated: '2026-10-19T08:46:58.716366+08:00'
supersedes:
  - decision-算子先建立完整block顺序基线并同步HW.md中本轮GPU尚未授权的状态；原CPU账本不修改
sources:
  - 用户最新明确：你监控本机gpu，有空的时候就上去跑，另外给我汇总昨晚的进度如何，两台机器都要
  - 前次已提出完整block GPU验证与计时不超过120分钟的具体范围
---

# 本机空闲GPU接续完整block验证，同时核验两机进度

用户已授权在确认本机GPU空闲后接续M24已准备的完整block数值、有效状态、A/A及配对计时，不再重复询问同一范围的许可。每个新作业独立登记预算与来源，不借M22或原CPU阶段余额；本次按既有具体建议限单A100、CPU不超过2核、生命周期不超过120分钟、新增不超过1GiB、完整block尝试不超过5000次。

持续低开销核验占用；他人任务自然结束才接入，不代终止他人进程。正式计时仍需无争用、A/A稳定及清晰边界。利用率与进程/显存不一致等动态异常写实验，不能直接签资源清洁通过。

本次范围不包含新机器人episode、训练、完整模型加载、生产替换、persistent scheduler或新质量放行。四项科学false与原tile、保护源、旧失败不变。同步核查本机与BAAI昨夜实际完成/停止/运行状态，不能沿用旧快照称正在运行。
