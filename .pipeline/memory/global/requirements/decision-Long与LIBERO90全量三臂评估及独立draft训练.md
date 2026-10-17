---
kind: decision
name: Long与LIBERO90全量三臂评估及独立draft训练
level: constraint
scope: experiment:libero-fullscale-20261017
status: active
updated: '2026-10-17T13:45:00+08:00'
supersedes:
  - 'M15结束后默认优先teacher诊断的下一步建议，仅针对后续工作顺序'
sources:
  - '用户明确要求Long与LIBERO90的Target、TF、TF+PF2全量结果一起汇报'
  - '用户明确90的draft需要重新训练'
  - '用户要求AutoDL所有卡并行使用，最后给耗时预估'
---

# 新增全量三臂研究，90独立训练draft

用户明确授权将后续工作扩为Long与LIBERO90的Target、TF、TF+PF2完整评测，并为90重新训练自己的draft；可以利用现有AutoDL全部两卡并行。Long冻结现有TF64/TF_ON及原norm/tau/树/bonus；同一套权重分别比较连续与PF2，Target另跑同批匹配基线以计质量和实际成本。

90先核实同代、真实可下载且训练身份明确的Target checkpoint，再登记教师数据与TF64训练；未找到前标BLOCKED，不将Long权重默认为已适配90，也不拿Fast-WAM/LingBot-VLA冒充同代目标模型。用户没有要求重新训练大Target。可独立推进已具备模型的Long部分。

完整覆盖是新研究namespace，不更正旧六pair投入NO_GO、原FAILED/retention/22缺证、旧四false或全局默认。新全量面板不会因一个自然算法失败跳过其他case；基础设施/数值身份错误、资源和安全硬限仍需停新任务，保留PARTIAL。来源含已见FIT/CAL/DEV的官方init必须明示，不能把全部官方50初态称作全新独立TEST。

资源后续授权见[AutoDL用至完成](decision-AutoDL全量研究使用至完成.md)：用户明确答复可用至完成，替代本新scope等待答复时临时沿用的23:11限制，旧M15账本不改。无新增机器采购；AutoDL仍无smoke、独立测试或SHA，所有对照默认已验证完整reset bank，必要审计随完整任务。正式性能权限与本机SXM4边界不借探索性计时改写。

具体任务清单、forward/存储上限、源版本和生命周期另登记，不把本决定当成无界训练、结构搜索或任意新环境授权。
