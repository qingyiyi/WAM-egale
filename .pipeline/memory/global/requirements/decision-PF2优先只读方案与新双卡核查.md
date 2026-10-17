---
kind: decision
name: PF2优先只读方案与新双卡核查
level: constraint
scope: cycle:wam-research-20261016
status: active
updated: '2026-10-17T10:40:21.117763+08:00'
supersedes:
  - '网页返回后默认优先D1/D2 teacher闭环诊断或追加训练的提案顺序'
  - '旧AutoDL禁止连接规则对本次新提供实例的适用解释'
sources:
  - '2026-10-17用户提供网页版D0/D1/D2及随后PF2两臂提案，并提供新双卡实例供核查是否加速'
---

# PF2两臂方案优先，只读核查新双卡资源

当前以固定4f04b35 paused72审查TF_ON连续投机与PF2方案。后一个PF2提案替代前一个默认teacher诊断/追加训练执行优先级，不叠加预算；原teacher证据保留。实际研究目录后续77终态另列，不覆盖固定72快照，也不改旧22清理缺证、FAILED、retention和四false。

用户新提供双卡实例连接，允许只读核查资源和既有环境，不保存凭据；这不启动GPU/数据采集/训练/TEST、不部署代码或安装依赖。提案mode=READ_ONLY_PLAN、execution_authorization=NONE仍有效。先交付源码与checkpoint清单、建议diff、固定六对计划和具体预算，实际代码/GPU执行须用户另行明确批准。旧10/17 10:00实验窗口已结束，不继承余额或deadline。

TF64/TF_ON权重、normalizer、tau、树、bonus、scheduler和数值边界固定，仅计划PF开关差异；不开PF3/4搜索、不加MLP/Target完整机器人臂、不重训。新两卡若获执行授权，只做独立完整pair分片，同pair同物理卡，不做小draft DDP；并行需处理全局lost_success止损与累计预算。

## 后续授权

本文件保留原只读阶段。用户随后批准实施及12h AutoDL窗口，执行权限部分superseded_by [本轮执行决定](decision-PF2源码执行与AutoDL十二小时窗口.md)；原PF2优先顺序和科学边界继续有效。
