---
kind: decision
name: 算子先直接复制Torch调用链完成复现
level: constraint
scope: cycle:wam-research-20261016;operator_baseline_reproduction
status: active
updated: '2026-10-19T09:46:57.535994+08:00'
supersedes:
  - 在基础复现交付前同时追求独立CUDA基元、复杂多引擎对照与后续调度的执行优先级
sources:
  - 用户最新明确只要求复现，可直接抄Torch实现，并质疑速度精度对齐为何久未交付
---

# 先交付直接复制的Torch等价基线

当前优先目标是独立、可替换、可审阅的复现版本，与原始Torch调用链保持精度和性能。用户明确允许直接复制/复用Torch实现，不要求此阶段先完成自有device基元或调度创新。先复制原forward/调用顺序，保留dtype、layout、参数、cache和RNG语义；不为“自研程度”额外包装或提前改数值边界。

已有有限精度通过与计时不确定分开报告。原A/A失败、原门和所有负结果保留；计时噪声不能被说成代码复现失败，也不能包装成已获得稳定正式性能证书。基础复现与后续kernel脱离Torch/融合/持久化调度分阶段，不改变原四false、tile保护或机器人质量门。

此决定不新增训练、机器人episode、正式TF性能或资源采购授权；本次单block接续仍在既有120分钟/5000调用/1GiB内。临时数字与进度写Experiment，不写长期规则。
