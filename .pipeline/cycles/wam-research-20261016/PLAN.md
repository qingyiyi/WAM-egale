---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-25T06:14:12.252638+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前计划：质量接续、成本证据和单层算子

- [x] M41-RTX5090-SPEC-SPEED：44组三臂132局主表，135完整终态含3条外部比较记录，2条已执行PARTIAL保留；结果回传/整5090临时根清理完成。[结果](../../experiments/rtx5090-spec-speed-20261025/RESULT.md)。不是质量PASS。
- [x] H02-LOCAL-STORAGE：25564文件核验归档后释放120.619GiB；TMPDIR统一，原tile/源码/失败表保留。[恢复与结果](../../maintenance/storage-placement-20261024/RESULT.md)。
- [x] M39-SM120-FFN-NEXT：唯一PAIR2048x2整层慢6.25%–7.89%，不采用，保留SELF-QKV-TMA-4MMA。[完整结果与17阶段表](../../experiments/sm120-ffn-next-20261024/REVIEW_RETURN.md)。原四候选负结果也保留。
- [ ] M40-EVAL100／STORAGE-RACE-RESUME：双A100持续固定六任务×100初态×三臂。当前459/600完整组，两worker alive、HALT为空；自动领取未开始pair并接续原已授权281恢复，不重跑406/T OOM。原160GPU小时/10天与更早10/29 14:03:22 guard不变。[接续与缺口](../../experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/PLAN.md)。
- [ ] M40-MICROWAVE-FINAL-DIAG：等微波炉100组三臂完整后先核官方Target复现，再定位额外失败；原CPU提醒/队列继续。[计划](../../experiments/robotwin-pilot-20261020/eval100-20261022/microwave-completion-20261023/PLAN.md)。
- [ ] M39-NEXT：进一步FFN2/投影调度仍待有界方案。当前全局阶段同步保留，不自动恢复旧完整DAG、大融合或失败驻留映射。
- [ ] M17-EVIDENCE：旧严格配对与FIT/DEV暴露缺口未追溯补齐；禁止借新速度面板改旧证书。

## 依赖与主线

投机改变工作负载形状→候选感知执行→质量约束下有效成本，是待证因果链。M41没有集成自研单层算子，也未完成shape×engine因果矩阵；不能相乘旧倍数。TF结构可暂冻，权重/调度未全局质量放行。

后续优先Robo分叉前一致性诊断；新训练、独立TEST或正式45chunk不自动启动。现有四false、旧FAILED/retention、state/cycle/rules、原tile保护不变。

M40-DUAL-UNTIL10等10/23临时停派时段已由用户双A100持续接续决定替代；原计划仍在[完整旧PLAN](PLAN.before-M41-20261025.md)，不是事后改原deadline。其余旧已完成项/原链接都保留于同目录before-M41文件。
