---
kind: experiment
id: tf-training-support-20261016
cycles: [wam-research-20261016]
status: completed_cpu_diagnostic
---

# 固定 TF64 后的递归监督与训练覆盖核验

用户已固定TF64为主模型，本轮仅围绕TF效果，不新增MLP或算子作业。基线仍为`b0dfac3a02f65de33a1217246b83003615166640`。

事前登记：[PREREGISTRATION](../../../投机采样/results/runtime/p0-tf-training-support-cpu-v1-20261016-r1/PREREGISTRATION.json)。输入为完整task4/init26 TF_CONT3000的50个chunk日志、原FIT174/14997混合池及固定normalizer。CPU12–13、2线程、900秒/64MiB，CUDA禁用；没有模型forward/backward、新teacher或环境动作。实际33.83秒完成，前台进程已退出。

## 真实结果

[RESULT](../../../投机采样/results/runtime/p0-tf-training-support-cpu-v1-20261016-r1/RESULT.json)：904树、2712个预测query；其中根904、真实到达parent770、反事实parent1038，真实访问query共1674。保存teacher/proposal重算与原GPU六score最大差`2.98e-8`；这是这些已保存904树的核验，不外推历史未保存velocity。

- 根query的局部最佳头通过85.18%；真实到达parent为95.84%；反事实parent65.99%。它们是不同条件下的局部指标，不能当整树/任务成功率，也不能直接与不同来源的CAL95%对比作因果结论。
- 本例高σ根query（σ≥0.6）50个均被原阈值拒绝。它提示一个明确的TF拟合/启用位置问题，但这些query没有提交draft，不能据此解释整局失败。
- 同task及相同scheduler步骤重加权后，当前state/previous的固定normalizer RMS没有显示简单的尺度爆炸；真实parent残差均值约0.817，旧混合池约0.863。**这不支持直接宣布严重递归分布偏移已经成立**，也不能排除更细的联合分布、条件歧义或多步误差。
- 历史14997仍是3906 A_off＋11091旧TF，不能称当前TF_CONT3000的on-policy数据。当前2712条已解盲DEV诊断query不进入新FIT/CAL。

## 对TF主线的含义

TF64结构继续固定，当前权重/训练配方未验收；MLP仅后续等容量baseline。下一步继续聚焦TF预测误差的sigma/递归分层和监督对齐，再以具体机制支持唯一一次新FIT/CAL及TF-cont/TF-on公平训练。不得因为用户固定TF主模型，就把观察性差异称为已证实根因，或把训练准入/质量字段强改true。

本轮`training_trigger=false`，没有新训练或采集启动，原四权限false，120未启动。前一步数值反事实见[M12结果](../draft-teacher-path-20261016/RESULT.md)；未来TF流程见[下一工作包](../draft-teacher-path-20261016/NEXT.md)。必要MLP公平对照后置，原FAILED/阈值/retention/默认不变。
