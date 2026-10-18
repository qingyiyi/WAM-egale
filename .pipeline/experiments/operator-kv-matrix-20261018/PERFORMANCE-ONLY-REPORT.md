# 限时性能采集：精度待验证

后续精度与完整query验证已完成，包含真实失败，见[统一结果](COMBINED-RESULT.md)。下文保留20:14性能采集时点的状态，不改原协议或计时样本。

更新：2026-10-18T20:14:53+08:00。状态：COMPLETE_PERFORMANCE_CAPTURE_VALIDATION_PENDING。

实际54/54固定点，单进程、3轮；684性能单元、18360 Event样本及等量同步墙钟样本。attention每轮10样本，Graph每样本8调用；block每轮5样本。完整raw保留。

用户明确剩余独占不足20分钟，优先采集性能、输入输出后验。本协议独立命名；旧7轮/双进程协议的31完成点、用户收尾与全部消耗保留，不追溯改为通过。

**以下均为探索性延迟。启动前可见进程/显存为空，但NVML利用率非零原因未查明；未宣称物理宿主独占。精度、有效KV和全部状态契约待验证，不据此集成或签性能/质量证书。**

下表固定展示每档layer0/first query，其余全部层/query与慢轮见CSV；它不是挑选最快层。attention列为Graph-core，block为真实完整block eager。单位ms。

| 输入 | 实际self K | Torch self | standalone self | Torch block | standalone block | shared block | guarded block |
|---|---|---|---|---|---|---|---|
| B2 video/short | 272 | 0.036288 | 0.035456 | 3.244032 | 3.459072 | NA | NA |
| B2 video/middle | 1136 | 0.056768 | 0.055552 | 3.167232 | 3.112960 | NA | NA |
| B2 video/long | 2144 | 0.081024 | 0.079872 | 4.206592 | 12.919808 | NA | NA |
| B2 action/short | 288 | 0.020480 | 0.019968 | 3.253248 | 3.061760 | NA | NA |
| B2 action/middle | 1152 | 0.038464 | 0.036224 | 3.337216 | 3.068928 | NA | NA |
| B2 action/long | 2160 | 0.057152 | 0.057088 | 3.763200 | 3.628032 | NA | NA |
| B14 action/short | 288 | 0.071488 | 0.070144 | 3.636224 | 3.481600 | 3.022848 | 3.046400 |
| B14 action/middle | 1152 | 0.185984 | 0.184064 | 3.896320 | 3.939328 | 3.098624 | 3.150848 |
| B14 action/long | 2160 | 0.328896 | 0.325824 | 6.207488 | 5.308416 | 4.431872 | 3.469312 |

## 时段与剩余工作

阶段秒数（日志与文件时间边界）：`{"process_start_unix": 1792325383.93104, "model_load_and_initialization_seconds": 80.31695985794067, "real_input_reconstruction_and_compact_save_seconds": 106.11714935302734, "performance_and_resource_checks_seconds": 78.94317579269409, "whole_GPU_job_lifecycle_seconds": 301.68426156789064}`。性能段含边界资源核查，不将其算成纯kernel时间。

- 真实B14仍为7个独立Q16/CFG2，候选私有KV没有跨分支共享；video与action的实际K分别记录。
- attention使用准备好的输入；B14 shared/guarded直接读取公共历史，完整block才包含其对应history/index/pack差异，二者边界不能混称。
- 仅1个独立进程和3轮，接近噪声的微小优劣不能作为稳定收益；未完成原2进程长重复要求。
- 原始与B14实际输入的代表性compact fixture已保存；后验先用它们验证单attention，完整block使用原冻结checkpoint，不复制大权重。
- 精度/有效KV待验；未测30层完整verifier、接受翻转、scheduler、动作或机器人E2E。四项false保持。

[全部性能](PERFORMANCE-ONLY.csv) · [逐轮](PERFORMANCE-ONLY-ROUNDS.csv) · [输入形状](PERFORMANCE-ONLY-INPUTS.csv) · [机器摘要及fixture索引](PERFORMANCE-ONLY-SUMMARY.json)

原始证据：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-kv-matrix-local-v1-20261018-r6`。

CPU聚合问题：无。
