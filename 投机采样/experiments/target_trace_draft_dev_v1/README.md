# Target-trace draft：40条完整LIBERO DEV

此入口只用于本机算法质量比较。Long10 task0–9，各固定init24，四臂为原target、现有A_off、相同新数据预训练的MLP96、TF64，共40条完整任务。每arm只加载一次原目标模型，每次只运行一个真实EGL客户端；全部保留原800 horizon和失败轨迹。不运行smoke，不调用旧ramp/zero数值函数，也不产生独立SHA作业。

运行前必须等 `p0-reset-online-v1-20261015-r1/RESULT.json` 报 `PASS_COMPLETE_RESET_BANK_EQUIVALENCE`，以及 `SAVED-TENSOR-PAIR-AUDIT.json` 报 `PASS_ALL_SAVED_EPISODE_TENSORS`（4对、0错误），8条完整reset等价任务、所有保存动作/视频latent/观测的CPU复核及自有模型清理全部完成；新训练目录必须六组终态完成。缺少任一前置条件，prepare和run均拒绝启动。四臂统一显式使用已验收的最终T5 embedding bank；默认部署和A_off preset不改。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/target_trace_draft_dev_v1/run.py --prepare \
  --job /rjs/cxz/论文探索/投机采样/results/runtime/p0-target-trace-draft-dev-v1-20261015-r1

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/target_trace_draft_dev_v1/run.py \
  --job /rjs/cxz/论文探索/投机采样/results/runtime/p0-target-trace-draft-dev-v1-20261015-r1
```

prepare只读JSON、小型checkpoint和源码作为集成provenance，不加载目标权重或初始化CUDA。正式run才创建NVML并按显存条件启动GPU。一个新目录只运行一次；失败保留，不自动续跑覆盖。`STOP_REQUESTED`可请求在资源监控周期内结束并清理自有PID。

## 对齐与边界

- 原目标模型加载后、任何draft构造/完整CAL执行之前，保存其CPU/CUDA RNG；setup结束恢复，READY保存指纹，每episode再次恢复。同task各arm的初始模型RNG、模拟器/控制器/观测/RNG必须一致。
- 新draft使用本轮固定训练目录的CAL中位seed。启动时真实20,237条CAL查询逐条与相同GPU上的原 `decode_batch` 比较，必须逐位一致；CPU/GPU差别另记为诊断。不运行合成查询，也不将该数值门当机器人质量保证。
- 提前设置 `action_per_frame`，保持七通道、K2/B2 bonus、`prefix_then_cost`、原阈值和共享KV/O1–O3。两个PREFIX开关均明确开启。T保持原target推理。
- 每条执行原完整reset，检查bank向量、空KV/VAE与RNG。保存全部commands、初终模拟器/控制器状态、逐请求target计数及全部接受树。原源码、当前reset作业、AutoDL和tile方向不改。
- 运行总时限8小时，单episode40分钟，初始空闲显存至少22GiB，运行时至少0.5GiB；输出最多4GiB，磁盘至少保留5GiB。共享GPU允许，所有墙钟数据只作诊断，正式性能声明恒为false。

## 预先固定的工程筛选

TF的完整DEV成功数须不低于旧A_off和新MLP；以原T为trace参照，在双方成功配对中，TF/T环境步数比中位数≤1.10、P90≤1.20，且至少有一个双方成功配对。全部通过才建议TF进入扩大质量评测；否则完整报告失败项，不强行选结构、不调整阈值或seed。TF/A_off、TF/新MLP同项比值也完整报告。

10对不能证明成功率下降不超过3个百分点；通过仅代表值得扩大评测。`RESULT.json`包含每任务成败四格、环境步数分布、显式失败800步惩罚分数、target/chunk、平均接受深度，以及诊断墙钟。`REPORT.md`为研究实验汇报，不是论文初稿。

自身模块均通过 `target_trace_draft_dev_v1.*` 导入；原launcher/prefix模块经单独bridge检查模块实际来源，避免旧实验同名 `protocol/adapter/paths` 串入。
