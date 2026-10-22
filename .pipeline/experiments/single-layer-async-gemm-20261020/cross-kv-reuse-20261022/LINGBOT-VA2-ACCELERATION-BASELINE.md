# LingBot-VA2 的加速基线与 Cross K/V 预计算

核验：2026-10-22。来源为 [Native Video-Action Pretraining for Generalizable Robot Control，arXiv:2607.08639v1](https://arxiv.org/html/2607.08639v1)，2026-07-09版本，§2.4.2及Table 3。以下是原文报告，未在本项目复现。

|累计方案|每chunk推理延迟 ms|异步控制频率 Hz|
|---|---:|---:|
|BF16 PyTorch异步rollout基线|927|35|
|再加一致性蒸馏|466|69|
|再加低精度编译执行|369|87|
|再加长历史attention优化|272|118|
|再加运行时开销优化|142|225|

论文的约6.5倍来自927/142。起点是PyTorch，最终使用FP8 TensorRT及FlashInfer；包括蒸馏改变去噪步数，因此不能解释为固定BF16模型的单个自有算子相对Torch获得6.5倍。225Hz按每chunk的32个控制步折算，不是每秒225次完整推理。

原文明确列出Cross-attention K/V预计算，但没有单独给出该项加速比。推理硬件及精确Torch/TensorRT版本在上述段落未找到，记UNKNOWN；不由本项目A100或论文训练配置推断。

## 本地依赖与下一步建议

本项目该层Cross K/V只依赖context、该层权重及数值配置，见 `算子研究/wam_ops/optimized/csrc/whole_layer.cu` 的 `context_linear` 和 `whole_layer`；不依赖action latent或sigma。Cross K有整行RMS，V无RMS；二者均无RoPE。可在条件固定期间缓存最终K与V，Q和attention仍每次计算。

须区分本轮已获准实施的“同一次调用内按相同context去重”和建议的“跨调用预计算”。当前新路径仍在每次调用计算两组K/V及K RMS；尚未实现持久化缓存。每层权重不同、正负CFG条件不同，不能共用一份；条件、权重、dtype或device变化应重新建立缓存。不能复用self-attention的候选私有KV。

现有 `optimized/prepared.py` 已有原形状K/V投影缓存，本轮计时列出warm cached强baseline；它尚未缓存K RMS。后续接入时建议缓存两组最终K/V，并同时给原Torch/guarded对照相同能力，冷构建与warm调用分列。本轮没有把该建议解释为新的数值放行、整模型集成或机器人实验授权。

这是共同工程优化，不能单独称为新的MegaTransformer机制。新增候选调度的主张仍需在相同缓存条件下成立。
