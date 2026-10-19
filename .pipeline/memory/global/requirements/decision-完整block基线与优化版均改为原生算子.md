---
kind: decision
name: 完整block基线与优化版均改为原生算子
level: constraint
scope: cycle:wam-research-20261016;operator_full_block
status: active
updated: '2026-10-19'
supersedes:
  - 当前阶段默认只替换attention而把其余block保持为Python Torch调用的交付范围
  - baseline当前源码不得改动的阶段性限制
sources:
  - 用户2026-10-19明确要求整个block都是自有算子，baseline也是，并对齐性能和精度
---

# baseline与optimized均覆盖完整原生block

用户已将当前算子目标扩展为完整block的独立原生算子，baseline和optimized都须覆盖投影、Norm、RoPE、attention、FFN、残差及KV语义，并对齐原Torch精度和性能。完整block计时不能代替完整实现，不再以仅attention CUDA替换作为最终交付。

原Torch模型是只读参考，旧baseline/结果先归档后改造。仅维护baseline/optimized两份主实现；允许在本轮算子目录内修改baseline，不影响投机源码、配置、运行依赖、原tile和HW保护文件。baseline先建立顺序原生调用链，optimized在其上逐项优化；完整block可以由多个CUDA kernel和成熟GEMM原语组成，真实依赖必须公开。

逐位输出、中间量、全部有效KV/metadata、输入/RNG和安装还原分层核对，再比较同输入同边界性能。旧失败、旧筛查与四项false不修改；不能把ATen高层运算的C++转调包装成已独立实现。实际作业另登记有限预算，仍等GPU自然空闲。
