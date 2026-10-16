---
kind: decision
name: 算子基线与接口限制
level: reference
scope: tile-and-system
status: active
updated: '2026-10-16T00:06:11+08:00'
supersedes: []
sources:
- .pipeline/research/tile-sdpa/PYTORCH29_FINAL_20261012.md
- .pipeline/research/speculative-sampling/MEGATRANSFORMER-TILE-INTEGRATION-20261014.md
---

# 算子基线与接口限制

原tile是在A100/SM80、PyTorch2.9、BF16、D128、contiguous BSHD、noncausal/无mask/dropout/inference范围，独立编译上游FA/CUTLASS核心，无builtin SDPA fallback。唯一owned CUDA入口不等于全部vendor内核为自研。

两进程×六真实fixture：576正确性、48attention、12full-block原门全部通过，原atol/rtol=.02。eager/core/full-block延迟分别降17.31–49.06%/.36–8.59%/.1060–4.6062%；只限固定输入。长video收益很小，历史+.4013%失败保留。

旧/新自有SASS相同不代表与Torch wheel机器码相同；host差异未被证明是全部历史回退的唯一原因。

tile主要B2/H24/Q128或Q16；当前投机树7状态×CFG2=B14，共享历史KV。不能直接套用B2 dense接口或split策略，更不能复制14份历史破坏共享。先profile真实B14，再决定最小系统原型；不将tile和投机速度比直接相乘。

原tile代码入口与完整原证据均由原报告导航，本记录只读引用，不改tile方向HW。
