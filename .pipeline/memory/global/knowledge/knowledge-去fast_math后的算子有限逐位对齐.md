---
kind: knowledge
name: 去fast_math后的算子有限逐位对齐
level: reference
scope: project:WAM-egale;operator-torch-alignment-v2
status: active
updated: '2026-10-18'
sources:
  - .pipeline/experiments/operator-torch-alignment-20261018/RESULT.md
  - .pipeline/experiments/operator-torch-alignment-20261018/RESULT.json
  - .pipeline/experiments/operator-torch-alignment-20261018/SOURCE-COMPARISON.json
---

# 新独立版本的有限严格对齐

operator_torch_alignment_v2复用固定Torch2.9.0/FlashAttention/CUTLASS，179个vendor和去命名空间后的CUDA源逐字相同，数学编译仅移除--use_fast_math。9真实保存输入、108attention、54block、六完整B14 30层、短中长三完整canonical policy配对输出与声明有效状态/RNG均逐位通过；CPU27保存输出复核相符。支持该编译因素在这些输入上的因果影响，未定位具体FTZ/除法指令。

旧M21 standalone六full30中三容差失败、全部非逐位仍是旧版本真实结果；不改旧FAILED或旧证书。新版本实际以独立install/restore替换30层self/cross共60个op，无Torch SDPA fallback，旧源码与默认不改。server_entry另提供可选入口，本轮未另跑RPC机器人任务。

新二进制性能未复测，不能继承旧fast_math的.9812/.9817比例；lhw GPU context阻塞正式计时。有限真实输入不保证任意输入/编译器/GPU或TF完整闭环，四项项目质量false不变。GPU生命周期615.057/1200秒、两job实际退出、714 full target forwards，无新机器人episode/训练/SHA。全KV只做现场逐位比较，没有复制完整KV用于离线重验。
