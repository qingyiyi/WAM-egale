# 原算子与Torch对照：完整既有证据复核

复核完成：2026-10-18T18:24:38+08:00。状态：`COMPLETE_EXISTING_DATA_AUDIT`。

结论：已验收的独立SDPA/单block实现，在六份真实输入与原容差、原中位数性能门范围内通过；不能写成逐位等价、任意形状稳定加速或整模型/机器人任务已验证。

本轮为CPU离线复核，没有导入Torch、创建CUDA上下文、运行GPU测试或计算新hash；没有改原tile/HW、原FAILED和质量权限。

19:29新增独立补充：[既有verifier静态基线核对](EXISTING-VERIFIER-REPORT.md)。它列出B2/B14三档KV的现有30层单query Graph数据，以及guarded六query有限逐位结果；不与本文standalone实现合并发证。新的真实算子/block矩阵另见[登记](../operator-kv-matrix-20261018/PLAN.md)。

## 身份与范围

- Torch 2.9.0+cu126，Torch commit `0fabc3ba44823f257e70ce397d989c8de5e362c1`；FlashAttention commit `979702c87a8713a8e0a5e9fee122b90d2ef13be5`。
- GPU：NVIDIA A100-SXM4-40GB，108 SM；自有扩展nvcc12.4，Torch wheel CUDA12.6，不声称二者SASS相同。
- BF16、B2/H24/D128、连续BSHD、无mask/noncausal/无dropout/推理；cross K512；fixture为block0、update_cache=0。
- 现存180项实现/头文件/测试/正式包.so与10/12验收归档对应成员逐字节一致。退役build副本不作active binary。细目见[SOURCE-COMPARISON.json](SOURCE-COMPARISON.json)。
- 六份真实输入，两独立进程，各7轮；正确性在fixture/阶段上检查，不是576条独立机器人轨迹。

## 数值数据

所有记录的576项原检查通过。拆分为168项fixture oracle、24项独立attention输出、144项block阶段、240项API/stream/scale契约。原保存的检查标志参与CPU聚合；未重新运行tensor比较。

`allclose`：逐元素满足 `abs(actual-ref) <= 0.02 + 0.02*abs(ref)`。这不是统一2%相对误差，也不要求最大绝对误差低于0.02。`rel_l2 = ||actual-ref||₂ / ||ref||₂` 是整体相对范数，不是最坏元素相对误差。

| 输入 | self max_abs / rel_L2 | cross max_abs / rel_L2 | block最终 max_abs / rel_L2 | 最终torch.equal |
|---|---:|---:|---:|---|
| F1 短video | 0.00390625 / 1.119417e-05 | 1.52587891e-05 / 2.275351e-06 | 0.00390625 / 2.756267e-05 | False |
| F2 短action | 0.000122070312 / 1.585282e-06 | 9.53674316e-07 / 1.060259e-06 | 0.015625 / 4.343406e-05 | False |
| F3 tile0长video | 0.00390625 / 1.491496e-05 | 7.62939453e-06 / 1.390726e-06 | 0.015625 / 5.331895e-05 | False |
| F4 tile0长action | 1.1920929e-07 / 1.805711e-09 | 9.53674316e-07 / 7.417475e-07 | 0 / 0 | True |
| F5 tile1长video | 0.0009765625 / 3.865543e-06 | 6.10351562e-05 / 8.920335e-06 | 0.0078125 / 3.407996e-05 | False |
| F6 tile1长action | 0.00390625 / 5.842678e-05 | 4.76837158e-07 / 3.694229e-07 | 0.015625 / 9.225096e-05 | False |

两进程上述数值统计完全相同。原Torch replay与fixture oracle的168项torch.equal均true；这说明参考重放复现原fixture，不表示自有算子所有输出逐位一致。自有block最终输出仅2/12项torch.equal为true（同一份F4在两个进程）。记录中的bitwise由torch.equal产生，并非原始字节核验。

最坏中间阶段是F6的ffn_output：max_abs=0.125，rel_L2约4.18265e-4；在原相对+绝对容差下通过。最终block max_abs最大0.015625，rel_L2最大约9.22510e-5。不能只展示最终小误差而省略中间放大。

[全部576检查](CORRECTNESS.csv)包含12阶段、原oracle及每项契约；无原始输出全张量，元素误差P99/ULP/更紧容差结果UNKNOWN。

## 性能数据

门限保持原 `median_of_round_medians(cuda) / median_of_round_medians(torch) <= 1`。每轮attention各50次，block各20次；graph attention每个graph含32次调用，原始值已按32归一化。同步墙钟也保留，但原gate使用CUDA Event中位数，不是墙钟或P99 gate。

| 输入 | self eager比r1/r2 | self graph-core比r1/r2 | cross eager比r1/r2 | cross graph-core比r1/r2 | 完整block比r1/r2 |
|---|---:|---:|---:|---:|---:|
| F1 短video | 0.519231/0.509434 | 0.914081/0.914373 | 0.718750/0.730159 | 0.979143/0.979094 | 0.974797/0.970408 |
| F2 短action | 0.565574/0.564516 | 0.962264/0.962264 | 0.569106/0.564516 | 0.977642/0.977642 | 0.953938/0.969623 |
| F3 tile0长video | 0.826923/0.817308 | 0.986005/0.985573 | 0.723077/0.718750 | 0.979687/0.980233 | 0.982988/0.998927 |
| F4 tile0长action | 0.660305/0.797753 | 0.991850/0.990964 | 0.576000/0.580645 | 0.973631/0.973631 | 0.987467/0.977871 |
| F5 tile1长video | 0.783784/0.817308 | 0.985566/0.986960 | 0.633540/0.723077 | 0.975723/0.980233 | 0.998940/0.998125 |
| F6 tile1长action | 0.806818/0.806818 | 0.995745/0.996353 | 0.580645/0.580645 | 0.975610/0.973631 | 0.977581/0.985646 |

全部48项attention门与12项block门通过。eager延迟降低17.31%–49.06%，graph-core降低0.36%–8.59%，完整block降低0.1060%–4.6062%；这三个边界分别报告，不能把eager收益全当纯kernel收益。

| 输入 | r1 Torch / 自有完整block(ms) | r2 Torch / 自有完整block(ms) | 自有不慢于Torch的轮数 r1/r2 |
|---|---:|---:|---:|
| F1 短video | 2.275328 / 2.217984 | 1.833984 / 1.779712 | 6/7；7/7 |
| F2 短action | 1.722880 / 1.643520 | 1.685504 / 1.634304 | 7/7；7/7 |
| F3 tile0长video | 3.160064 / 3.106304 | 1.909248 / 1.907200 | 4/7；6/7 |
| F4 tile0长action | 1.756672 / 1.734656 | 1.712128 / 1.674240 | 5/7；6/7 |
| F5 tile1长video | 1.931264 / 1.929216 | 1.911808 / 1.908224 | 5/7；5/7 |
| F6 tile1长action | 1.735680 / 1.696768 | 1.712128 / 1.687552 | 7/7；7/7 |

全部420个“输入×进程×类型×轮”的比较有16个轮中位数自有更慢；其中完整block 12/84轮更慢。原gate通过并不意味着每轮或每次调用都更快。未删除任何慢轮或离群值。

例如F3长video完整block，r1 Torch/自有为3.160064/3.106304ms，r2为1.909248/1.907200ms；跨进程绝对值波动明显，r2收益仅约0.107%。原稀疏资源gate通过、无计时中profiler/管理API，但物理宿主完全独占未证明。两次进程不足以给出普适稳定加速的统计保证。

[全部60单元](PERFORMANCE.csv)、[全部420逐轮值](ROUND-METRICS.csv)、[240组原始样本分位数](SAMPLE-QUANTILES.csv)。分位数采用线性插值type7，基于同一单元/引擎的7轮样本，作为描述性分布，不改原gate，不把重复样本当独立实验。

## 覆盖缺口与建议

[完整覆盖矩阵](COVERAGE.csv)。当前证据未覆盖中KV1152/更多合法尾块、B14共享prefix与私有KV、update_cache其他分支、30层、CFG方向、接受翻转、scheduler/RNG/完整chunk、最终动作与LIBERO闭环、FSDP整模型。后续G1/G2/guard/cache是独立研究实现，不继承本证书。

推荐补测顺序：先在独立namespace登记真实短中长输入与合法边界；再测真实完整block/30层累计及全部有效KV；之后检查接受分数/路径/scheduler/RNG/最终动作。独占正式性能与闭环质量另按原门放行，不打断当前已登记GPU任务，不自动追加或放宽门限。

原生接口不支持的mask/dropout/FP32/梯度等应按契约拒绝，不能把“拒绝输入通过”写成支持这些输入的数值等价。更紧容差可以前瞻登记，但不能重写旧原门。

## 原始证据与失败保留

- [r1原raw](/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/release-r1.json)
- [r2原raw](/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/release-r2.json)
- [原完整审计](/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/actual-release-report-audit.json)
- [原验收摘要](/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/final-acceptance-summary.json)
- [资源边界](/rjs/cxz/论文探索/wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/final-resource-and-boundary-audit.json)
- [原报告](/rjs/cxz/论文探索/.pipeline/research/tile-sdpa/PYTORCH29_FINAL_20261012.md)
- [上一版本失败](/rjs/cxz/论文探索/.pipeline/research/tile-sdpa/PYTORCH29_RELEASE_FIRST_20261012.md)：长video full-block +0.4013%原FAILED保留。不同版本不混进当前60门，也不能删除。

本次导出只引用原raw，不复制大JSON或模型tensor。新数值实验为NOT_RUN；四项质量/正式性能权限及全局默认均不变。

## CPU审计完整性

问题清单：[]。汇总见[SUMMARY.json](SUMMARY.json)。
