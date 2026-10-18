# 既有verifier证据的补充CPU核对

更新：2026-10-18T19:29:27+08:00。没有新GPU重放；数值一致性来自原GPU记录，本轮仅独立重算原始时间样本。

此报告与原standalone SDPA验收分开。这里使用现有shared-prefix与guarded G2路径、同shape静态prompt缓存，不能给standalone继承逐位证书。

| KV | query | B2未缓存/缓存 ms | B14 shared未缓存/缓存 ms | B14 guarded未缓存/缓存 ms |
|---|---|---|---|---|
| 288 | 0 | 24.0419 / 19.0543 | 67.0801 / 31.2987 | 65.7069 / 30.0335 |
| 288 | 18 | 24.0877 / 19.1828 | 67.0718 / 31.1072 | 65.8350 / 29.9628 |
| 1152 | 0 | 30.7979 / 25.8690 | 71.8504 / 35.6512 | 69.3248 / 33.3551 |
| 1152 | 20 | 30.8966 / 26.0284 | 71.7194 / 35.8226 | 69.3778 / 33.5076 |
| 2160 | 0 | 38.3289 / 33.3636 | 77.7491 / 41.4133 | 69.8602 / 33.8028 |
| 2160 | 20 | 38.2607 / 33.5388 | 77.7441 / 41.4306 | 69.8511 / 33.7963 |

## 数值与资源边界

- 六个真实冻结query、全部30层：guarded路径六query完整输出逐位/有限，180层位点真正调用guarded，记录有效KV、slots、RNG和版本不变。它仍只是有限fixture证据。
- 静态baseline有36次eager准入、36次首次Graph replay、216次计时样本；原记录输出均与同shape原Torch逐位一致。两套实验的自有进程实际退出。
- 表是单个完整30层model query的静态Graph延迟，不含整个候选树、draft/pack/accept、有效提交g、完整policy或机器人E2E。
- B2与B14分别使用原shape投影；缓存普通收益单列，不能将cached B14与uncached B2比较后包装为MegaTransformer协同。
- prompt首填不在热Graph内：B2 payload383778816B，B14 payload2686451712B，均保存61个投影；实际首填开销见SUMMARY，不忽略构建和缓存容量。
- 只覆盖1/7状态、不含3状态；没有任意新输入/真实cache生命周期、完整block逐层误差分布或单query内有效提交成本证据。
- 原未guarded G2的失败不因后续几何guard改写；当前新算子矩阵仍要独立测性能与误差。四项质量/性能权限不变。

完整36单元CSV：[EXISTING-VERIFIER-PERFORMANCE.csv](EXISTING-VERIFIER-PERFORMANCE.csv)；机器摘要：[EXISTING-VERIFIER-SUMMARY.json](EXISTING-VERIFIER-SUMMARY.json)。

原始证据：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-wam-static-prompt-baseline-v2-20261016-r1/BASELINE-RESULT.json`、`/rjs/cxz/论文探索/投机采样/results/runtime/p0-wam-candidate-tile-guarded-v2-20261016-r1/GUARDED-RESULT.json`及各SUPERVISOR-RESULT.json。

CPU复核问题：无。
