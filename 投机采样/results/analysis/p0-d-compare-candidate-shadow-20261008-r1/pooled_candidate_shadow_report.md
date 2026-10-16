# Pooled candidate-state shadow replay（action / K=2）

- 生成时间：`2026-10-08T04:19:41.032720Z`（2026-10-08，Asia/Shanghai 采集日）。
- 覆盖：`75` 个完整 rollout chunks、`1800` 个唯一 paired anchors。
- 比较对象：同一 root/candidate state 上的 draft `d_hat` 与真实 target `d/d_cfg`；不使用 ZOH 作为比较基准。
- 采集范围：action branch only，K=2，真实 target verifier 与 accept/rollback prefix 规则的 shadow replay；没有发送新的 policy action。
- acceptance threshold：`0.007839680526`，语义为封存 K=2 successor-L2 q90 calibration，不是 ZOH gate。

## 完整性门禁

- 结果：**PASS**。
- worker：`2`；task shard 不重叠：`True`；variant pairing：`True`。
- unique anchor keys：`1800`；每 chunk anchors：`24`；chunk sequence complete：`True`。
- finite numeric values checked：`147770`；EGL + parallel + resume provenance：`True`。

## Worker/task coverage

| task | chunks | anchors |
|---:|---:|---:|
| 8 | 50 | 1200 |
| 9 | 25 | 600 |

## Pooled variant summary

| variant | anchors | first successor-L2 | second successor-L2 | first accept | both accept | accepted prefix |
|---|---:|---:|---:|---:|---:|---:|
| phase4_mlp | 1800 | 0.0042579643 [0.0038802735, 0.0046673512] | 0.0050727823 [0.0046954108, 0.005493677] | 92.33% | 89.61% | 1.81944 |
| phase4_transformer | 1800 | 0.0043569923 [0.0039716767, 0.0047837223] | 0.0052566107 [0.0048810756, 0.0056761131] | 91.28% | 88.50% | 1.79778 |
| phaseA_mlp | 1800 | 0.0042299139 [0.0038520971, 0.0046393345] | 0.0049485502 [0.0045711961, 0.0053640639] | 92.44% | 90.33% | 1.82778 |
| phaseA_transformer | 1800 | 0.0041681199 [0.0037954969, 0.0045736236] | 0.0048264574 [0.0044652946, 0.0052509846] | 92.44% | 90.67% | 1.83111 |

## Task-stratified summary

| group | variant | anchors | first successor-L2 | second successor-L2 | first accept | both accept | accepted prefix |
|---|---|---:|---:|---:|---:|---:|---:|
| 8 | phase4_mlp | 1200 | 0.0040482706 | 0.0048187643 | 93.25% | 90.25% | 1.835 |
| 8 | phase4_transformer | 1200 | 0.0041532287 | 0.0049998079 | 91.75% | 89.25% | 1.81 |
| 8 | phaseA_mlp | 1200 | 0.004015058 | 0.0046801569 | 93.25% | 90.83% | 1.84083 |
| 8 | phaseA_transformer | 1200 | 0.0039646621 | 0.0045637194 | 93.08% | 91.17% | 1.8425 |
| 9 | phase4_mlp | 600 | 0.0046773516 | 0.0055808183 | 90.50% | 88.33% | 1.78833 |
| 9 | phase4_transformer | 600 | 0.0047645195 | 0.0057702163 | 90.33% | 87.00% | 1.77333 |
| 9 | phaseA_mlp | 600 | 0.0046596257 | 0.0054853368 | 90.83% | 89.33% | 1.80167 |
| 9 | phaseA_transformer | 600 | 0.0045750354 | 0.0053519333 | 91.17% | 89.67% | 1.80833 |

## Denoising-phase summary

| group | variant | anchors | first successor-L2 | second successor-L2 | first accept | both accept | accepted prefix |
|---|---|---:|---:|---:|---:|---:|---:|
| early | phase4_mlp | 600 | 0.0089971331 | 0.0097444557 | 77.17% | 69.83% | 1.47 |
| early | phase4_transformer | 600 | 0.0092809191 | 0.010304904 | 74.00% | 66.50% | 1.405 |
| early | phaseA_mlp | 600 | 0.0089504388 | 0.0094947962 | 77.33% | 71.83% | 1.49167 |
| early | phaseA_transformer | 600 | 0.0087616271 | 0.0091260805 | 77.33% | 72.83% | 1.50167 |
| middle | phase4_mlp | 600 | 0.0018851129 | 0.002395808 | 99.83% | 99.83% | 1.99667 |
| middle | phase4_transformer | 600 | 0.0018936984 | 0.0023792359 | 99.83% | 99.83% | 1.99667 |
| middle | phaseA_mlp | 600 | 0.0018502315 | 0.0022755105 | 100.00% | 100.00% | 2 |
| middle | phaseA_transformer | 600 | 0.001856697 | 0.0022950427 | 100.00% | 100.00% | 2 |
| late | phase4_mlp | 600 | 0.0018916467 | 0.0030780833 | 100.00% | 99.17% | 1.99167 |
| late | phase4_transformer | 600 | 0.0018963593 | 0.0030856925 | 100.00% | 99.17% | 1.99167 |
| late | phaseA_mlp | 600 | 0.0018890715 | 0.0030753439 | 100.00% | 99.17% | 1.99167 |
| late | phaseA_transformer | 600 | 0.0018860355 | 0.0030582489 | 100.00% | 99.17% | 1.99167 |

## Paired comparisons

| comparison | anchors | first Δ mean [95% CI] | second Δ mean [95% CI] | prefix Δ mean [95% CI] | both-accept Δ mean [95% CI] |
|---|---:|---:|---:|---:|---:|
| phase4_mlp_vs_phase4_transformer | 1800 | -9.9028021e-05 [-0.00012919713, -7.1146287e-05] | -0.00018382834 [-0.00024100539, -0.00013134244] | 0.0216667 [0.0138889, 0.0294444] | 0.0111111 [0.00555556, 0.0166667] |
| phase4_mlp_vs_phaseA_mlp | 1800 | 2.8050335e-05 [1.2448293e-05, 4.3049259e-05] | 0.00012423214 [9.4535362e-05, 0.00015352849] | -0.00833333 [-0.0144444, -0.00277778] | -0.00722222 [-0.0116667, -0.00333333] |
| phase4_mlp_vs_phaseA_transformer | 1800 | 8.9844384e-05 [6.8380302e-05, 0.00011122293] | 0.00024632494 [0.00020306407, 0.00028906197] | -0.0116667 [-0.0188889, -0.005] | -0.0105556 [-0.0161111, -0.00555556] |
| phase4_transformer_vs_phaseA_mlp | 1800 | 0.00012707836 [0.00010018213, 0.00015439828] | 0.00030806048 [0.00026458246, 0.00035512119] | -0.03 [-0.0388889, -0.0222222] | -0.0183333 [-0.025, -0.0127778] |
| phase4_transformer_vs_phaseA_transformer | 1800 | 0.0001888724 [0.00016714924, 0.00021258905] | 0.00043015328 [0.00037824167, 0.00048195781] | -0.0333333 [-0.0422222, -0.025] | -0.0216667 [-0.0288889, -0.015] |
| phaseA_mlp_vs_phaseA_transformer | 1800 | 6.1794048e-05 [4.4228565e-05, 8.0035383e-05] | 0.0001220928 [8.1866765e-05, 0.00015940108] | -0.00333333 [-0.00833333, 0.00166667] | -0.00333333 [-0.00722222, 0] |

## 解释边界

- first/second successor-L2 是真实 target 输出上的局部 discrepancy，不等于完整环境 endpoint error。
- accepted prefix / acceptance 是在固定 threshold 下的 route comparison，不是 LIBERO task success。
- 本轮没有 policy rollout、没有 task success、没有 E2E speedup；target forward 是 shadow replay 的测量成本。
- cache/reset/VAE 的工程开销仍需第三层真实 formal K=2 rollout 单独测量。

## Provenance

- trace：`投机采样/traces/p0-libero-long-k1-alternate-10tasks-init0-20260928`。
- dataset branch metadata sha256：`9337057a7c555a3f47ae416c6dfe1fa5d2f963634dc219f39aa4b31bc9914055`。
- worker launch lines：`2` 条，均记录 `workers=2`, `resume=true`, `EGL=egl/egl`, `device=0`, `CUDA_VISIBLE_DEVICES=0`。
