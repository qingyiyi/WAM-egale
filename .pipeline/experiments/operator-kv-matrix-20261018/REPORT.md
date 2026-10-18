# 原始与verifier规模短中长KV算子对照

更新：2026-10-18T20:06:38+08:00。输入/原始样本：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-kv-matrix-local-v1-20261018-r5`。

CPU复核：PARTIAL，见缺口；实际1个进程、31点、221数值单元、366性能单元、112140 Event样本。

计时资源状态：EXPLORATORY：存在未完整结束或GPU利用率计数未满足登记空闲条件的进程；时间仅作探索性结果，不作为已确认无争用的正式性能。

每个KV、输入规模使用真实first/last query与层0/8/29。B14仍为7个独立Q16、CFG2；不是跨候选Q112。video实际K与action K分开列在FIXTURES。

表中是同组12个点（两进程×两query×三层）的比例中位数及完整范围，不是单一模型或任务加速比。大于1表示比Torch慢；B14必须同时看相对shared列。

| 输入 | KV档 | 部分/计时 | 实现 | 实现/Torch中位数 [min,max] | 实现/shared中位数 | .02通过/检查 | 逐字节相同 | 最大绝对误差 |
|---|---|---|---|---|---|---|---|---|
| B2 video | middle | self/eager | standalone | 0.7452 [0.7107,0.7476] | NA | 6/6 | 0/6 | 0.0078125 |
| B2 video | middle | self/graph_core | standalone | 0.9869 [0.9816,0.9914] | NA | 6/6 | 0/6 | 0.0078125 |
| B2 video | middle | cross/graph_core | standalone | 0.9770 [0.8875,0.9794] | NA | 6/6 | 0/6 | 0.00012207031 |
| B2 video | middle | block/eager | standalone | 0.9866 [0.9312,1.0433] | NA | 6/6 | 0/6 | 0.125 |
| B2 video | long | self/eager | standalone | 0.7665 [0.7465,0.7937] | NA | 6/6 | 0/6 | 0.0078125 |
| B2 video | long | self/graph_core | standalone | 0.9909 [0.9841,0.9941] | NA | 6/6 | 0/6 | 0.0078125 |
| B2 video | long | cross/graph_core | standalone | 0.9786 [0.9766,0.9816] | NA | 6/6 | 0/6 | 6.1035156e-05 |
| B2 video | long | block/eager | standalone | 0.9717 [0.8493,1.0060] | NA | 6/6 | 0/6 | 0.25 |
| B2 action | middle | self/eager | standalone | 0.6874 [0.6386,0.7024] | NA | 6/6 | 0/6 | 0.00390625 |
| B2 action | middle | self/graph_core | standalone | 0.9842 [0.9581,1.0933] | NA | 6/6 | 0/6 | 0.00390625 |
| B2 action | middle | cross/graph_core | standalone | 0.9786 [0.9758,0.9875] | NA | 6/6 | 0/6 | 6.1035156e-05 |
| B2 action | middle | block/eager | standalone | 0.9626 [0.6371,1.0153] | NA | 6/6 | 0/6 | 0.0234375 |
| B2 action | long | self/eager | standalone | 0.7512 [0.7119,0.7573] | NA | 6/6 | 0/6 | 0.00390625 |
| B2 action | long | self/graph_core | standalone | 0.9980 [0.9888,1.0141] | NA | 6/6 | 0/6 | 0.00390625 |
| B2 action | long | cross/graph_core | standalone | 0.9777 [0.9756,0.9798] | NA | 6/6 | 1/6 | 1.5258789e-05 |
| B2 action | long | block/eager | standalone | 0.9692 [0.9236,1.0144] | NA | 6/6 | 2/6 | 0.03125 |
| B14 action | middle | self/eager | G1 | 0.8171 [0.8171,0.8171] | 0.7143 | 1/1 | 1/1 | 0 |
| B14 action | middle | self/eager | guarded | 0.8171 [0.8171,0.8171] | 0.7143 | 1/1 | 1/1 | 0 |
| B14 action | middle | self/eager | shared | 1.1440 [1.1440,1.1440] | 1.0000 | 1/1 | 1/1 | 0 |
| B14 action | middle | self/eager | standalone | 0.8872 [0.8872,0.8872] | 0.7755 | 1/1 | 1/1 | 0 |
| B14 action | middle | self/graph_core | G1 | 0.8551 [0.8551,0.8551] | 0.6999 | 1/1 | 1/1 | 0 |
| B14 action | middle | self/graph_core | guarded | 0.8542 [0.8542,0.8542] | 0.6992 | 1/1 | 1/1 | 0 |
| B14 action | middle | self/graph_core | shared | 1.2217 [1.2217,1.2217] | 1.0000 | 1/1 | 1/1 | 0 |
| B14 action | middle | self/graph_core | standalone | 0.9925 [0.9925,0.9925] | 0.8124 | 1/1 | 1/1 | 0 |
| B14 action | middle | cross/graph_core | standalone | 0.9894 [0.9894,0.9894] | NA | 1/1 | 0/1 | 5.9604645e-08 |
| B14 action | middle | block/eager | guarded | 0.7070 [0.7070,0.7070] | 0.9608 | 1/1 | 1/1 | 0 |
| B14 action | middle | block/eager | shared | 0.7359 [0.7359,0.7359] | 1.0000 | 1/1 | 1/1 | 0 |
| B14 action | middle | block/eager | standalone | 1.0066 [1.0066,1.0066] | 1.3679 | 1/1 | 1/1 | 0 |
| B14 action | long | self/eager | G1 | 0.8719 [0.8508,0.8859] | 0.7289 | 6/6 | 6/6 | 0 |
| B14 action | long | self/eager | guarded | 0.5175 [0.5055,0.5219] | 0.4307 | 6/6 | 6/6 | 0 |
| B14 action | long | self/eager | shared | 1.1901 [1.1607,1.2229] | 1.0000 | 6/6 | 6/6 | 0 |
| B14 action | long | self/eager | standalone | 0.9207 [0.9116,0.9263] | 0.7715 | 6/6 | 2/6 | 0.0078125 |
| B14 action | long | self/graph_core | G1 | 0.8928 [0.8678,0.8978] | 0.7203 | 6/6 | 6/6 | 0 |
| B14 action | long | self/graph_core | guarded | 0.5060 [0.4921,0.5072] | 0.4080 | 6/6 | 6/6 | 0 |
| B14 action | long | self/graph_core | shared | 1.2393 [1.2091,1.2429] | 1.0000 | 6/6 | 6/6 | 0 |
| B14 action | long | self/graph_core | standalone | 0.9920 [0.9769,0.9987] | 0.8022 | 6/6 | 2/6 | 0.0078125 |
| B14 action | long | cross/graph_core | standalone | 0.9960 [0.9924,0.9993] | NA | 6/6 | 1/6 | 0.00024414062 |
| B14 action | long | block/eager | guarded | 0.5665 [0.5257,1.0109] | 0.9665 | 6/6 | 6/6 | 0 |
| B14 action | long | block/eager | shared | 0.5795 [0.5457,1.0427] | 1.0000 | 6/6 | 6/6 | 0 |
| B14 action | long | block/eager | standalone | 1.0113 [0.9332,1.3551] | 1.7205 | 6/6 | 1/6 | 0.03125 |

## 证据边界

- 全部点/样本/慢轮保留在CSV与runtime原JSON；分位数用type7线性插值。eager与Graph-core独立，Graph每replay32调用、按32摊分。
- 完整block包括原norm、projection、FFN和所列attention替换；不是30层verifier、scheduler、动作或机器人闭环证书。
- .02为原绝对加相对容差；更紧容差与ULP是诊断统计，不改旧门。容差通过不等于逐位相等。
- captured B14 oracle仅证明冻结query在原Torch路径复现；不证明替换算子后完整模型/接受路径不变。有效KV检查只覆盖这些真实block重放。
- timing_window_qualified综合进程完整终态及GPU利用率准入；process_complete只代表本轮覆盖完成。若NVML非零而可见进程/显存为空，按用户明确窗口先采输入，延迟标探索性，不猜测计数根因。其他版本、设备、任意输入仍未覆盖。
- 四项研究权限保持false，未跑TF正式45chunk、独立TEST或任务E2E；不能据算子结果声称完整投机或机器人加速。

## 完整数据

- [每点输入与实际形状](FIXTURES.csv)、[全部精度](ACCURACY.csv)、[全部性能](PERFORMANCE.csv)
- [逐轮](ROUND-METRICS.csv)、[样本分布](SAMPLE-QUANTILES.csv)、[分组总表](GROUPED.csv)、[机器可读摘要](SUMMARY.json)

## 缺口

- 进程1完整54点覆盖
- 进程1未完整结束
- 进程2尚无RESULT.json
- 双进程launcher尚未完整终态
