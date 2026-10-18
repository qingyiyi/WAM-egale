# 原始与verifier输入：性能、精度和完整输入输出验证

更新：2026-10-18T20:40:34+08:00。数据采集完整，standalone完整30层出现真实数值失败，按原记录保留。

结论：现有standalone SDPA在54个真实单attention/block点满足原.02绝对加.02相对容差，但多数并非逐位一致；全30层同时替换self/cross后，六例均不逐位一致，后段三例连该容差也未通过。shared-prefix与guarded路径在六个完整B14 query的原始/CFG输出及有效KV、输入字节、RNG检查上逐位通过。

这支持把Torch/shared-prefix/guarded继续用作这组真实verifier的参考；不能把standalone现有有限单block证书提升为严格同算法完整Transformer证书。原tile源码、原报告与失败未改。

## 覆盖与计时边界

- 新性能：54固定点，三KV×B2 video/B2 action/B14 action×first/last×层0/8/29；684性能单元、18360 Event及等量同步wall样本，单进程3轮。精度在用户允许共享GPU后独立完成，不混入性能计时。
- B2 action与B14 action实际K288/1152/2160，B2 video实际K272/1136/2144。H24/D128/BF16；B14仍为7个独立Q16、两CFG历史组、各自private KV。
- 54点共414次输出比较和54次有效KV/metadata检查；六个完整30层B14 query×四引擎共24次完整模型执行；另对性能阶段真实保存的九输入执行27次直接算子输入/输出/布局/device/RNG检查。
- 性能阶段独占窗口内未见其他可见GPU进程，但NVML利用率空context仍非零，原因未确认；延迟继续标探索性。三轮/单进程也不足证明约1%的微小差异稳定。

性能全覆盖也不等于稳定对齐：standalone self/cross Graph-core相对Torch的54点比例中位数分别.9812/.9817；self无慢点，cross一例慢约.52%。完整block有14/54慢点，比例范围.2465–3.0713，中位数.9665；保留极端波动，尚未定位原因。guarded相对强shared完整block中位数.9960、7/18更慢，也不能由最好长KV点宣称普遍新增收益。详见PERFORMANCE-COVERAGE-CAUTIONS.json。

## 单attention与真实block精度

| 部分 | batch | 实现 | .02通过/比较数 | 逐字节相同 | max_abs最大值 | rel_L2最大值 |
|---|---|---|---|---|---|---|
| self | 2 | standalone | 36/36 | 1/36 | 0.0078125 | 5.7969362e-05 |
| self | 2 | torch | 36/36 | 36/36 | 0 | 0 |
| self | 14 | G1 | 18/18 | 18/18 | 0 | 0 |
| self | 14 | guarded | 18/18 | 18/18 | 0 | 0 |
| self | 14 | shared | 18/18 | 18/18 | 0 | 0 |
| self | 14 | standalone | 18/18 | 5/18 | 0.0078125 | 2.0863972e-05 |
| self | 14 | torch | 18/18 | 18/18 | 0 | 0 |
| cross | 2 | standalone | 36/36 | 1/36 | 0.00012207031 | 2.1664284e-05 |
| cross | 2 | torch | 36/36 | 36/36 | 0 | 0 |
| cross | 14 | standalone | 18/18 | 4/18 | 0.00024414062 | 2.2826667e-05 |
| cross | 14 | torch | 18/18 | 18/18 | 0 | 0 |
| block | 2 | standalone | 36/36 | 2/36 | 0.25 | 0.00096886919 |
| block | 2 | torch | 36/36 | 36/36 | 0 | 0 |
| block | 14 | guarded | 18/18 | 18/18 | 0 | 0 |
| block | 14 | shared | 18/18 | 18/18 | 0 | 0 |
| block | 14 | standalone | 18/18 | 3/18 | 0.03125 | 9.842492e-05 |
| block | 14 | torch | 18/18 | 18/18 | 0 | 0 |

原容差逐元素为`abs(actual-ref)<=0.02+0.02*abs(ref)`。不是统一2%误差，max_abs大于.02仍可能通过相对项；完整block B2最坏max_abs=.25，不隐藏。

## 完整30层：实际有效action通道的CFG方向

| 实现 | .02通过/6 | 逐字节一致/6 | max_abs | rel_L2最大值 | 输入/KV/RNG严格门通过/6 |
|---|---|---|---|---|---|
| torch | 6/6 | 6/6 | 0 | 0 | 6/6 |
| standalone | 3/6 | 0/6 | 0.1875 | 0.035122622 | 0/6 |
| shared | 6/6 | 6/6 | 0 | 0 | 6/6 |
| guarded | 6/6 | 6/6 | 0 | 0 | 6/6 |

standalone六例逐项（CFG有效action）：

| KV | query | max_abs | rel_L2 | .02容差 |
|---|---|---|---|---|
| short | 0 | 0.015625 | 0.0031921277 | True |
| short | 18 | 0.1875 | 0.035122622 | False |
| middle | 0 | 0.015625 | 0.0026898023 | True |
| middle | 20 | 0.050109863 | 0.0180957 | False |
| long | 0 | 0.015625 | 0.0031628676 | True |
| long | 20 | 0.1015625 | 0.026375657 | False |

此处standalone同时替换self与cross；尚未做self-only/cross-only消融，不能把完整输出差异唯一归因到某一部分或某个浮点指令。观察到了累计输出偏差，但未识别完整数值根因。

## 保存输入的直接I/O检查

九组固定layer0/first真实性能输入，共27实现比较；实际通过输入字节、shape/stride/dtype/device/data_ptr不变、输出shape/dtype/device及RNG检查。无效torch.empty KV槽未比较。原输入370763331B。这里只覆盖九个保存代表输入，不伪称54输入全部逐元素保存。

## 可写与不可外推

- 可写：固定真实范围内的表中性能与误差；shared/guarded六个完整B14 query严格输入输出/有效状态通过；standalone完整query数值失败。
- 不可写：任意输入严格等价、全部54形状稳定加速、完整30层B2/video已验证、接受决策完全不变、完整scheduler/chunk/decoded action/机器人闭环通过。
- 六query来自既有真实A_off verifier输入，不是当前TF的新完整机器人质量验证；没有SR/trace/E2E新结论。四项false与原NO_GO保留。
- 原长重复双进程方案按用户新窗口收尾31点PARTIAL，未补造第二进程；新性能协议单独登记，不据最好轮选结果。

## 完整数据与源码

[性能与阶段耗时](PERFORMANCE-ONLY-REPORT.md) · [所有684性能单元](PERFORMANCE-ONLY.csv) · [414精度单元](PRECISION.csv) · [完整30层](FULL30-IO.csv) · [保存输入I/O](SAVED-INPUT-IO.csv) · [机器摘要](SUMMARY-COMBINED.json)

原始GPU结果：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-kv-validation-shared-v1-20261018-r8/process1/RESULT.json`，`/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-saved-input-io-shared-v1-20261018-r9/RESULT.json`。完整模型输出张量全部保存在前一目录；误差统计由真实GPU记录，本CPU聚合不冒充独立重算未保存的所有逐元素输出。

执行入口：`投机采样/experiments/operator_kv_matrix_v1/`；旧tile实现与其HW只读。新fixture压缩仅去重已CPU逐字节验证的七候选重复组，未对数值做量化或更改。

实现名：standalone为原tile `_parity.sdpa`（完整block/30层同时替self与cross）；shared为既有`_shared_torch_v1.shared`（仅替self、cross仍Torch）；G1为candidate grouped(...,1)，只测self kernel；guarded为候选几何guard下group1/2（仅替self）。G1没有本次独立完整block/30层证书。

另有[24份完整输出CPU独立核对](CPU-SAVED-OUTPUT-AUDIT.json)：原始字节一致标志、max_abs、.02容差标志与GPU记录一致；rel_L2仅允许CPU/GPU归约舍入的2e-6相对审计误差。未在CPU执行WAM或初始化CUDA。

累计登记生命周期1649.274/1800秒，未用150.726秒；CUDA active UNKNOWN。不自动追加融合、训练或性能实验。CPU聚合问题：[]。
