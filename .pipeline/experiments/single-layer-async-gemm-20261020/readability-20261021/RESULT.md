# 当前算子入口精简结果

2026-10-21。沿用M39，直接修改optimized；baseline 7个源码/文档文件逐项未变，投机及原tile未改。

## 已完成

- Python直接绑定原参数，摆脱旧NativeBlock构造及其模式分支；参数引用/eps在外部核验一致。
- 删除无调用的action_ffn_linear和self_kv；未删除数学路径。
- 普通forward只组装output；debug=True才构造中间量字典、查询kernel资源。旧直接extension.forward调用的默认debug兼容保留。
- 计算workspace仍必须分配；tile边界、KV可见性、BF16分段、同步及CUDA调用失败传播保留。设备端数学/阶段/任务划分不改。

## 有限验证

- 九真实签名（B2 action/B14 action/B2 video×三KV）全部通过，原Torch输出、23共有中间量、18有效KV转移、重复/输入/RNG核验通过；实际81次完整block调用。
- 新增九组普通返回与debug返回/状态逐位一致。仅有限fixture，不外推任意输入、完整30层或机器人质量。

## 同轮完整单层Graph计时

|进程|KV|精简前ms|精简后ms|耗时变化|
|---|---|---|---|---|
|1|288|2.506752|2.506752|+0.000%|
|1|1152|2.725888|2.724352|-0.056%|
|1|2160|2.975232|2.978304|+0.103%|
|2|288|2.505728|2.506240|+0.020%|
|2|1152|2.724864|2.721280|-0.132%|
|2|2160|2.974720|2.973696|-0.034%|

两独立进程、同GPU0、交错顺序、相同1410MHz实测时钟；不修改时钟。变化小且无一致退化，保留简洁入口。不是相对Torch新倍数，不称本次有设备端加速；GPU1并行RoboTwin，CPU绑定分离。

## 资源、身份与下一步

GPU生命周期55.563769秒，CPU构建133.949637秒；M39累计2538.679523/36000秒。0失败/timeout，owned均退出。
源码`043c77597cb16f600ac42f10ae63e7fb4dc055becfb70e2394f328fe76e4b69a`；二进制`2a8a250ee31daca7640c0ba77fd167c1f72957fe4df577abc30f4c4563d0a50a`。本机复制同一远端验证DSO，未运行本机GPU。

源码导航与带读prompt见optimized/WHOLE_LAYER.md和READ_CODE_PROMPT.md。下一步仍为Self-Q嵌入差距→FFN原语→ready-DAG，本轮未新增调度优化，原Self-Q退化与所有负结果保留。

原始数据：SIMPLE-GATE-RESULT.json、TIMING-1/2.json、WHOLE-LAYER-COMPARISON.csv、SOURCE-MANIFEST.json、CHANGE.patch、STATUS.json、SELECTION.json、RESOURCE-AND-STOP-REPORT.json。
