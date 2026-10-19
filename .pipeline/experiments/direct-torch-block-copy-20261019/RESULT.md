# 直接复制Torch完整block：有限复现已完成

更新：2026-10-19T09:55:09.086497+08:00

## 结果

固定ca5f37e的`custom_sdpa`、完整`WanAttention`（含cache助手）及`WanTransformerBlock`三个定义逐字/AST相同。复制版源码独立可读，不依赖运行时git提取；Torch、diffusers、einops作为原基元依赖保留。`copied_model_blocks`提供可撤销整模型替换入口；本次实际GPU验证范围仍只有六个action block0 fixture。

同一module、Parameter对象及地址、cache对象与地址，在计时外切换原版/复制类。六个真实保存输入，B2/B14×KV288/1152/2160，uncached原路径。上述KV是attention实际K总长度，公共历史272/1136/2144加当前私有16；每实例Q仍16，未跨候选互相注意。18次数值检查对原版及历史GPU gold逐位相同（max_abs=0），全部有效KV/元数据、输入与当前CPU/CUDA RNG通过；计时后同样通过。共294次完整block，worker自然退出，无其他GPU进程，进程生命周期14.600秒。本阶段此前118次，合计412/5000；第一次移植supervisor的相对import失败在GPU前发生，原始记录保留，修正不涉及计算。

| 输入 | 原版中位ms | 复制版中位ms | 复制/原版中位耗时 |
|---|---:|---:|---:|
| short-B14-action-layer0-first | 3.8728 | 3.7478 | 0.9677 |
| middle-B14-action-layer0-first | 4.5998 | 4.6152 | 1.0033 |
| long-B14-action-layer0-first | 5.1062 | 5.0652 | 0.9920 |
| short-B2-action-layer0-first | 3.0781 | 2.9896 | 0.9712 |
| middle-B2-action-layer0-first | 2.5631 | 2.5636 | 1.0002 |
| long-B2-action-layer0-first | 2.8052 | 2.8022 | 0.9989 |

三轮ABBA/BAAB/ABBA，每段三次重复，每臂五次warmup；每轮原始Event/wall值全部保留。表中为每臂三个轮次中位数再取中位数，比例是两者之比；与三轮配对比例的几何均值分开。

六点中位耗时比最大偏离3.23%；全部配对比例几何均值0.980489。单轮比例范围0.858717–1.097281，说明eager计时波动仍明显。复制版没有新的数学优化，不能把更快的中位数/0.9805称为加速创新。只支持该有限范围的数值复现与接近原版耗时，不签稳定全形状性能证书。

## 边界与后续

旧B0的66次GPU检查及旧A/A2.512635%>2.5%的失败原样保留；新的直接复制范围不回写旧门。此前多引擎自动矩阵/fixed3 AA均未再启动，旧watch已按范围修正退出。GPU现已释放，本子项不自动增加重复、30层或机器人任务。

原fixture没有保存生产CUDA stride/alias/RNG，不能恢复或声称全生产条件逐位重验。视频、完整30层、完整policy、所有cache写路径未由本轮新增证明；整模型安装入口已提供，实际模型替换运行另登记。Torch库基元不等于独立CUDA/persistent调度。四项false、旧NO_GO/FAILED/retention、原tile和保护文件不改。

默认下一步以此直接复制版作为后续完整模型替换与性能测量的基线，先验证整链，再研究调度。停止将复制本身、自研基元、调度优化三者混在同一个交付门。

## 原始证据

- [逐定义源码与真实权重绑定核对](SOURCE-AND-BINDING-RESULT.json)
- [GPU完整数值/状态与原始计时](attempt-02/matrix/RESULT.json)
- [实际清理收据](attempt-02/RECEIPT.json)
- [全部原始计时CSV](RAW-TIMING-SAMPLES.csv)
- [GPU前import失败](PREPARATION-FAILURE-01.json)
- [本项登记及原预算](PLAN.md)
