# 当前精度版本融合阶段对照

OBSERVED：三方同RTX5090、Torch2.9/cu130、B14/Q16、同原B14 CrossKV预计算、eager无Graph。两独立进程三真实block0输入；自有26中间量＋输出逐位一致，guarded与静态dense原Torch完整输出逐位一致，所有有效KV/输入/RNG未变化。当前正常数学源码未改。插桩版三档输出一致，资源同正常版，扰动+0.22%/+0.11%/+0.17%。

完整层Torch/自有短中长0.8896/1.2810/1.6427倍；短KV自有更慢，不能沿用数值失败旧版本收益。guarded/自有0.7867/0.7868/0.7873倍，即自有仍慢约27%。两进程逐轮值见results与TIMING-VARIABILITY.json；guarded长KV第一进程IQR约5.09%，第二约0.50%，进程中位数相差约2.1%，保留波动，不宣称统计显著。其余大多数IQR约0.18%–1.07%。

阶段归因：整个融合FFN相对原Torch约0.50倍，占自有阶段区间总和46.1%–53.1%；Self输出、Cross Q、Cross输出各约0.68–0.71倍，合计占16.2%–18.7%。Self QKV约1.06倍，目前不是最大退化点。Attention相对dense Torch有收益，但guarded self-attention仍快；不能把普通共享历史收益记为新整层贡献。

HYPOTHESIS：FFN2仅48输出tile、四MMA warp和SM120七段有序BF16提交可能限制吞吐；三个独立小投影96tile/十K段可能有驻留和分段写回成本。这些是源码支持的待分离机制，不是当前计时已经证明的根因。

UNKNOWN：FFN内部两部分各自的独立成本、计算与grid等待分解、动态local流量与硬件counter。阶段数据为嵌入自有kernel的globaltimer与独立Torch eager同数学操作，包含不同launch/同步边界；不可相加成整层或称纯设备kernel加速。旧A100计数不能移用于5090。

RECOMMENDATION：先保持精度契约，在融合FFN内部诊断生产/等待/七段计算与提交，选一个主因再改；然后处理三个独立小投影。暂不加完整DAG、统一warp角色或改tau。当前正常候选仍ffn-bias-last；本轮只改变测量入口及guarded硬件准入，不集成生产。

完整表：[STAGE-TABLE.md](STAGE-TABLE.md)；原始数值：[SUMMARY.json](SUMMARY.json)；固定入口：[SELECTED.json](SELECTED.json)；资源：[RESOURCE-AND-STOP-REPORT.md](RESOURCE-AND-STOP-REPORT.md)。
