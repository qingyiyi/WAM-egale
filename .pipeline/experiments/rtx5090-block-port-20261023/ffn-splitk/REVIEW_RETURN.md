# FFN2局部split-K实际结果

终态：CLOSED_RESEARCH_ADOPTED。评估并保留 **SPLITK-M64-FFN2**；冻结GRAIN的A源码和独立A.so继续保留。

## OBSERVED：改动与收益

只将FFN2的96个输出tile、每tile七个K2048片段从同CTA串行计算改为672个跨CTA领取的计算任务。GPU仍为170个cooperative常驻CTA，不是672个同时驻留CTA。四个行组按partition-major领取，96个输出tile各有有序提交计数；部分和仍按原BF16顺序归并，最后一段才加bias和门控残差。四MMA warp、M64×N128/K64、原cp.async三级主循环、FFN1和其他阶段不变。新增96×4=384B全局计数器；共享内存没有增加。

|完整层，无插桩|短KV288|中KV1152|长KV2160|
|---|---:|---:|---:|
|原Torch/新版本|1.1493×|1.6245×|2.0497×|
|原Torch/同轮GRAIN|1.0872×|1.5405×|1.9580×|
|新版本相对GRAIN耗时下降|5.404%|5.174%|4.474%|
|Guarded/新版本|1.0156×|0.9978×|0.9750×|

两独立进程三档改善均同向，四臂交错测量、eager无Graph、同一GPU/输入/预计算CrossKV；冷构建另列。Guarded比较显示短档快约1.54%，中档慢约0.22%，长档慢约2.56%，不能忽略强baseline。

独立FFN2＋bias＋门控残差区域，相对串行版快1.431/1.434/1.440×；原Torch/新版本为1.006/1.013/1.018×，可称这一区域接近Torch。原Torch纯Linear另列约0.114–0.115ms，而此区域约0.132ms，不能说纯GEMM已经对齐。FFN2本身没有KV长度维度，三档只是来源fixture不同。

完整FFN融合区间的原Torch/自有为0.869/0.862/0.872×，自有仍慢约15%–16%。Self输出、CrossQ、Cross输出约0.67–0.71×，依旧是明确差距。

## 正确性与实现边界

三份真实B14/Q16/block0，26个中间量＋输出、normal/debug、有效KV/输入/RNG有限逐位通过；public默认流与非默认流均通过。四行组各领取168任务、96个tile各提交七段、每行完成24个输出，计数正确。320个有限CPU调度模型通过，逆序负控检出死锁；无环说明及适用前提见SCHEDULE-AUDIT.md，不能脱离cooperative全驻留前提推广。

normal和原A资源相同：170CTA、256线程、100096B shared、255寄存器/线程、784B静态local/线程。独立probe双方168寄存器/线程、0local，与整层编译不同；不能直接用两个测量相减推算容器开销。静态local大小不等于实测动态local流量。

此次TMA没有被启用；此前TMA负结果不改写。数学基元ffn2_partition64/ffn2_commit64相对A保持原样；只改变执行组织。当前仍是单层研究入口，四科学权限和production_admitted全部false，不代表30层、在线接受或机器人E2E收益。

## HYPOTHESIS / UNKNOWN

受控改动支持原FFN2的K段串行组织是性能损失之一，但尚无本轮硬件计数器归因，不能将全部收益归于某种带宽、occupancy或Tensor Core利用率变化。融合FFN剩余差距尚未被完全分解。TMA能否在新调度下获益未知。

## RECOMMENDATION

保留本次split-K版本。下一步优先对Self/Cross输出投影及Cross Q做同类有序K分段候选，仍分别核对分段和舍入，不能直接复制672任务数；这三项仍占候选插桩区间约20%–24%。随后在已稳定的调度上隔离TMA搬运/寄存器预取，避免再次同时改变阶段数和交接。没有新增候选在运行。

## 交付和资源

完整17阶段按实际融合展示，全部是本候选本轮测量，见[STAGE-TABLE](STAGE-TABLE.md)。正常完整层与诊断插桩分开，阶段区间含同步，不相加替代整层时间。

正常、profile、standalone三个构建成功，七个GPU验证/测量进程成功，无GPU失败/超时。完整block7389/15000，独立 FFN2 5094/12000。GPU worker进程生命周期合计55.718秒不等于CUDA active；保守整轮计30分钟，累计31486.413/36000秒，余额1.254小时。新增证据约47MiB，低于1GiB上限。两次CPU清理收据命令因未激活环境返回127，修复后记录，不涉及GPU试验重跑。

20个owned guard/child身份核验均已退出，5090显存0；租卡实例未关闭。源码及二进制六份回本机byte一致，最终快照SELECTED-source，原A保留。双A100在00:40只读检查仍有RoboTwin模型/模拟上下文，后台巡检正常；本轮不改队列、不重算SR。

## 固定入口

- [当前实现](../workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu)，action_ffn2_splitk约584行；public入口whole_layer.py。
- [固定身份](SELECTED.json)、[完整层结果](reports/SPLITK/SUMMARY.json)、[独立FFN2结果](reports/standalone/FFN2-SUMMARY.json)。
- [独立审查](INDEPENDENT-REVIEW.md)、[清理](OWNED-CLEANUP.json)、[回传核验](RETURNED-BYTE-CHECK.json)、[资源](RESOURCE-AND-STOP-REPORT.json)。
