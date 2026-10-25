# 双CTA共享FFN1权重：本轮实测

## OBSERVED

当前判定：COMPLETE_NEGATIVE_RETAIN_S，保留S。本轮评估D，完整层/阶段数字见[阶段表](STAGE-TABLE.md)和SUMMARY.json。原S/W二进制独立保留；L仅cluster启动、配对映射与起止cluster同步，不共享B。

两CTA负责不同M64行块、同N128权重列，共224对/448输出tile。每CTA仍W0 TMA、W4–7 MMA、W1–3 bias/GELU/store；输出不翻倍、无跨CTA输出归约。完整K3072用48个K64，保留FP32累加到末尾、bias→BF16→GELU→BF16。FFN2七段K2048递推及提交函数与W逐字节相同，FFN阶段grid屏障保留。

D中rank0全局加载A+B，rank1仅A；rank0将16KiB swizzled B异步bulk复制到rank1已有local槽，因此原ldmatrix保持读本地shared。rank1先arm ready计数并发目的credit；复制完成后ACK源槽。源EMPTY等四MMA读完及ACK，目的EMPTY等四MMA读完。预填两个槽后逐步回填step+2，没有每K cluster.sync，只有初始化/排空的cluster汇合。

三真实fixture、两进程：27捕获tensor、L对照tensor、七段FP32 partial/BF16 running、有效KV/输入/CPU-CUDA RNG、非默认stream、计时前后与插桩输出全部有限逐位通过。实际角色/传输/提交计数通过；不能外推所有输入或机器人闭环。

实际计数对应D的FFN1：A 21504次×8KiB＝168MiB；B 10752次×16KiB＝168MiB；DSM 10752次×16KiB＝168MiB。W/L的B为21504次＝336MiB。逻辑global输入合计504→336MiB，但增加168MiB DSM复制；这不是硬件DRAM读流量测量。

## 归因与UNKNOWN

D-L是共享B协议的净变化，包含DSM复制、credit/ACK与生产者等待；L-W包含cluster启动、配对映射及起止cluster同步，不能称纯cluster启动成本。D-S决定是否值得替换保留版。

新增接收方依赖链是global TMA B完成→peer copy→ready→MMA。虽保留双槽，源槽复用还需对端ACK。减少global加载是否抵偿这些成本由实测决定；本轮没有细分copy/握手计时或GPU硬件计数器，不能把总差精确归因DSM带宽，也不能宣称HBM减半/SM利用率提高。阶段观察器扰动单列SUMMARY，不进入正常完整层性能。

## 追加等待定位（同预算内，非新候选）

[等待表](WAIT-ANALYSIS.md)：接收方READY约2.79μs/K64、发送方EMPTY约3.32–3.33μs/K64、MMA输入等待约2.90–2.94μs/K64；bulk提交约0.023μs。三fixture观察器精度/状态通过，整层扰动0.49%–0.94%。这些重叠区间不可相加，也不分离纯DSM网络时间。本轮负结果主要是复制/交接路径，而不是cluster启动、shared增大或occupancy下降。

D FFN1约558–561μs，同场W约153μs；L完整层相对W快0.98%–1.19%，相对S约快0.56%–0.65%。L只是本轮控制，未自动替换S，也未将这点小增量称为稳定新收益。

## RECOMMENDATION

按两进程完整层决定采用，不因理论加载减少或局部阶段改善而放行。若负收益，保留S与本轮全部D/L证据；下一步优先分析FFN1同tile输入等待/结果排空，不继续扩大cluster或重复旧驻留融合方案。现有四科学权限保持false；没有修改投机侧、baseline或任何旧失败。

## 固定入口

- 本轮源码：`whole_layer.cu`；`FFN1_SHARE_B=1`为D、0为L，同源编译，独立模块/二进制。
- 原W：`W-whole_layer.cu`、`W.so`；原S：`S-whole_layer.cu`、`S.so`。
- 原始结果：`measure-1/RESULT.json`、`measure-2/RESULT.json`、三份七段numeric-candidate PT。
- `SOURCE-AND-SCHEDULE-AUDIT.json`、`CPU-REVIEW.md`、`JOB.json`及phase STATUS是审计/预算索引。
- 当前公开SM120源码入口：`/rjs/cxz/论文探索/.pipeline/experiments/rtx5090-block-port-20261023/workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu`。

## 收尾

4836次完整block调用；生命周期17.09分钟，remote产物约167.22MiB。9个phase正常退出，无GPU timeout/失败；CPU文本生成的一次定位错误已保留。S公开入口三fixture再次逐位通过；15个关键源码/二进制/原tensor回传逐字节核验，无SHA。owned进程出生身份核验已退出，5090清空；不关闭实例。
