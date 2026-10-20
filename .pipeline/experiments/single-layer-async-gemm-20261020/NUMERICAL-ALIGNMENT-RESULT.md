# M39 后继 O1：完整单层有限输入逐位对齐

2026-10-20，北京时间。终态：`COMPLETE_FINITE_GATE_NOT_PRODUCTION_ADMISSION`。这是本次新授权的数值后继，不改旧 M34/M39 FAILED 或资源账本。

## OBSERVED

[原始九点结果](numeric-followup/FULL-GATE-3-RESULT.json)：B2 action、B14 action、B2 video，各短/中/长 KV，**9/9 最终 hidden 对原 Torch 逐位一致**。相同输入上的 native 对 Torch 同样 9/9；全部可配对的已保存 DAG 中间边界对 native 逐位一致。

六个 B2 action/video 起点，各按合法 `update=1→2→0` 执行，**18/18 更新输出与全部有效 KV/metadata 直接对原 Torch 一致**。九点重复确定性、输入不变、RNG 不变、tile exactly-once 同时通过。未比较无效 KV，未增加机器人任务或 30 层运行。

修复落在 optimized 内：

- `norm_codegen.py` 从固定 Torch LayerNorm 设备体提取原向量化 Welford/输出计算，映射到四个 compute warp；保留原 FP32 算术。
- 根据真实 cuBLASLt 调用记录，复现有限形状的 serial BF16 分段归约，部分和不加 bias，最终顺序为 `(当前 FP32 部分和 + bias) + 先前 BF16 部分和`。
- B2 M32/N3072/K3072 投影复现原 `sliced1x2`：交替 K32 块分别进入两个 FP32 累加器，最终 `(slice0 + bias) + slice1` 再舍入 BF16。

早期 bias 次序、提前 bias、转置 MMA、首次完整门的失败均保留在 [numeric-followup](numeric-followup/)；不只保留最后通过轮。关闭 cuBLAS BF16 reduced-precision 的诊断仅在隔离进程内使用，未修改原 baseline 或全局配置。

O1 共 **261 次完整 block 调用**，上限 300；最后完整门 72 次。最后版本资源为 108 CTA、256 threads、255 registers/thread、82464 B shared、2304 B localSizeBytes，理论 1 CTA/SM；不是实测 occupancy 或 spill 流量。

## UNKNOWN

有限 fixture 通过不能证明所有输入逐位一致，不能继承为接受路径、完整 verifier、chunk 或闭环质量。尚未测新版本两进程九点完整性能；未得到 NCU 流量、stall 或 achieved occupancy。`production_admitted=false`，项目四项质量权限仍 false。

## RECOMMENDATION / 当前执行

进入已授权 O2。先以独立 stage-clock 副本定位瓶颈，普通计时不混入插桩。实际大头仍是 cross K/V 投影、FFN 与归一化。只做有依据的单因素变化：归一化 IO 向量化，以及大 M 的 64×128×64 GEMM；小 M 保持 64×64×32。每次先检查相同真实输入，随后恢复完整数值门。O2 总上限 400 block、2 GPU 生命周期小时；最终性能要与同输入强 baseline 比较，不称机器人加速。
