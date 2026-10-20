# 固定CUTLASS设备GEMM嵌入完整单层：有限结果

OBSERVED：本轮已实现、编译和运行，保持整个一层一个compute kernel。9真实签名（B2 action/B14 action/B2 video×短中长）整层输出与每格23个可配对中间量逐位通过；18次合法有效KV转移直接对Torch通过，输入/RNG/tile exact-once通过。63完整block门通过后，两个独立进程各21block完成B14工程计时，六份profiler trace各1 compute kernel。

| KV | 同轮native中位ms | CUTLASS整层中位ms | 两进程新/native耗时比 |
|---|---:|---:|---:|
| 288 | 2.6537 | 7.9375 | 2.992–3.015× |
| 1152 | 2.9433 | 8.1437 | 2.683–2.771× |
| 2160 | 3.3851 | 8.6423 | 2.541–2.558× |

每格四个普通事件样本（每进程两个），样本少，逐样本和范围在CUTLASS-SUMMARY.json；profiler样本不进入计时。GPU0启动前为空、运行中只有本worker计算context，原始NVML/命名空间信息保留。完整单层uncached eager，workspace准备不计，非最强cache/Graph、完整verifier或机器人E2E。

旧普通版B14相对各自native为5.022/4.515/3.956倍，本轮为约3.0/2.7/2.55倍。两批native本身有变化，不能将旧/新绝对ms之比当作严格同轮增益；目前只足以说新版本在当前配对测量中仍明显慢于Torch，尚未达到性能目标。没有追加旧二进制交错A/B。

## 实现与来源

固定NVIDIA CUTLASS v3.5.1，commit f7b19de32c5d1f3cedfc735c2849f12b537522ee。复用[mma_multistage.h](https://github.com/NVIDIA/cutlass/blob/f7b19de32c5d1f3cedfc735c2849f12b537522ee/include/cutlass/gemm/threadblock/mma_multistage.h)设备级mainloop，不调用host device::Gemm operator，不产生子kernel。device::Gemm类型仅用于提取SM80配置。保留上游许可证，生成器只改类名与两处同步范围。

本地实现：`算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh`；生成适配：`optimized/cutlass_gemm_codegen.py`。M≥128输出tile64×128、K64、3级cp.async，4个计算warp各32×64；将逻辑线程0..127映射到物理warp3..6，使用128线程named barrier。warp7保留原BF16 partition/bias顺序写回。小M原GEMM路径保留。

这一路径的warp1/2当前空闲，计算warp自行执行CUTLASS搬运/计算流水。尚不是最终的专职load/MMA/epilogue各角色充分重叠；调度仍按原stage，不声称完成MegaMoE式独立流水。全CTA依赖和最终BF16边界保留。

资源：108CTA、256线程，shared由旧121376B降至82464B（减少32.06%）；register仍255/thread，localSizeBytes由1880降至1536。资源数不是实际occupancy/spill流量。旧FFN ready开关仍False，新二进制ready路径没有本次单独验证/计时，不继承旧ready的通过。

## 原失败、资源与限制

首编译因两个CUTLASS const-pointer适配与一个旧FFN缓冲引用错误失败，0GPU计算；原日志/源码版本保留。修正后新编译成功，起点和绝对deadline不重置。当前profile12block+此次105block，原1380后累计1497/1500；另150次同输入Linear，仅用于诊断。未花剩余3次，不扩大实验。资源/清理详见RESOURCE-AND-STOP.md。

九签名有限通过不证明所有输入逐位，也没有本轮重复调用状态门；重复标UNKNOWN而非PASS。B6/其他shape、ready、30层、生产投机路径、最终接受决定都不在本门内。四质量权限与production_admitted仍false，baseline/投机/旧tile未动。

HYPOTHESIS：剩余成本可能来自monolithic寄存器压力、GEMM输出转换/单warp写回、全局中间张量和阶段同步；当前测量尚不能分解新版本各项贡献。

UNKNOWN：新CUTLASS版阶段profile、实测occupancy/L2/DRAM/stall、原版同轮重测、强cached/Graph矩阵。不能将普通缓存作为新增调度贡献。

RECOMMENDATION：保留成熟设备mainloop；下一步先针对新版本定位cross K/V、epilogue及local-memory成本，优先B14专用路径缩小monolithic资源压力。只在原语接近native后再做跨阶段ready tile填尾波。下一GPU阶段须有明确剩余或新增预算，不突破本轮1500帽。

证据：[门检](CUTLASS-GATE-RESULT.json)、[进程1](cutlass-timing-1-RESULT.json)、[进程2](cutlass-timing-2-RESULT.json)、[摘要](CUTLASS-SUMMARY.json)、[来源/清理](FINAL-AUDIT.json)、[旧版本当前profile](PROFILE-RESULT.md)。
