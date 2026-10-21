# 本轮机制证据与边界

## 已观察

1. 固定真实 B14 context=[14,512,3072]，M7168、N3072、K3072；历史KV288/1152/2160只代表不同fixture及其前序，不改变此GEMM的K。独立投影固定256×128×64、8warp、672CTA，不继续搜索CTA/warp数量。
2. 本机NCU：修改前K/V各2,064,384次shared-store bank conflict；pad136后两次均0。shared-store wavefront从2,752,512降到688,128。独立DSO始终254regs/local0/shared147472 bytes/1CTA每SM。没有证据将独立投影慢因归于local memory。
3. 远端两进程同前序计时：padding单因素使约615µs降到593–594µs（3.49%–3.66%）；第二候选四元素写回在padding基础上再减少2.75%–2.93%，约576.5–578.6µs，比本轮Torch原版stage慢4.07%–4.26%。两轮均固定输入/权重/加法顺序/BF16舍入，逐位通过、672tile各一次。
4. vec4最初脚本两个标签误读同一二进制；该差分对照无效且原件保留。paired-r2记录明确不同二进制路径，原GPU消耗计入；不能用无效对照的相近数字下结论。
5. 整层集成首次CPU编译被遗漏的generic dispatch模板参数挡住，0 GPU；修正记录见whole-pad136/BUILD-FAILURE.md。当前4warp单层的集成收益须读取后续r2完整结果，不能将独立投影收益直接外推。

## 解释与未确定部分

原IteratorC将MMA accumulator按连续128列放入shared。同一warp不同quad的行映射撞bank。将物理行stride设136后错开bank；逻辑输出仍128列，只有布局改变。四元素写回进一步减少搬运/循环次数，并保持逐元素FP32加法和BF16round。padding消除写冲突与改善时延有控制实验支持，但不证明它解释全部差距。

独立投影剩余约4%差距仍UNKNOWN：未分离主循环指令调度、predicate、寄存器依赖及cuBLAS的epilogue细节。NCU本机、计时远端分别列出，不混合绝对延迟。不把shared padding称为MegaTransformer新贡献，不宣称整层对齐或机器人加速已完成。两候选到此停止，不开启无依据大网格。

这种shared布局优化有既有资料：[CUTLASS GEMM/epilogue执行说明](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)、[CUDA Best Practices：shared bank与padding](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-c-best-practices-guide/index.html)。本地CUTLASS v3.5.1 `mma_tensor_op_tile_iterator.h` RowMajor accumulator iterator约3290、3390行提供实际lane映射依据；本轮修改采用现有设备MMA主循环。

本轮仍为单层真实tensor replay/算子机制测量，未运行机器人正式性能、30层或独立TEST。四科学权限维持false。下一步在整层精度/性能通过后，先审计剩余原语与整层资源成本，再就绪DAG；不把本次修改描述为已实现异步调度。

SASS补充：静态指令核查（SASS-STATIC-COUNTS.json）显示epilogue shared load由LDS.64变为LDS.128；最终全局store仍展开为两条32-bit STG.E，而不是一条STG.E.64。因此准确表述为“四元素搬运/循环，shared读取向量化”，不声称已生成64-bit全局写回。静态出现次数不等于动态执行次数。
