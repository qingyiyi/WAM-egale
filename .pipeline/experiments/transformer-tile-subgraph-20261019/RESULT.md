# M31：独立optimized与静态tile子图已实现

终态：`ISOLATED_STATIC_TILE_SUBGRAPHS_FINITE_BLOCK_VERIFIED`。本轮完成用户批准的执行路径隔离和第一版可执行tile子图，GPU仅共享正确性。新完整30层、性能与persistent调度未完成，不继承旧实现的对应证据。

## 实现

baseline目录和其实际依赖的旧attention/FFN动态库、源头、编译入口冻结，88个源/依赖/保护文件核验不变。optimized不再继承baseline，独立参数容器、native绑定、C++执行器和`wam_optimized_block_cuda`。公共install.py只为optimized分派新执行器；baseline默认行为经同输入复核。新ELF没有旧WAM动态库依赖，全新GPU进程实测没有导入baseline模块或加载旧WAM DSO。

默认optimized为sequential等价路径且FFN1融合关闭；另有tiled，以及tiled＋FFN1融合，避免把既有融合混成新调度收益。`csrc/block.cu`实际执行self/cross/FFN三段子图。同一stream按阶段推进，严格等待全D归约、全部head和合法私有KV；没有跨阶段ready队列。

attention的显式task包含batch/head/query_block/kv_split/kv_splits，直接进入固定Flash设备mainloop，保留在线softmax和原split/combine。FFN1提取显式row/column tile的CUTLASS MMA/epilogue，保留原BF16→GELU舍入。已有库本身已经tile化；本轮新增是独立可调用的显式坐标入口及可执行子图，并未证明新增速度。

QKV/Wo/FFN2继续cuBLASLt，尚非可从persistent worker调用的GEMM tile。旧FFN2数值失败保留。本轮不是完整设备化大融合，也没有新的生产集成。

## 实际验证

| 项目 | 覆盖与结论 |
|---|---|
| 完整真实block | 六action：B2/B14×短中长；三video：B2 Q128×短中长，共9组 |
| 对照 | 原Torch、冻结baseline、独立sequential、tiled、tiled＋FFN1；每组公共forward与debug输出逐位一致 |
| 中间量 | sequential/tiled每组24项；action fused每组23项，FFN1原输出不物化；video fused实际沿用顺序FFN，24项 |
| KV状态 | B2 action/video六组×四执行配置×write1→write2→read0，共72项逐位通过，含真实容量驱逐 |
| 输入/API | 保存identity、输入、CPU/CUDA RNG、原参数地址/grad flags、类安装还原及非法配置拒绝检查通过 |
| 真实tile dispatch | short B2/B14实际kernel名称观测到split/standard attention tile和FFN tile，输出逐位；不使用profile时间作性能 |
| 独立进程 | 另一个全新进程9输入/54状态通过；无baseline Python模块/旧WAM库加载 |

最终原始证据：[完整矩阵](final-identity-state-audit/results/RESULT.json)、[独立进程](optimized-isolated-process/results/RESULT.json)、[ELF](ELF-DYNAMIC.txt)、[源冻结](FROZEN-SOURCE-RESULT.json)。输入均为既有真实记录，不是新机器人episode；action有效K288/1152/2160，video本批有效K272/1136/2144，cross K512。有限输入通过不证明所有输入或完整闭环。

## 错误与限制

首次综合attempt9个block/72状态已通过，但检查器未识别mangled FFN kernel名，原总状态REJECTED保留；独立派生审计和最终新attempt完成修正。旧工具case_id把video显示为action，原input_path保留正确来源；最终explicit identity/label及写状态后的不可变检查补齐。CPU构建收据finalizer曾缺文件，编译本身成功；详见[修正记录](VALIDATION-CORRECTIONS.md)。未修改数学源码来迁就这些工具问题。

本机其他程序占约35GiB，余约4.9GiB，只运行完整block。新30层需要约28GiB可用显存，当前BLOCKED；未启动耗资源的整模型尝试。无独占性能；无强B14/Graph比较；无新chunk/E2E；四质量权限false。

## 资源和下一步

3个GPU子作业实际生命周期合计81.007/1800秒，483/600次独立block意图，0整模型/0机器人episode，所有自有进程退出，CUDA active时间未单测。CPU两核编译成功、无系统安装；新存储远低于512MiB。unused预算不自动改投其他实验。[账本](RESOURCE-AND-STOP.json)。

下一阶段先在显存窗口补新30层正确性，并确定适用于当前block的最强B14执行边界；随后在optimized以已验证tile入口实现有界persistent worker/依赖ready。新增收益必须实测，不把普通tile、缓存或host迁移重新包装为MegaTransformer创新。

源码与使用见[optimized子图说明](../../../算子研究/wam_ops/optimized/SUBGRAPHS.md)。baseline、投机侧、原tile和旧失败保留。
