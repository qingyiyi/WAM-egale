# 当前单层算子的简要带读 Prompt

你是一名 CUDA/CUTLASS 工程师。请用中文带我读当前实现，目标是先理解源码，暂不改代码、不启动实验。

项目根目录：/rjs/cxz/论文探索
当前优化目录：/rjs/cxz/论文探索/算子研究/wam_ops/optimized
baseline目录：/rjs/cxz/论文探索/算子研究/wam_ops/baseline（只读）

先按项目AGENTS/HW入口读取必要约束，然后聚焦以下文件，不展开全部历史实验：
1. whole_layer.py
2. csrc/whole_layer.cu，以及新调度csrc/ready_graph.h、csrc/ready_tiles.cuh
3. csrc/cutlass_gemm.cuh
4. csrc/gemm_pipeline.cuh
5. csrc/fa2_stage.cuh
辅助说明：WHOLE_LAYER.md。需要了解原数学时再读baseline/csrc/block.cu。

请按顺序分六小段讲，每段引用当前源码完整路径、函数名和实际行号，先讲第一段，等我追问或说“继续”再往下：
1. Python入口怎样绑定27组参数、输入/输出与KV；普通调用与debug的区别。
2. 从whole_layer()画出一层的计算链：调制/Norm→self QKV/RMS/RoPE/attention→残差→cross QKV/RMS/attention→残差→FFN/GELU→最终残差。
3. 用B14 action举例说明tile：hidden[14,16,3072]，GEMM可展平M224；attention仍是14个Q16实例，不能说成Q112。说明六小投影、context K/V、FFN1/2各自的M/N/K和tile任务数。
4. 先区分已验证顺序DSO和新源码中的ready_tiles入口（新路径尚未GPU验证），再解释108个常驻CTA如何执行任务：动态GEMM/B2路径用begin_stage/claim_tile/finish_tile；point/norm/FFN/B14双组attention是静态分派，不再重复重置队列。区分任务数与CTA数。解释每CTA的8个warp在GEMM、Norm、FA2中的具体分工，不能把某一阶段的分工套到全部阶段。
5. 跟踪一个投影tile的global→shared→register/MMA→shared→global；再解释self/cross FA2如何读取公共历史与候选私有KV，以及online softmax。只在理解所需时展开生成的FA2/CUTLASS上游代码。
6. 跟踪新claim_ready→run_ready_tile→完成发布：普通节点等待完整前置节点、Cross-K/V补空闲、每个FFN2输出tile的第二段独立就绪；只有启动时grid.sync，CTA/warp同步和全局内存可见性必须保留。解释FFN2改为先第一段后第二段的潜在并行损失，以及原子/fence开销。区分CPU图审查、GPU正确性、完整层性能三种证据，不用旧速度证明新调度有效。

要求：每段尽量300–500字，必要时用一个小图或shape例子；避免上来逐行翻译整份文件。区分当前已实现、历史方案、未来计划。性能与逐位一致性只能引用对应源码版本的实际结果，不能把旧版本数据当成最新数据，不能外推闭环/E2E。
