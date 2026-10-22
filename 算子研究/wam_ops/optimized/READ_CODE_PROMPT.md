# 当前单层算子的简要带读 Prompt

你是一名 CUDA/CUTLASS 工程师。请用中文带我读当前实现，目标是先理解源码，暂不改代码、不启动实验。

项目根目录：/rjs/cxz/论文探索
当前优化目录：/rjs/cxz/论文探索/算子研究/wam_ops/optimized
baseline目录：/rjs/cxz/论文探索/算子研究/wam_ops/baseline（只读）

先按项目AGENTS/HW入口读取必要约束，然后聚焦以下文件，不展开全部历史实验：
1. whole_layer.py
2. csrc/whole_layer.cu；当前是阶段同步版加按需调制，ready源码仅在证据归档中
3. csrc/cutlass_gemm.cuh
4. csrc/gemm_pipeline.cuh
5. csrc/fa2_stage.cuh
辅助说明：STAGE-MAP.md、WHOLE_LAYER.md。需要了解原数学时再读baseline/csrc/block.cu。

请按顺序分六小段讲，每段引用当前源码完整路径、函数名和实际行号，先讲第一段，等我追问或说“继续”再往下：
1. Python入口怎样绑定27组参数、输入/输出与KV；普通调用与debug的区别。
2. 从whole_layer()画出一层的计算链：调制/Norm→self QKV/RMS/RoPE/attention→残差→cross QKV/RMS/attention→残差→FFN/GELU→最终残差。
3. 用B14 action举例说明tile：hidden[14,16,3072]，GEMM可展平M224；attention仍是14个Q16实例，不能说成Q112。说明六小投影、context K/V、FFN1/2各自的M/N/K和tile任务数。
4. 当前按需调制版有108个常驻CTA；动态投影领取任务，其余point/norm/FFN/B14双组attention静态分派。逐项用STAGE-MAP说明tile数、公式和同步点；区分数学tile、领取粒度、CTA数。
5. 跟踪一个投影tile的global→shared→register/MMA→shared→global；再解释self/cross FA2如何读取公共历史与候选私有KV，以及online softmax。只在理解所需时展开生成的FA2/CUTLASS上游代码。
6. 解释普通B14路径36次全grid同步的来源，以及modulation4如何把原表加法移入消费处；debug额外物化表时为37次、FFN2两K段如何有序写回。再解释已回退的ready实验为什么将约1.4万逻辑tile逐个领取可能昂贵；不得称当前源码仍在使用ready队列。先讨论更粗调度粒度与合法融合，不自动改代码。

要求：每段尽量300–500字，必要时用一个小图或shape例子；避免上来逐行翻译整份文件。区分当前已实现、历史方案、未来计划。性能与逐位一致性只能引用对应源码版本的实际结果，不能把旧版本数据当成最新数据，不能外推闭环/E2E。

当前还融合了B14 self Q/K RMS+RoPE：rms_pair_tile/norm_rope保留归约、BF16舍入与FP64旋转；普通不物化SQN/SKN，debug保留审计值。以STAGE-MAP和RMS融合RESULT为当前状态。
