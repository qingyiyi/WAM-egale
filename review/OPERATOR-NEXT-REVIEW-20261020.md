# 给网页版的下一轮审查请求

请固定本页所在commit，不跟随main替代该版本。只读审查和设计建议，不把建议当GPU/训练/TEST授权。先读CURRENT-20261020.md及其中M34实际结果、DIAGNOSTIC-RESULT、PRECISION-CUTS、CLAIM-EVIDENCE-MATRIX，再读whole_layer.cu、whole_layer.py与固定Torch/CUTLASS上游来源。原4f04 paused72及review/RESULTS.json只作历史。

我的目标是一个完整Transformer block放在一个persistent kernel内，包含self/cross attention和所有投影、FFN、norm/RoPE/调制/残差/KV；在内部划tile，由职责不同的warp领取和执行，借鉴MegaMoE思想，不做30层大融合。baseline与共享依赖冻结，修改只能在optimized；不动投机代码、阈值或旧失败。A100 SM80，私有CUDA12.4、Torch2.9 cu126；不能默认TMA/WGMMA/UMMA/动态寄存器分配可用。

实际原型已经单kernel运行：8warp分调度/两加载/四计算/写回，64×64×32 GEMM、Q16/K64 attention，cooperative阶段barrier，全局中间量，172 registers/thread、59616B shared、理论1 CTA/SM。9真实签名的结构/tile/输入/RNG等通过，原hidden输出全部逐位FAIL，部分KV写入数值FAIL；耗时为native的5.12–11.28倍。600调用有限任务已关闭，不再扫参。

请解释并设计下一步：

1. 原型相对MegaMoE缺少哪些真正影响速度的执行机制？区分单kernel结构、角色分工和异步生产者消费者流水，不只建议改名字或普通融合。
2. B14阶段计时GEMM74.66%，其中cross K/V昂贵；但成熟cached/shared-prefix/Graph本来就可消除重复投影，不能把cache包装为新贡献。如何在这个强baseline上选择一个最小且可证伪的tile流水原型？
3. 同输入Q和FFN2控制有差异；video的已保存attention上游一致但attention输出不同；cross K/V九例一致。cuBLASLt API重建得到B2 FFN2 splitK7、B14 Q/FFN2 splitK2，但B2 Q splitK1也有68元素差异，旧四种累加公式都失败。请基于固定上游公开代码判断：维持当前数值契约有哪些可行设备原语，哪里存在闭源库或线程布局限制？不要保证抄Torch host wrapper就能在device内复现cuBLASLt，不建议无界猜配置。
4. 若提出改算术，请独立列为新数值路径，说明所需前瞻校准/质量预算，不能覆盖旧逐位门或将match-v1当匹配Torch。若原契约下本轮不可行，请明确止损与最小缺口。
5. 给出最多两条设计路线、默认只选一条；写具体tile依赖/ready条件/warp角色/workspace代际/死锁规避和强baseline。先设备原语→一层完整正确性→成本，不扩30层或新环境。
6. H1只有限支持，H2缺同冻结7-query逐个/批量对照，H3当前新调度无正收益，H4仍缺质量约束闭环证书。不要补造1/3/7矩阵、交互或HBM流量。L2/DRAM/bank-conflict/achieved occupancy当前UNKNOWN。

四套原6000完整局及新计时子面板另外列在CURRENT入口，训练/DEV曝光、首两S/组件失败和长尾保留。四质量字段false。计时重复不增加独立N，Object首轮0episode CPU实施失败不重试、面板最多330/360。不要用target call减少代替ΣT_round/Σg，也不要把7候选说成互相注意的Q112。

请输出“源证据→观察/假设/未知→唯一默认路线→最小预算和止损→到10/22的go/no-go→允许写入论文的claim”。10/23冻结、10/28主表、11/01 18:00北京中文证据包；不要保证论文录用。需要原tensor但公开快照没有时标UNKNOWN，不从JSON表声称独立重算。
