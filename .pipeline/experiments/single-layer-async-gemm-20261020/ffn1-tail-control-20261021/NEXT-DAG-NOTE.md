# 下一步DAG边界（源码分析，尚未实现/运行）

OBSERVED：whole_layer.cu约687–692行中，cross Q依赖R1→norm2；cross K/V的context_linear只读取p.context与固定权重，数学上不依赖self attention的hidden结果。K投影后RMS需要一整行3072维已写齐；cross attention需Q、归一化K和V均就绪。不能把一个投影输出tile完成当作整行RMS就绪。

当前B14 context为[14,512,3072]，合并M7168；M128×N128投影每个自然产生56×24=1344任务。各阶段保留适配的tile，108为常驻CTA池上限。context K/V当前仍在后续阶段各自全量运行，未作为早期尾波补位。

RECOMMENDATION：先用最小受控方式测出阶段最后完成CTA与较早空闲CTA之间的真实等待，区分长尾计算和同步本身。之后可让早期阶段无本阶段任务的CTA领取有限的cross K/V任务；任务完成一次，写入既有workspace并发布完成标记。必须拆开当前begin_stage/grid.sync包装和tile计算体，不能让不同CTA进入不同全局barrier序列。

第一轮不同时改CUTLASS数学体、tile、缓存或数值边界；不提前启动缺完整行输入的RMS，不让候选读别人的private KV。CTA共享workspace按任务结束同步后复用，K/V全局workspace保持到cross消费。若有限数量的投影tile延长早期关键路径而不缩短整层，应保留负结果并收缩，不扩成30层。

当前同源嵌入差约±1.3%，全局同步不再能解释旧大额集成差；cross K/V外侧各比native多约73–77µs、FFN1约57–58µs、FFN2约29µs。它们是分别测量的阶段差，不能直接相加当作完整层分解闭合。

UNKNOWN：动态spill/L2/DRAM、CTA等待分布、提前投影对cache与关键路径的作用，均未测。本文件不等于新GPU作业已登记/启动；任何后继作业沿M39原10h累计单独有界登记。
