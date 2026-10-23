# 网页版设计任务：单层WAM persistent Transformer的任务依赖图与调度

请用中文，基于真实源码，为我们设计完整单层Transformer的tile任务依赖图及kernel内部调度。现在只做审阅和设计，不把方案写成已实现或已获得性能收益。

仓库：https://github.com/qingyiyi/WAM-egale
固定算子审阅commit：319ac6a17daed34e80692796f72a97bf44f4a99f
请固定此提交读取源码和算子证据，不用main后续变化覆盖判断。main上的新增机器人排队进度与本次算子结构设计分开；历史TF质量快照的false不改。

## 先阅读这些实际文件

1. 算子研究/wam_ops/optimized/csrc/whole_layer.cu：完整层、claim_tile、self_qkv_projection_group、norm_rope、action_ffn_pipeline及同步。
2. 算子研究/wam_ops/optimized/csrc/cutlass_gemm.cuh：self_qkv8_tile、K分段与BF16写回。
3. 算子研究/wam_ops/optimized/csrc/fa2_stage.cuh：self/cross attention分组、就绪输入和private可见范围。
4. 算子研究/wam_ops/optimized/whole_layer.py，以及源码真实引用的构建/设备辅助文件，按需追踪。
5. .pipeline/experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md、RESULT.md、SELECTED.json、CURRENT-STAGE-SUMMARY.json。
6. .pipeline/memory/global/requirements/requirement-单层Transformer统一persistent内核调度全部tile.md
7. .pipeline/memory/global/requirements/decision-回退阶段全局同步并重新规划tile调度.md

若网页无法读取某个文件，请列出具体缺失路径、给出需要Codex补充的最小片段；不要假装读过，不据函数名字推断实现。关键判断引用固定提交的文件/行号。

## 目标与已知实现

目标：完整一个Transformer block放在一个persistent CUDA kernel中，内部包含self/cross attention、norm/调制、投影、FFN和残差，按tile或合理任务组调度。只做一层，不做30层，不退回host逐算子启动多个kernel作为最终方案。

A100-SXM4，SM80、108 SM。当前108常驻CTA，每CTA8warp/256线程，112384B shared/CTA、255寄存器/线程、静态local1016B、1CTA/SM。这些是当前资源属性，不是动态local流量或根因占比。远端不能提权，也不能访问NCU硬件计数器；可用CUDA Event、软件时间线、编译资源报告和有限数值核验。本机若有另获准窗口才能补硬件计数，不能作为设计必有条件。

当前输入为B14×Q16×D3072，24 heads、head_dim128，FFN宽14336，真实短/中/长KV为288/1152/2160。7候选×2 CFG；线性层可展平为M224，但attention仍保留每候选的Q16及合法共享分组，不能变成跨候选任意互相注意的Q112/Q224序列。

当前六个M224、3072→3072投影使用64×128输出tile、8个MMA warp、warp tile32×32、K64/3stage；每投影96个数学输出tile。Self Q/K/V目前共用288任务队列，task0..95算Q、96..191算K、192..287算V。三份权重和输出独立，队列入口/出口仍有grid同步。原两段K1536、段间BF16舍入、bias与累加顺序保留。

Self Q/K的RMS沿每token完整3072维归约，随后RoPE，RMS后先舍入BF16，旋转保留当前源算术边界。完成一个128维投影列块，不等于完整RMS输入就绪。当前64行投影块跨4个16token候选，仍需按候选/CFG/合法G1或guarded G2分组确定实际attention依赖。

历史K/V为本次调用只读输入，候选private K/V保持独立；当前query的self attention需要本候选全部合法可见private K/V，不能只因某个Q行就绪就忽略未完成的key。共享历史不等于共享private suffix。

Warm路径已预计算最终Cross K（含RMS）/V；此时它们是合法缓存输入，不再生成对应计算任务。Cold路径单独设计，不能把已缓存节点再次拿来填补执行空闲，或把预计算当新增调度收益。

FFN1/GELU/FFN2/残差3已有四组局部依赖流水，不是完全阶段同步；FFN1有108主tile+12尾tile，FFN2为96个分段任务、48个最终输出tile，部分和提交保持原顺序。请读action_ffn_pipeline后再画图。其余多数阶段仍使用grid同步；当前完整层23次grid同步，数学tile为3408/3480/3408。数学tile数、调度节点数、常驻CTA数必须分别定义。

## 当前证据的边界

同轮、双方同等Cross KV预计算、warm CUDA Graph下，完整单层相对原Torch静态dense-history路径为1.2967/1.8469/2.9910×；不是最强guarded实现增量、30层、chunk或机器人E2E。

阶段诊断原Torch/自有：Self QKV联合约0.86×，Self输出投影约0.71–0.80×，Cross Q约0.80–0.82×，Cross输出投影约0.74×，LN3约0.61–0.62×，FFN共同区间约1.01–1.02×。阶段对照的缓存/计时边界不同，不能相加预测完整层收益。QKV联合队列本身只带来0.08%–0.44%的整层额外改善，不称显著。

较大128×128输出tile虽通过有限精度门，整层慢9%–11%，已回退。先前过细的ready原型已撤回，失败和负结果保留。当前三个可执行候选分别完成9真实签名/36结构控制/225次有限完整调用核验，不证明所有输入或闭环质量。

## 本次要你判断的问题

用户希望最终形成任务依赖图：前驱全部完成、未完成前驱计数降到0的任务才可派发。QKV按行块交错的顺序留到这个统一设计阶段再决定，当前不单独实现交错。

请客观比较三种可落地机制：静态分配/预排调度；静态DAG上的全动态就绪队列；静态亲和分配加有限动态补位/偷取。不要预设全动态更快，也不要将静态DAG等同于静态任务分配。

必须具体回答：

1. 图的最小合理节点是什么？投影输出tile、完整RMS行或行块、FA2合法任务组、FFN生产/消费组如何映射？给出节点/边/计数器规模，说明避免重新产生大量过细任务的办法。
2. 哪些前驱是数据依赖，哪些只是当前实现的全局屏障？RMS全行归约、attention全部可见K/V、输出投影完整head维、FFN2完整归约及ordered BF16部分和分别何时就绪？输入/输出buffer别名、共享workspace复用和历史有效长度也要入图。
3. 如果Q/K/V改为行块内交错派发，何时真正能释放RMS/RoPE和attention？普通单候选与G2合法分组、跨CFG/行块边界分别处理；不要把“连续发出”当成“已完成”。
4. 入度计数初始化、producer完成、release发布、consumer acquire、恰好一次入队、queue head/tail、退出判定怎么做？给SM80可实现的伪代码。无任务时CTA怎么办？证明不会所有消费者占满资源而阻塞生产者；仅“图无环”不足以证明执行器无死锁。
5. 原子热点、全图扫描、队列空转、任务切换、权重/L2局部性和关键路径怎么权衡？能否用隐式依赖/行块事件/分层队列替代每条细边原子更新？确定性优先级是否有帮助？
6. 一个kernel内如何处理8warp GEMM、两个4warp FA2组和归约任务的不同执行契约、barrier编号、cp.async完成与union workspace生命周期？CTA/warp各自领取什么？哪些职责固定、哪些可按阶段切换？不要为“调度专职warp”无依据牺牲当前8warp MMA，也不要把混合分支造成的最大寄存器/栈帧代价忽略。
7. 给出一条你推荐的主方案和有证据才触发的备选。解释为何适合当前形状及剩余瓶颈；如果静态策略更合适，请直接说。估算仅做可检查的成本模型，未知参数标UNKNOWN，不虚构加速比或计数器。

可查MegaMoE、其他persistent GPU任务图、CUTLASS与FlashAttention的原文/官方实现，区分硬件和算术契约。MegaMoE只借鉴调度思想；A100不能默认使用TMA/WGMMA/UMMA。外部关键判断给直接原文/官方链接及实际版本；不要只凭名称类比Transformer与MoE。

## 请交付给Codex的设计结果

A. 基于源码的完整单层DAG（Mermaid）及节点表：公式/输入输出、数学tile数、调度节点数、前驱、ready粒度、读写与复用生命周期、warp协作组、现有同步和可替代条件。
B. 静态/动态/混合的比较表、选定主方案及原因；给调度器与producer/consumer伪代码和死锁/可见性论证。
C. 最小增量实现次序，最多一个首要原型和一个备选；保持optimized内迭代、baseline/投机生产/旧tile不改。先CPU DAG核查，再有限真实输入核验，再短中长KV完整层比较。不要一次改tile、GEMM数学、归约树、缓存引擎和调度。
D. 对照和验收：相同计算体的静态/动态/混合调度；原Torch唯一外部性能对照；单独量化队列/计数/同步与等待；同输入/资源/缓存边界，保留输出、全部有效KV/RNG及BF16边界。调度可重排独立任务，不改变同一输出的浮点归约顺序。
E. 明确可移除和必须保留的同步、风险及停止条件。如果剩余关键路径或维护开销令收益空间很小，应建议保留简单方案，不为了去掉所有grid.sync牺牲整层速度。

结论分为：源码已证实、设计推导、待实测、缺失信息。不要保证性能或质量，不把本设计当作GPU执行/预算授权，不把已有缓存或普通batch收益重新包装为新贡献。
