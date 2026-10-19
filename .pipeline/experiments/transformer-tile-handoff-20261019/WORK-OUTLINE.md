# 算子tile化与persistent warp：实际工作大纲

2026-10-19。当前交付是CPU形状/源码/DAG审查，不是新的GPU实验。用户已明确目标为tile原语及persistent kernel内的warp调度，并建议参考MegaMoE；以下具体子图与预算是可审阅实施建议，不预设性能或论文创新成立。

## 已有可用基础

- M28完整native baseline/optimized、有限逐位与动态block性能已交付；baseline从此作为冻结对照。实现仍是多个host顺序发射的CUDA kernel。
- FFN1+GELU有SM80设备tile原语，保留Linear输出BF16与GELU输出BF16两次舍入；FFN2目前依靠匹配原Torch的cuBLASLt，旧device FFN2逐位失败是未解决门。
- [交接材料](HANDOFF.md)给出真实9签名/27参数、M32/224和合法共享范围；[DAG](BLOCK-DAG.json)明确归约与状态依赖。
- [MegaMoE参考](MEGAMOE-NOTES.md)已经固定官方源码，未运行外部实现；可移植机制与SM100专有原语分开。

## 依赖顺序与交付

| 阶段 | 要做什么 | 新增部分/目的 | 交付与准入 |
|---|---|---|---|
| T0，当前完成 | 冻结接口、真实shape和可见性；标注现有tile及强baseline | 防止把已有M扩行/缓存重命名为创新 | HANDOFF、SHAPES、PARAMETERS、DAG；原始stride未知明确保留 |
| T1 | 选定共同最佳B14 shared-prefix/O3/cached/可用Graph路径；采剩余热点和实际tile | 确定可改的GPU开销，而不是只优化Python提交 | 同输入/缓存生命周期，准备/pack/workspace全计入；profile与正式计时分离 |
| T2 | FFN1+GELU保持同tile/同算术，增加resident CTA领取多个tile | 首先隔离persistent任务调度的增量 | 普通grid vs persistent，B2/B14和诊断B6；逐位、任务覆盖、无越界/无漏tile |
| T3 | 对齐FFN2设备原语 | 消除host GEMM边界，建立完整FFN设备执行前提 | 原FFN2逐位失败不改；按真实输入、中间量和归约路径定位后新版本重验 |
| T4 | FFN1/GELU→FFN2先按行块ready交错，再视证据细化K块；逐项加入warp角色 | 尝试重叠生产消费、减少launch或中间量往返 | 就绪/发布/释放、bias一次、全部K归约、buffer generation、终止与无死锁证据 |
| T5 | 加候选/CFG/共享数据的task metadata与调度策略 | 利用新增合法tile和共享结构提高单位有效工作的执行效率 | 与同tile普通grid、无候选感知persistent分别比较；共享history/private边界保持 |
| T6 | 从子图接到完整block、30层verifier和完整chunk正确性 | 核验局部收益能否传递，计入调度与pack等全部开销 | 输出/分数翻转/合法路径/scheduler/动作/RNG/全部有效KV；不借旧证书 |
| T7 | 质量门允许后，独占硬件正式性能与完整任务统计 | 回答真实有效提交与闭环收益 | ΣT_round/Σg、chunk、自然episode、cold E2E分别报告；原四项false不自动改变 |

T2与T3可以先做有界实现审查；完整persistent FFN接入依赖T3通过。不能把带host cuBLASLt的路径报为“完整FFN驻留kernel”。初轮主攻FFN子图；共享历史attention仅在profile支持时作为备选，不同时展开大网格或30层融合。综合HW的具体任务/ready设计见[DESIGN.md](DESIGN.md)。

## 候选感知到底新增什么

普通库已经把7候选的Linear行从M32扩到M224；Flash已经内部tile化。这些不是待实现的新贡献。新增执行组织应当是：把相同权重/合法共享历史下的工作描述为有依赖的tile任务，在常驻CTA内分配warp角色，并按ready状态、共享数据和资源约束执行子图。

对FFN，候选行本身没有特殊数学，普通token-row调度可能同样有效。若候选metadata没有独立收益，就如实称为通用persistent子图优化；只有shape×引擎交互证据支持时，才称投机工作负载带来额外优势。attention方案可研究同CFG的公共history复用，但private KV仍逐候选独立，不能把7候选拼为互相注意的Q112。

应至少区分四个实现因素：共同强基线；相同算术的tile原语；persistent队列/静态调度；warp分工及候选感知策略。不要一次全部打开再把差值归给某一个因素。

## 初轮实验登记建议

建议至多两个工作日、总GPU生命周期≤7小时，先登记绝对截止、输入manifest、各阶段forward/存储上限与清理身份，再使用自然空闲窗口。当前M30的GPU预算仍为0，不以这份建议隐式启动作业。不自动租卡、改时钟、杀他人或扩闭环实验。

输入用现有真实action短/中/长K288/1152/2160、sigma分层、1/3/7状态合法子集；B6仅诊断，当前原语仅准入M32/224，扩入口必须重新核验。对同一7-query集分别比较逐B2与现有B14，前者只证明batch经济性，不冒充顺序机器人加速。

每次记录完整输入/条件/缓存身份、dtype/stride、任务总数与完成数、每任务有效M/N/K、launch、等待、workspace、CTA、寄存器、shared memory、occupancy及L2/DRAM。L2请求减少不能写成同等比例HBM减少。保存原始轮次、A/A与资源中断；准备/布局/scatter/归并开销全部计入。

机制主要结果为ΣT_round/Σg；g为实际合法scheduler更新，既不等于7候选，也不等于环境步。算子replay没有运行接受提交时，g必须NA，不能为得到该指标捏造实际提交数。

## 逐级止损与日程目标

- FFN2严格门不通过：停止完整FFN集成，保留FFN1调度的独立结果和数值缺口。
- 新调度相对最佳B14的30层改善不足约5%或不能稳定复现：不继续大融合；完整chunk改善不足约3%且接近噪声则收缩系统主张。这些是投入筛选线，不是改写质量门。
- 普通B14已获得全部收益、只有通用优化或质量未收口：分别收缩协同主张/生产集成，不改变历史FAILED和四false。

| 北京时间目标 | 默认工作 | 若受阻 |
|---|---|---|
| 10/19 | CPU接口/DAG/MegaMoE交接，远端继续既定Object队列 | 缺原stride/最强路径明确UNKNOWN |
| 10/20–21 | 有界T1/T2/T3及一个最小子图样机 | 不扩结构、接受器或训练搜索 |
| 10/22 | 质量/系统go-no-go，审视强baseline增量 | 缺证据即保留负结果并缩小claim |
| 10/23 | 冻结候选/API/engine身份 | 不以冻结日期冒充质量通过 |
| 10/24–28 | 获准的集成/分层结果，10/28主表收口 | 质量/独占未放行则性能待定 |
| 11/01 18:00前 | 中文证据包供用户飞书整合给老师 | 列缺口，不保证论文录用或全部通过 |

以上为目标日程，排队时间另列，不保证硬件空闲或数值门一定通过。投机生产、原tile、保护state/rules和旧证据全保持；仅baseline/optimized两份活跃算子实现，后续改动限定optimized。中文论文正文仍在会话交付，不新建本地初稿。
