# Draft选型、扩大训练与verifier优化的并行安排

> 最新夜间结果（2026-10-15T05:45:28+08:00）：45次训练无稳定新权重，保留A_off；30条开发质量任务、新TRAIN与PF2/3/4已独审。算子融合已撤回，原tile不改。见[统一报告](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)、[扩大准备](NEXT-SCALE-READINESS-20261015.md)，HW记录 CHAT-20261015-OVERNIGHT-DRAFT-ACCEPTANCE-RESULT。


> **2026-10-14 研究追加（CHAT-20261014-DRAFT-DATA-VERIFIER-FOLLOWUP）**：历史Long4802行重建、15组CPU训练、6份真实profile、64个CUDA数值案例及6个完整chunk验收已完成；新draft未通过多seed原准入，保留A_off。四处融合数值一致，正式性能待独占；42chunk私有导入入口已封存。累计GPU31065.8421秒/147346 intents，余82.24分钟，自有训练/GPU已退出。详细结果与下一步见 [本轮报告](DRAFT-DATA-VERIFIER-RESULT-20261014.md)。以下原报告与历史状态保留。


> 最新结果（2026-10-14T20:10:22.540+08:00）：[18组draft结构初筛与verifier状态](DRAFT-ARCHITECTURE-RESULT-20261014.md)：Transformer在6/6组公平配对中优于容量相近MLP，但全部候选未过准入，保留A_off；低学习率root中位515/630对527/630，退化集中中sigma root。先核验旧train来源并扩大有效数据。verifier profile代码/CPU预检完成，GPU外来占用未启动，尚无融合原型或新性能收益。本轮0新GPU、余90.83分钟，自有训练/监测已结束；tile及preset保留。Knowledge：CHAT-20261014-DRAFT-ARCHITECTURE-SCREEN-RESULT。下方“尚未训练”等为历史时点。

2026-10-14，Asia/Shanghai。用户提出先确定draft结构，再扩大训练，同时优化Transformer。本页整理本地证据与下一阶段顺序，不是新增实验结果或论文正文。

## 决定

先冻结输入输出及投机树接口，用有限对照选出首版draft结构，再扩大有效训练数据；verifier的profile和最小block原型可以现在推进，不必等最终权重。首版结构是可验证的工程选择，不宣称它是最优结构。

当前缺少的是**当前接口、相同信息和训练协议下的结构对照，以及更广的闭环质量证据**。不宜直接押注Transformer，也不宜把新增训练轮次等同于解决轨迹偏移。

## 已核实的依据

| 事实 | 对安排的影响 |
|---|---|
| 当前A_off为227→51→51→224双候选残差MLP，25,928参数；状态112＋前一direction112＋3个时间量 | 现有输入输出已经明确；更换内部结构不必改变verifier接口 |
| 最近六个MLP续训候选各1000更新、batch512，单核CPU记录约9.52–13.47秒 | 当前主要成本是教师采集和闭环评估；此耗时不能外推新Transformer或扩大数据后的训练 |
| 当前新同查询标签5796条，其中3906训练、1890验证；仅task0/1，训练4个episode、验证2个episode | “扩大训练”应先扩大任务、初态和噪声阶段覆盖；验证集继续留出，最终测试另行保留 |
| 旧context数据有67,277训练行，但与当前bonus同查询标签合同不同 | 不能直接混池；先核对输入信息、查询状态、标签和拆分来源 |
| 旧Phase4 MLP235,744参数、Transformer583,136参数；1800 shadow anchors上两层接受89.61%/88.50%，PhaseA为90.33%/90.67% | 不是容量匹配的当前结构实验，也不是新闭环轨迹，不能据此宣布Transformer胜出 |
| R_GATE_A4离线root通过527→543/630且未丢旧通过样本，但闭环最差步数相对T增13.11%，未通过原质量门 | 局部拟合或接受率改善不能保证成功率、轨迹长度与任务耗时 |

A_off最近一次续训使用833个独立样本；它继承了更早的父权重，833不是全部历史训练数据。当前模型也不能描述为已在新增5796条标签上完成最终训练。

证据：[draft源码](../../../投机采样/scripts/p0_batched_verify.py)、[结构与成本核查](SPECULATION-DRAFT-COST-ANALYSIS-20261013.md)、[原shadow报告](../../../投机采样/results/analysis/p0-d-compare-candidate-shadow-20261008-r1/pooled_candidate_shadow_report.md)、[同查询训练与闭环结果](TRACE-SUCCESS-CONTROL-RESULT-20261014.md)、[CPU训练原始结果](../../../review/OMISSIONS.md)、[按episode拆分代码](../../../review/OMISSIONS.md)。

## 有限结构对照

固定A_off作为部署参照，训练候选限定为三类：当前宽度的两头MLP、与新Transformer容量接近的两头MLP、单层小Transformer。具体宽度和参数量在运行前计算并冻结；不依据实验结果无限扩展搜索。

Transformer以16个动作位置作为tokens，每个位置输入状态和previous direction各7维，加位置及时间信息；输出仍是两组112维direction。把227维整向量作为唯一token无法检验动作位置间attention的价值。首轮不添加视觉、语言或target hidden features；添加新信息应另作消融。

公平对照要求：

1. 同一整episode训练/验证拆分、同一输入信息、训练集计算的归一化、loss、数据顺序和更新预算；记录参数量、计算量和实际训练耗时。相同更新次数只代表相同样本暴露，不代表相同计算成本。
2. 结构主对照从相同初始化策略开始，用预先固定的多个种子；A_off续训另列，不能把其父权重经验当成纯结构优势。少量优化设置预先固定并只用验证集选择；单个不收敛设置不足以否定一类结构。
3. 先看held-out的分task、root/child、sigma误差和旧通过丢失，再检查实际递归提案及合法前缀接受。已有child标签对应旧draft访问状态，新draft到达不同状态时，必须在新查询上补teacher，不能沿用旧状态标签。
4. 仅对入围者做batch1 draft延迟、完整chunk耗时和少量配对完整episode。T和当前A_off保留为双参照，沿用既有质量门，不看结果后改阈值。
5. 三类均不通过时保留A_off，先诊断数据覆盖、提案访问状态和目标函数；不为按期“选结构”强行采用退步候选。少量pilot只能筛选，最终成功率与步数仍需更广评估。

生产loader目前硬编码两头MLP，Transformer需要隔离加载适配与接口验证，不能仅替换checkpoint文件。

## 并行接口和依赖

第一阶段固定：K2/B2、7个逻辑节点/CFG14、两候选direction语义和112有效维、50次action更新、当前bonus与prefix选择、target权重/BF16、CFG2持久history及共享KV布局、短/中/长真实状态基准。

| 阶段 | 算法与数据线 | verifier系统线 |
|---|---|---|
| 现在 | 冻结公平对照协议、数据拆分，比较三类小候选 | 用冻结当前模型做真实CFG14 profile，定位可消除成本 |
| 候选入围 | 少量闭环筛选首版结构，采其实际访问状态的同查询teacher | 独立实现并核验一个block的最小融合 |
| 结构暂定 | 扩大训练分区与覆盖，保持验证/最终测试独立 | 推广全30层verify，做同算法完整chunk精度和正式计时 |
| 两线各自通过 | 更广任务质量评估 | 组合新draft与新verifier，重做数值、路由、质量和耗时验证及单独消融 |

内部结构、权重变化且接口不变时，已有verifier优化通常可以复用；新输入数值与接受分布仍会变化，所以组合验收不能省略。改变树宽/深、CFG批量、target隐藏特征导出或KV布局会改变系统工作负载，应作为新变体重新评估。算法对每chunk验证次数的改变也会影响系统收益，不能把独立加速比直接相乘。

“并行”指研发依赖可分开。单卡正式计时期间不能同时训练或运行其它GPU任务；训练与非计时正确性工作按资源情况安排。

## 已完成的系统线准备与待办

已完成只读数学/布局审查：首个候选是self-attention前和FFN前的两处modulation、以及对应两处gated residual，共四处逐元素融合。先保留原attention、GEMM、norm reduction、FP64 RoPE、table加法及BF16物化边界；FP32乘法与加法分别舍入，不能被FMA改变。scale/shift/gate是每token的非连续view，不能误作token广播或偷偷在计时外复制。工作区复用作为后续独立变体，先验证融合。

这是静态候选，不是已测瓶颈或已实现收益。真实profile、CUDA原型及GPU数值/性能均未执行。详细验收顺序与tile兼容边界沿用[MegaTransformer整合方案](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)。原tile代码/HW保留，新代码仅放独立目录。

## 时间、资源与当前状态

沿用内部目标：10/17前收口首版结构和评测协议；入围后逐步扩数据，并行推进verifier；10/21形成更广质量表；10/26系统原型完成数值与完整chunk验收；10/30冻结主证据，11/01 18:00交老师。以上是研发目标，不是已验证能完成的承诺；如果候选质量未过门，优先质量和现有系统证据。

累计GPU30550.48840228282/36000秒，余5449.51159771718秒（90.83分钟）；本轮0新增GPU计算，尚未启动结构训练、扩大采集或profile。该余额只够有边界的下一轮验证，不能当成覆盖所有任务与消融的大规模实验预算。下一次GPU job必须沿用最后成功job及已计入失败的累计账本。

本轮完成的是证据复核、系统静态审查和并行顺序整理。未更换当前preset、修改受保护state/rules/cycle、删除资料或生成论文正文。
