# 投机收益为何有限、draft结构与reset/其他开销分析

2026-10-13T17:30:17+08:00（Asia/Shanghai）。HW explain只读取证，另走chat追加研究记录。0新完整episode、0target、0GPU；CPU载入实际A_off权重与原helper做36组消费关系检查，CUDA未初始化。公共实现不改。

## 结论与消融分母

纯direction投机D未证明完整任务稳定收益：原Torch/cache保留时，两个任务合计T 720.813→D 728.401秒，增加1.053%。主要新增成本在policy，reset没有增加。共享KV与O1/O2/O3能减少这套投机实现的开销；当前相对原T历史10.890%属于整套组合，不能称为纯投机贡献，也尚无“相同当前共享/工程基线仅开关投机”的同轮消融。

| 历史两episode累计 | 原T target-only | D direction投机/原Torch | S：D＋共享历史KV |
|---|---:|---:|---:|
| 完整任务秒 | 720.813 | 728.401 | 666.360 |
| reset RPC秒 | 233.115 | 230.195 | 221.836 |
| policy RPC秒 | 428.068 | 439.521 | 388.357 |
| 观察/KV RPC秒 | 20.818 | 19.785 | 18.237 |
| 客户端余项秒 | 38.812 | 38.900 | 37.930 |
| 环境步数/chunks | 593/38 | 574/37 | 574/37 |

各版2/2成功，libero_10/task0、初态3/4。T从v1复用已完成存档，D/S在v3完成，各臂双独立EGL worker同卡、分批；同checkpoint/seed/config但不是随机化多任务因果结论。S增加共享之后观察到D→S降8.517%；新O1/O2/O3另轮对现测S0降9.172%。不同批次S0时间波动明显，不能把这些比例连乘。后续固定原baseline＋上一版＋候选同轮计时协议保持，并保留原D消融说明算法贡献。

## 1. target调用少但每次更重

T每个action普通forward为CFG batch2。K2B2展开root＋2个中间节点＋4个叶节点，一次验证CFG batch14；当前bonus关闭，每树实际前进1或2次scheduler更新，并非通常3步。两任务action target forward 1938→1122（-42.1%），但CFG行调用3876→14556，为3.755倍。行数是工作规模，不是精确FLOPs或wall time；批量摊销与并行仍可能有净收益，但不能只看调用次数。

共1026树、3078次draft前向、6626次树scheduler transition，仅提交1850次实际更新；很多分支最终弃用。50次内部去噪更新没有删除。原T38chunks、D37chunks只少一个；T平均完整policy/chunk 11.265秒，D 11.879秒（+5.451%），所以轨迹短3.20%仍不能抵消每chunk贵。policy累计增加11.453秒，reset少2.920秒、cache少1.033秒，整局净增7.587秒。

短/中/长KV共享实验的分母是相同batch14原Torch verifier D，证明省下历史KV展开/复制成本；不是batch14投机比原batch2 target-only都快。共享没有同时消除各候选当前QKV、FFN、cross-attention、video/VAE、CPU打包/同步。旧同状态OFF动作分支：短KV单需求约快31%、长KV单需求慢4.7–11.2%；同卡双需求长KV慢28–30%。这是旧未共享D，不能描述为当前共享O3也同样慢；双worker反映采集负载，不把它充作用户单需求部署结果。

原成本诊断ON的verifier占action约69–84%，draft约2.2–4.7%，只是受观察器影响的线索，不能给当前E2E精确拆秒。当前首边接受74.561%、两边全接受59.942%，draft提交占74.595%；平均每树前进1.7456步。接受第二边与“第一边接受、第二边用target纠正”当前都前进2步，所以提高全接受比例不自动按比例减少验证次数；优先优化成本而非仅追接受率。

## 2. 额外发现：叶节点target预取未消费

[score_k2_b2_tree](../../../投机采样/scripts/p0_batched_verify.py)明确仅用target第0行计算首边误差、第1/2行计算二层误差；第3–6行叶target留给prefetch。全接受执行返回所选叶target作为prefetch，见[execute](../../../投机采样/scripts/p0_batched_verify.py)。[当前在线循环](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)仅在bonus=True时用它第三次target更新；实际mode=verifier_k2_b2、bonus=False时提交leaf_state/第二个draft方向并step_index+=2，下一棵树不消费prefetch。4个叶状态本身仍用于选路径/提交，不能把“无需计算叶target输出”误解为“可删除叶状态或第二层draft”。

使用原build/score/select/execute与实际A_off CPU predictor：3seed×FP32/BF16×depth0/1/2×global/greedy=36组，保持根/中间target、树状态不变，仅改变4个叶target，误差、选择、accepted_depth、提交状态及next_previous_direction完全相同；全接受时只有prefetch改变。确定性小scheduler验证消费关系，不是完整生产KV或target模型等价测试。

这意味着当前14行中8个CFG叶行的target结果未用于在线接受/后续状态，值得设计3逻辑target节点/CFG6行验证；7个draft状态仍可保留。按既有调用账本同样树数的行数算术14556→6348，但**不是已实现，更不代表GPU时间按比例下降**。实际缩batch可能改变BF16内核/数值；现SharedPrefix/rowproof/buffer硬编码7/14，必须同时适配并验证输出、接受路径、KV及完整任务。不能直接删4行当成已安全优化。

另一候选是利用已算好的叶target启用bonus；它改变推进与后续draft输入/轨迹，需要独立质量/成本验证。当前用户只要求解释，本轮不改bonus或验证树。

## 3. 当前draft模型的实际结构

冻结[A_off.pt](../../../review/OMISSIONS.md)，SHA d55f59c3b467034d21bc21895136bb797c03a752d6934f854ac5e0ee7e031e7c；实际权重shape与loader一致，25928可训练参数，FP32，约101KiB参数存储。

输入227维 = 当前7×4×4=112维动作去噪状态＋前一条实际使用的112维direction＋sigma、next_sigma、timestep/1000三标量。使用冻结feature均值/标准差归一化。

`227 → Linear(227,51) → SiLU → Linear(51,51) → SiLU → Linear(51,224)`

最后224拆为center112与delta112，生成两候选残差center+delta/center-delta；反归一化后加上一direction，返回[2,7,4,4]并转回target状态dtype。见[MLP](../../../投机采样/scripts/p0_batched_verify.py)、[predict输入/解码](../../../投机采样/scripts/p0_batched_verify.py)。direction是flow/diffusion内部更新方向，不是直接预测最终可执行机器人动作。

这是小型双候选自回归残差MLP，没有attention层、视觉/text encoder或自己的历史KV；没有直接输入图像、prompt或Transformer隐藏特征，只通过当前动作状态及上一direction间接获得条件信息。首个去噪更新需要原target得到初始direction；每树root调用一次draft，再对两个中间节点各调用一次，共3次draft forward。

它来自离线teacher方向监督。最近minimal continuation在冻结父权重上使用833个独立A_off样本、1000updates/batch512、学习率1e-4、seed20260922，固定最后一更新；未更新target，不是RL或联合训练。此前训练是否接触评估初态未排除，不能据此声明完全未见任务泛化。结构小且无显式世界条件可能限制proposal适应性，但现证据未证明这是主成本/低收益唯一原因。

## 4. 投机与reset以及其他阶段

**无直接新增reset计算路径。** [infer reset分支](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)仍只调用共同_reset并返回。_reset按videoCFG创建原batch2空cache、清时序/streaming VAE状态、mask/归一化、CPU T5编码正/空负prompt及目录/empty_cache；没有build tree、draft forward或batch14 verifier，也不为投机另编码prompt。shared适配器只包action helper/packer，lease在action finally关闭；O1/O2/O3同样包action，没有修改_reset/encode_prompt。10条已存episode的reset请求账本target forward均0。

T/D/S reset合计233.115/230.195/221.836秒；纯投机D并没有变慢。最新S0/O3合计280.985/280.727秒，也相近。此前原T→当前跨批+47.611秒，只证明批次reset变慢，不能归因于投机。CPU T5、线程竞争、两worker启动/请求交叠、CPU调度、GPU分配/同步等都可能间接影响RPC，未独立拆分，不能断言绝无间接影响或指定唯一原因。

| 部分 | 实现影响与观测 |
|---|---|
| action verifier | 直接增加候选target工作、分支scheduler、打包、评分/选择、finite检查和CPU/GPU同步；共享与O1–O3仅抵消其中部分 |
| draft准备 | [lazy loader](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)在首次action helper读取小checkpoint；predict首次把model移到GPU，属于policy/首chunk，不属于reset；norm统计仍每predict执行.to，GPU finite→Python bool产生同步，其精确当前时间未单独计 |
| 共享安装 | 旧每chunk重新构造Python AST约一秒级，发生action入口；O1模板复用已减少，不是reset编译或每步CUDA编译 |
| video/VAE | 原video20步/编码路径仍运行，投机只针对action；动作/观察不同可能间接改变后续输入/轨迹 |
| 观察/KV更新 | 原更新接口、持久CFG2布局保持；历史D比T减少约1.03秒，没有账面增时 |
| 环境/客户端/RPC | 原调用与horizon不变；不同动作影响步数。历史客户端余项只增加0.088秒；policy客户端/服务器账面差只差约0.019秒，不支持“网络耗时抵消收益” |
| 服务启动/加载 | 进程模板准备/模型bootstrap/任务后async drain另记，不计主任务E2E；不能把它们与reset混算 |

最新policy改进17.365%，旧policy仅占整局51.308%；当前reset约43.705%且没优化，所以完整E2E只降9.172%。但纯投机D的问题是policy本身增加，并非有一大块reset吞掉本已存在的policy净收益。这两个问题需要分别处理。

## 建议次序与未证明项

优先做有效target验证节点/叶prefetch的CPU最小语义方案与成本账本，再在原短/中KV完整chunk计划中固定原T＋上一版＋候选同状态验证；需要时保留D消融，分开算法/共享/主机优化。3-row或bonus只是建议，本轮未实现、无新增GPU预算或测量。然后对reset正/负T5、缓存clear/alloc与RPC拆分；prompt embedding缓存应三臂共同使用，不能算作投机专属收益。先降验证成本，再判断更强draft是否值得训练；轨迹控制原计划保留。

未证明：当前组合投机开关的同轮因果贡献、3-row真实BF16 target/KV/完整任务等价和净性能、reset独立T5秒数、跨任务稳定质量与收益。本轮已证明消费关系与实际checkpoint结构，重算已有10条任务账本，不追加GPU。

[CPU消费检查](../../../review/OMISSIONS.md)；[完整复算](../../../review/OMISSIONS.md)；[既有E2E归因](SPECULATIVE-E2E-ATTRIBUTION-20261013.md)。
