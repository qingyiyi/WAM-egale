# 当前主张与证据

|主张|OBSERVED|UNKNOWN / 缺口|当前允许表述|
|---|---|---|---|
|整个单层在一个persistent kernel内运行|全部数学含self/cross/FFN/KV；最终ready18/默认9个trace均1compute kernel|不代表30层/全模型/其他形状|可以描述有限实现事实|
|当前完整block对齐原Torch|9底层真实签名×2路线最终输出逐位；可配对中间边界、各18有效KV转移、重复/输入/RNG审计通过|任意输入、未成对保存字段、完整verifier/接受路径及闭环未证|可以写“有限真实输入逐位通过”，不能写普适保证|
|H1小action形状效率限制|实际action Q16、GEMM M32→224；shape与原kernel记录见旧profile|实际achieved吞吐/带宽/occupancy、完整归因不齐|有限结构事实，不笼统说action无法tile|
|H2候选批验证经济性|结构已给出真实多候选工作负载|本轮未做同一冻结7-query逐个B2对B14完整配对|本轮不能宣称已重新证明|
|FFN子图按行组就绪交接|四行组有72/96消费者在全部producer完成前领取；完整归约与可见性保持|精确重叠时长、各开销占比未知；单行组无此空间|可以描述交接发生，不能称已经提速|
|H3超过强B14 baseline|当前默认仍慢native2.68–5.02倍，ready另慢1.2%–6.0%|3状态、cache/Graph最强矩阵未完成，NCU计数缺失|当前实现没有新增收益证据，不能宣称MegaTransformer加速成立|
|H4质量约束下有效成本下降|本轮没有新verifier/chunk/机器人执行|ΣT_round/Σg、完整闭环质量与E2E均未测|UNKNOWN，不从block时间推造g或E2E|
|RoboTwin迁移成功|官方权重/资产/私有依赖已准备|真实原渲染两次非法访存；0policy/teacher，Target未复现|只能报告迁移环境受阻，不能报告模型失败或成功率|

历史LIBERO完整6000局及其成功/长度/成本仍用原报告；训练/DEV曝光、首两S FAILED、长尾及四false保留。普通batch/cache、FA2和LayerNorm上游复用本身不是新贡献。未做interaction矩阵，不声称协同放大效应。
