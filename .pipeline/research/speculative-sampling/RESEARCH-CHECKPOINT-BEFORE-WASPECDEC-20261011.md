# WA-SpecDec调研前的投机采样检查点

记录时间：2026-10-11 17:10:27（Asia/Shanghai）
范围：LingBot-VA action连续去噪投机采样；与tile算子研究完全隔离。
性质：保存调研之前的事实和计划；不是新实验授权，历史状态不追溯改写。

## 已完成工作及阶段性成果

1. 已打通draft→真实target/verifier→accept/rollback，历史稳态单chunk profile的action 4.4360→2.8211秒、policy 6.2772→4.6559秒（下降25.83%）。不同campaign不能混算为全局E2E收益。
2. 发现较早四-init对照env steps 276.5→314.25，E2E只快0.34%；随后六-init描述池E2E慢0.89%、steps多9.37%。小样本不证明成功率非劣。L2仅是局部接受指标，不是任务进度或环境步数的充分代理。
3. 最小冻结target对照已完成20完整episode：4采集、16开发评测，A_off离线更新和B_on固定S0真实访问状态上的teacher监督各1000更新×512，同预算、不更新target。T/A_off各4/4成功，S0/B_on各3/4；B_on不是RL或自身访问分布多轮DAgger。A_off少27总步但总chunk仍72，平均E2E慢9.67%，未晋级。见RESEARCH第13–14节。
4. 补齐默认关闭观察器、原生contact/pose/In/夹爪状态、终止记录、种子及逐步状态/RNG/双RGB/命令指纹。固定动作回放、同S0的OFF/OFF/ON六局全输出复现通过；受控T/S0四局均成功，init3同293步，init4为T300/S0600步，主要返工在第二罐建立抓取。不能概括所有初态都退化。
5. T/A_off成本包装CPU预检及真实六局不扰动验收通过。T为293/300步（19/19chunks），A_off OFF为283/291步（18/19），ON精确匹配各自OFF的完整action/video及已测轨迹。本组A_off总步少19、总chunk只少1；未证明全hidden一致、自然E2E或跨任务非劣。
6. 现有成本CPU账本：32对matched请求、覆盖64/75（85.33%），仅2对初态数值全同。action均值7.759→9.771秒（+25.93%），server policy +18.60%。短KV144–720段action −10.14%，增长864–2016段 +24.81%，长KV2160段 +82.83%；verifier调用成本0.186→0.401秒，depth2/tree 71.97%→34.03%。target action调用减少40.38%，但CFG行调用约3.86倍（不是FLOPs）。长KV、接受深度、任务阶段和并发负载尚不能唯一归因。
7. CPU完成短/长KV×单/双需求固定负载规划，旧84产物/440 raw帧及8规划合同检查通过；这不是新snapshot/replay工程CPU或GPU恢复通过。

## 当前正在做什么／做到哪一步

当前停在固定负载方案等待预算与任务专用实现范围确认；未实现新重放包装，未运行新的GPU fixture或成本四格。调研WA-SpecDec作为旁支，不替代现有成本关。

候选预算（尚未批准）：0新完整episode，最多2次双EGL初态初始化；4真实历史fixture（init3/4×KV288/2160）×单/双需求，ABBA+BAAB；96主测+48预热/参考+24诊断=168固定chunk上限；GPU35–55分钟估计、建议90分钟硬cap，CPU工程及复核另计，总90–165分钟非保证。

## 原未来计划与关卡

1. 用户确认预算及仅任务专用默认关闭snapshot/replay/监督器/CPU比较器范围。
2. CPU工程预检（不是已完成8规划测试）→新鲜GPU/EGL/端口/资源门→同算法fixture恢复及全输出/有效cache/RNG精确复现。
3. profile OFF固定负载成本主测；profile ON分解另列；核查单双需求/短长历史下真实verifier成本、构树开销、接受收益。失败不补局、不改容忍，不自动旧8局或训练。
4. 据成本结果决定保留K2B2、缩树/降低验证行负担或成本感知启停的后续最小实验；任何方法变更须另确认。
5. 有成本空间后才做冻结target、真实访问状态/困难阶段的draft训练，任务成功率、配对步数、长尾和自然E2E共同晋级；不再只追L2/接受率。RL/KL、大小模型联训后置，尚未授权。
6. 最终多任务/多init、预登记非劣与失败计费的自然E2E评估。

## 证据入口

- [研究历史](RESEARCH.md)，尤其第13–14、18、21、29–32节。
- [六局不扰动结果](GPU-NONPERTURB-SIX-RESULT-20261011.md)
- [成本CPU读数](COST-LEDGER-CPU-READOUT-20261011.md)
- [未批准固定负载方案](FIXED-LOAD-COST-PLAN-20261011.md)

本次checkpoint写入不启动GPU/EGL/服务/episode/训练，不改源码、权重、阈值、cache协议、protected state/rules/cycle；旧证据与tile不变。用户本次另授权文献检索和HW记录，未授权据论文重构或新GPU预算。
