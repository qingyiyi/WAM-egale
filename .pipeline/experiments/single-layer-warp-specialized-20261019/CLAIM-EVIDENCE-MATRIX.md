# 当前主张与证据边界

|主张|观察证据|缺口/反证|当前允许写法|
|---|---|---|---|
|整层计算在一个persistent kernel内|9真实block的GPU trace各1整层计算launch，全部tile恰好一次；完整CUDA包含self/cross、投影、FFN、Norm/RoPE/残差/KV|仅block0/有限真实输入；不是所有输入的证明|可以限定为本原型的结构结果|
|warp职责分工已运行|warp0调度、1/2搬运、3–6数学、7写回，实际单kernel执行|粗阶段barrier，全局中间量，没有跨阶段异步生产者消费者细流水|可称职责分工原型，不能称完整MegaMoE式高效流水已实现|
|与Torch精度一致|原冻结baseline有限通过；新原型输出9/9逐位FAIL、写KV9数值失败|hidden相对L2 0.13%–0.56%不是物理动作界；没有接受路径/闭环证书|不允许宣称新原型精度通过|
|新原型速度更快|两独立进程270原始样本|耗时为native的5.12–11.28倍；未与更强cached/Graph晋级比较|只能报告负结果与成本诊断|
|共享布局有局部收益|唯一ldm40挑战者匹配v1，2交错样本/格，延迟下降6.5%–19.0%|一进程、带计时副本；未与原native同轮重测；仍原精度FAIL|可写有限诊断信号，不能写最终加速|
|GEMM是该原型B14主要耗时|三B14设备阶段计时加权74.66%，cross K/V投影占49.2%–63.1%|针对未缓存原型；强baseline已有prompt缓存，不能外推全部WAM|可限定归因于该原型/这些输入|
|bank conflict是唯一根因|布局反事实有改善|无NCU bank-conflict计数；register、同步、访存和epilogue代价同时存在|HYPOTHESIS，不能写已证明|
|H1：原小action存在shape效率限制|旧真实aten::addmm M32→224；固定FFN tile几何利用率改变，旧有限profile|不是数学上无法tile；实际带宽/算力/launch因果分解仍不完整|有限机制支持，不写唯一瓶颈|
|H2：同7-query批量验证有经济性|实际B14已合并Linear行；既有投机成本较低|尚缺同一冻结7-query逐B2与集中B14的严格对照；没有3-state矩阵|尚未完成该因果证明|
|H3：候选感知调度超过强B14|旧通用FFN1+GELU有约0.7%–3.2%有限block收益|M32/M33 persistent无正收益；本M34明显负；普通batch/cache不算新增贡献|当前不成立；保留负结果|
|H4：质量约束下有效提交/E2E获益|四套6000完整终态，新隔离子面板测ΣT_round/Σg|质量/首两S/组件门未过，长尾和训练曝光；新kernel未集成|仅报告未准入算法工程成本，不能宣称最终闭环保证|
|算法×系统协同|需要shape×engine及独立因素消融|本轮没有1/3/7完整矩阵及有利交互；新kernel无改善|不允许写协同已成立|

因子命名必须随具体登记：旧文稿A0S0中的A指算法，新网页提案中的A指candidate scheduling、S指cache。两套定义不能混用。本轮只测whole/native成本诊断，没有通过变更字母含义补出任何缺失消融。

原证据：[首批结果](RESULT.md)、[阶段/布局诊断](DIAGNOSTIC-RESULT.md)、[源/资源/止损](RESOURCE-AND-STOP-REPORT.md)。旧强baseline与真实GEMM证据见[前期设备profile](../operator-profile-device-20261019/RESULT.md)。四质量字段false、旧FAILED和default A_off不变。
