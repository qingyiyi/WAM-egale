# 系统证据结果

状态：M34有界原型与诊断完成、原数学/性能NO-GO；M35 CLOSED_PARTIAL_PRE_REGISTERED_PANEL，330/360。完整原数据分别在[M34](../single-layer-warp-specialized-20261019/RESULT.md)、[M35](ISOLATED-RESULT.md)。

|诊断候选状态数|CFG/Q/线性M|本轮whole与Torch/native|缓存/强baseline状态|
|---|---|---|---|
|1|2/16/32|短中长完整block已测|本轮uncached；旧强cached/Graph有限证据另列|
|3|2/16/96|NOT_RUN，当前whole入口未准入B6|没有补造1/3/7完整矩阵|
|7|2/16/224|短中长完整block已测，仍7候选各Q16|本轮uncached；不是Q112跨候选序列|

另有B2 video Q128×三历史用于完整单层兼容验证。9 trace各1计算kernel；新原型相对native耗时5.12–11.28倍，原hidden9/9失败。175-register计时副本→198-register ldm40副本局部延迟下降6.5%–19.0%，两者理论均1 CTA/SM，仍不是原Torch精度通过。[阶段/资源](../single-layer-warp-specialized-20261019/DIAGNOSTIC-RESULT.md)、[实现问题与原语metadata](../single-layer-warp-specialized-20261019/IMPLEMENTATION-ISSUES.md)。

实际主函数172 registers/thread、59616B shared、108 CTA、LOCAL0；实测L2 requests/DRAM bytes/bank conflicts/achieved occupancy UNKNOWN。已有G1/G2或强cache的旧计数不能移填本新kernel；旧30层或tile allclose也不能继承。单kernel结构不是速度主张。

M35核心指标为ΣT_round/Σg；完整policy/chunk、全体暖episode及双方成功补充分别列[计时表](ISOLATED-RESULT.md)，[成本分解](COST-LEDGER.md)。这些作业未安装新whole kernel，其收益属于原投机实现。原观测器仍有首两S/组件FAILED；暖计时含审计/RPC/环境，不是无审计生产cold E2E。没有用target calls或接受率代替成本。

H1有限支持；H2仍缺同一冻结7-query逐B2/集中B14；H3新调度无正收益；H4在质量约束下的闭环收益未建立。A0S0/A0S1/A1S0/A1S1及1/3/7形状交互未完整执行，不计算缺失R，不声称协同。

算法侧还缺少步数Target25/17和匹配MLP等质量—成本强对照；目前不能宣称投机优于直接少步数。这些是论文缺口，不是当前GPU队列中已启动的工作。
