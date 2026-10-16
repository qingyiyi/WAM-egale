# 短KV优化：候选行映射证明复用CPU阶段结果

> 2026-10-12T22:49:30+08:00 | 投机采样 | zh-CN | Asia/Shanghai

CPU原型与21项检查已完成。实际CPU profiler确认，30层候选行检查范围内的标量取值从30次降到1次；原共享S0与优化S1的CPU输出逐字节一致。**尚未验证A100上的耗时收益，也没有新的训练目标GPU逐位结果**。本轮新GPU启动/target前向/直接attention均为0，原短档mixed及中长档正向结果保持不变。

## 本轮执行与监控

用户在上一轮下一步建议后指示“那你监控着，做完了给我汇报结果”。检查确认上一轮唯一GPU worker PID2810292已退出，最新GPU任务仍为21:48:41完成的两档job，没有本会话正在运行的新GPU测试。当前代理执行并跟踪新的私有CPU阶段；21项检查在22:34:33完成，耗时18.195秒，0errors/0failures。没有创建无人看守的GPU队列或后台任务。

新增私有目录为[shared_prefix_rowproof_cpu_v1](../../../投机采样/experiments/shared_prefix_rowproof_cpu_v1/PLAN.md)。公共模型/server/config及受保护state/rules/cycle不改，旧失败、旧原始计时和所有原CUDA/参考实现保留。原型默认关闭，`row_proof_enabled`需显式bool true；`enabled`和`allow_cuda`仍独立默认false，旧环境变量不启用。R/HC走原检查，S0为原共享路径，S1为新证明复用。

## 唯一实现改动

原共享adapter在同一lease的30层各执行一次`bool(((rows>=0)&(rows<2)).all())`，GPU情况下包含取值和同步。新原型保留第一次实际张量值域检查，再依据相同lease/request/generation/epoch、固定CFG2，以及同一行张量的身份/地址/版本/shape/stride/device/dtype复用证明；各层shape/device/dtype继续检查。

`_precheck`仍在每层投影前后执行，五个历史张量key、私有slots版本、模型状态/RNG、原共享attention数学及CUDA v2均原样继承。证明是固定大小的主机元数据；lease结束、异常、缓存失效或模型/行映射变化时失效，没有新增history14/fullKV14或GPU索引缓存。没有改动文本prompt投影/cross-attention。

该证明沿用既有Torch版本计数的失效边界；CPU检查覆盖可跟踪的写入、写入后恢复值及投影hook期间写入。不据此宣称能检测绕过Torch版本计数的任意底层存储写入。

## CPU结果

| 检查 | 结果 |
|---|---|
| 实际生产loader／完整30层／3个变化query | S0/S1全部20,160元素逐字节一致 |
| 行检查scope内实际`aten::_local_scalar_dense` | S0 30次；S1 1次 |
| 动态值域检查／证明复用 | S0 30／0；S1 1／29 |
| 历史272／1136／2144的接口测试 | 三档S0/S1 CPU输出一致；各使用2层小CPU模型 |
| 原R／HC路径 | 保持原CPU Torch输出逐位一致 |
| 历史状态／RNG／零拷贝CFG2／lease释放 | 通过 |
| 行写入／恢复值／换同值对象／投影hook突变 | 拒绝旧lease或证明 |
| K/V/mask/id/is_pred写入／私有slots写入 | 拒绝旧lease或证明 |
| 模型权重／train／backend／cache_mode变化 | 拒绝旧lease或证明 |
| 外部恢复异常／新generation／线程／CUDA边界 | 通过 |
| 当前候选其他13行／CFG另一组7行隔离 | 通过 |

独立CPU脚本另用实际生产loader核验3份变化输入，保存[S0/S1实际输出](../../../review/OMISSIONS.md)，重新读取后校验全部字节。原/新5个私有实现文件以及2544个旧源/证据SHA一致，状态、历史/RNG及本轮0GPU计数通过。

这里使用随机小CPU检查点，验证接口、证明有效性及新旧共享数学路径的一致性。不能把CPU标量次数的变化解释为GPU整体耗时下降96.7%，不能用这些CPU输出替代训练目标的原Torch GPU逐位门，短KV稳定加速仍未知。

## 为什么下一轮比较三臂

仅将优化S1和上一轮S0耗时比较，会混入运行时间及负载变化。因此拟定同一真实短档KV288／完整30层／全部27自然query的R原Torch、S0原共享、S1优化版三臂对照。

| 轮 | 顺序 |
|---|---|
| 1 | R→S0→S1 |
| 2 | R→S1→S0 |
| 3 | S0→R→S1 |
| 4 | S0→S1→R |
| 5 | S1→R→S0 |
| 6 | S1→S0→R |

每臂恰好两次位于每个位置，任意两臂的先后各三次；每臂完整27query预热。正式OFF不加profiler，不剔除样本；staging、lease/setup、keys、持久journal及release仍在原计时内。分别报告R/S0、R/S1和S0/S1，各保留六组及≤7%MAD门。与R更快和比同轮S0更快是不同结论；优化若没有收益须如实保留。

## 预算与当前执行状态

GPU设计为capture50＋resource probe1＋三臂完整预热81＋六轮三臂486＋条件诊断最多3＋三臂收尾3＝**最多624次target**；1worker/1GPU启动/无重试，拟定含加载/IO/同步/清理600秒。所有输出仍须逐位原Torch，all30状态/RNG、lease及资源门保留。没有完整policy chunk、EGL、episode或训练。

当前累计仍1865/2200次、2007.1380846686661/2700秒，余335次及692.8619153313339秒。此设计最多累计2489次、2607.138084668666秒，若后续批准累计target至2500，则GPU仍可保持45分钟。

**这只是GPU设计，GPU执行包装及其CPU预检尚未实现，未生成新的授权回执，未排队或启动GPU。** 21项PASS来自行证明原型，不代表未来GPU包装已通过CPU预检。现有2200次回执不能自动扩限或覆盖此新实验。

## 下一步与进展边界

先准备默认关闭的短档R/S0/S1三臂GPU执行包装并完成其CPU计数/计时/恢复/预算拒绝门；全部通过后，再以绑定新plan哈希的明确回执确认累计target2200→2500，GPU累计仍2700秒。随后执行同条件六组，保留全部query预热、原Torch逐位门和全部样本。当前不自动GPU、prompt共享或真实任务闭环。

本阶段状态：`CPU_ROWPROOF_PASS_GPU_RUNNER_AND_BENEFIT_PENDING`。短档稳定收益仍未确认；中长档性能不改写；文本prompt共享、真实成功率/接受率/E2E仍待后续独立验证。执行和实现审查为同一主代理，独立CPU审计脚本不称为独立代理复核。

## 复查入口

- [21项CPU结果](../../../review/OMISSIONS.md)、[测试日志](../../../review/OMISSIONS.md)、[独立CPU字节/来源/预算审计](../../../review/OMISSIONS.md)。
- [CPU原型](../../../review/OMISSIONS.md)、[具体方案](../../../投机采样/experiments/shared_prefix_rowproof_cpu_v1/PLAN.md)、[三臂GPU设计](../../../review/OMISSIONS.md)、[旧来源冻结](../../../review/OMISSIONS.md)。
- [前轮短中长结果](SHARED-PREFIX-SHORT-MIDDLE-RESULT-20261012.md)、[前轮最终复查](../../../review/OMISSIONS.md)。
