# 短KV三组GPU对照：准备阶段结果

> 2026-10-12T23:18:11+08:00 | zh-CN | Asia/Shanghai

CPU原型21项检查通过后，新GPU执行包装的16项CPU集成预检也通过，独立字节、来源、计数及预算审计PASS。**GPU尚未启动，S1的训练目标GPU精度和耗时收益尚未验证。** 上一轮GPU任务已结束，本阶段监控的是新私有准备工作，没有在运行的本实验GPU任务。

## 已完成工作

原型保留一次实际候选行值域检查，以相同owned lease、行张量身份/版本和CFG2复用后续层的值域证明。CPU实际行检查scope scalar次数S0为30、S1为1；完整30层3个变化输入的20,160个元素逐字节一致。投影前后history precheck、版本失效、共享attention数学与CUDA v2保留。这些计数不能解释为GPU整体加速比。

新包装用同次运行的R原Torch、S0现有共享、S1启用证明复用做三组对照。恢复真实短档KV288，全部30层和全部27个自然query；三组完整预热后执行全部六种排列，每组各位置两次、任意两组先后各三次。直接继承原Runtime.sequence，staging、首次建立、keys、fsync journal和release保留在计时内。

预热时额外记录S1首query全部30层Q/K/V与attention，核对同次原Torch首query，正式测量前移除记录。GPU实际所有输出仍须逐位等于原Torch，不沿用CPU容差。正式18份保存文件共486个query输出全部复查；若某对不稳定positive，最多追加R/S0/S1首query各一次profiler解释，不删样、不重跑。

16项新CPU预检已验证真实快照/来源、实际30层三组调用、首次attention记录、原计时方法、输出保存、恢复、target意图保留、上限/超时、三组统计、查询或精度漂移拒绝以及无预算回执时在NVML前拒绝。独立审计复查S0/S1实际输出全部字节、CLOSE journal的12次CPU实际前向及所有冻结源码。CPU写日志进程当时存活、审计时已退出，持久ledger/校验链相同；初审将存活字段作静态比较的失败日志已保留，修正审计后通过。

新增失效机制有21项CPU检查；本轮不声称重新进行过新增GPU故障注入或完成生产接入验收。公共源码和受保护HW状态未改；旧失败与旧原始数据保留。

## 当前预算与唯一待决策项

本轮最多capture50＋resource probe1＋三组预热81＋正式486＋条件诊断3＋收尾3＝624次target；1次GPU启动、1个worker、无重试，包含加载/IO/同步/清理最多600秒。

当前累计1865/2200次、2007.1380846686661/2700秒，只余335次前向。完整方案最坏累计2489次、2607.138084668666秒，仍在GPU45分钟内。因此需确认**累计前向上限2200→2500**，并写入绑定[新job plan](../../../投机采样/results/runtime/p0-shared-prefix-rowproof-perf-v1-20261012-231215/plan.json)哈希的回执。旧2200回执不能自动覆盖新方案。没有确认前不启动、排队或创建GPU尝试。

## 前序已完成的GPU结果

| KV总长 | 原Torch中位秒 | 共享中位秒 | 中位耗时下降 | 轮次胜出 |
|---|---:|---:|---:|---:|
| 短288 | 3.11438 | 2.74855 | 11.75% | 5/6，mixed |
| 中1152 | 3.83437 | 3.29048 | 14.18% | 6/6，positive pilot |
| 长2160 | 5.17056 | 3.35975 | 35.02% | 6/6，positive pilot |

短档首组回退约5.56%，所以未确认短档稳定收益。两档新job660个正式输出共4,435,200个元素逐位一致；长档旧父监督器失败保持，完成测量已独立核验。各档query数27/28/29不同，不能把跨档总延迟差全部归因于KV长度，也不能声称所有长度均稳定加速。

文本prompt投影/cross-attention仍按候选批次计算；目前只共享两行CFG历史self-attention KV。真实任务接受率、成功率和端到端加速没有被这些verifier重放证明。本阶段无EGL、episode、完整policy chunk、训练或新环境动作。

## 下一步

收到新预算明确确认后，保存绑定新plan哈希的回执，执行唯一GPU worker并监控计数、逐位结果、资源和清理；结束后独立CPU复算所有18组原始数据及486份输出，并报告S1对同次S0及原Torch的实际效果。无确认不得启动；短KV稳定优化、prompt共享及真实任务E2E仍待验证。

复查：[16项CPU门](../../../投机采样/results/runtime/p0-shared-prefix-rowproof-perf-v1-20261012-231215/preflight/CPU-GATE.json)、[独立准备审计](../../../review/OMISSIONS.md)、[可执行方案](../../../投机采样/experiments/shared_prefix_rowproof_perf_v1/PLAN.md)、[21项CPU原型报告](SHARED-PREFIX-ROWPROOF-CPU-RESULT-20261012.md)、[前序GPU报告](SHARED-PREFIX-SHORT-MIDDLE-RESULT-20261012.md)。
