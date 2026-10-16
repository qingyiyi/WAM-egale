# 完整模型闭环复现结果：先OFF/OFF，再ON/OFF

日期：2026-10-11T12:46:52+08:00（Asia/Shanghai）。范围：投机采样/action-only，非tile。用户要求定期检查既有6局任务、结束后总结、按OFF/OFF→观察器开关顺序核对并记录HW。本次只等待既有任务和CPU读取已保存结果，不追加GPU采集、训练或公共源码修改。

## 1. 是否跑完与本步完成程度

- 实际状态 **completed**，6/6完整episode，6局成功；启动2026-10-11T12:10:35.560799+08:00，结束2026-10-11T12:41:37.370201+08:00，运行31.03分钟。
- **OFF/OFF复现验证：True；本轮两init ON/OFF验证：True。** 包括全部已采状态/RNG/RGB/执行命令/终止结果，以及保存的完整action和video张量，不只比较最后成功标志。
- 没有额外训练，也没有把观察器成功判定为训练改善。完整隐藏状态快照、旧失败唯一因果、成功率正式非劣和任务端到端加速仍未证明。

## 2. 逐局结果

同一原S0 checkpoint+冻结target/verifier；所有worker首局，client seed20260922/PYTHONHASHSEED0与公共指纹插桩一致，native只在ON开启。各条件两独立EGL worker并行，三个条件串行，完整800 simulator horizon，环境步数排除5稳定化步。

| 条件 | init | success | env steps | policy chunks |
|---|---:|---|---:|---:|
| repeat_off_A | 3 | True | 293 | 19 |
| repeat_off_A | 4 | True | 600 | 38 |
| repeat_off_B | 3 | True | 293 | 19 |
| repeat_off_B | 4 | True | 600 | 38 |
| observer_on | 3 | True | 293 | 19 |
| observer_on | 4 | True | 600 | 38 |

这些条件不是target-only vs候选训练对照；全6成功不是正式非劣结果，也不能说S0或B_on训练已达目标。本次指纹/观察器插桩是诊断，**不使用本轮耗时筛选加速模型**；旧launcher的latency_benchmark标记不能覆盖job的clean_timing=false合同。

## 3. 先看OFF/OFF重复基线

- 原始自动关卡检查两个init初态hash、client RNG、双视角RGB、逐步测量状态、实际命令、reward/done和终止长度一致后，才启动ON两局；若不一致会四局后停止，不自动重跑/加局。
- CPU复核两轮保存的初始20类状态NPZ逐值相等，且与首行hash一致；全部逐步状态/RNG/RGB/命令记录没有第一处分叉。
- 各init对应所有policy chunk的**完整30通道action tensor和video latent逐位相同**，不是仅7个部署动作维度或已执行前缀相同。
- 因此本轮受控重复基线成立。此前正式client种子合同缺口仍是真实风险，但新旧还同时改变PYTHONHASHSEED/公共指纹等条件，**不能把旧分叉或旧失败唯一归因于client seed**。

## 4. 再看观察器ON/OFF

- 两init的ON与OFF-A在完整已测轨迹、动作、观测、终止长度/成功结果及全部保存action/video张量一致；OFF-A和OFF-B已一致，故ON同时与重复基线一致。
- ON的capture包装在每个初始/动作后检查20类sim/controller状态、Python/NumPy RNG及原始RGB读取前后不变，未触发assert；原生行无字段错误，goal合取与native success一致。
- **结论只限本轮两个init和当前S0/target/实现/指纹合同：未观察到native observer影响完整闭环。** 不扩展成所有任务、seed、工作负载或未来版本的一般纯函数证明；全server RNG/cache和全部sim隐藏字段未保存。
- observer特权字段未进入policy输入，没有teacher采集、模型更新、gate/bonus/阈值或scheduler/cache策略更改。

## 5. CPU独立对账与数据完整性

- 6 merged/worker episode一致；2685逐步指纹行连续，timestep=env_step+5。
- 171 policy chunk、2679执行命令，按当前BF16量化动作解码合同逐值重建并匹配指纹；初始化只执行12动作，终止chunk按done真实裁切。
- 2616组保存server动作后双视角RGB逐数组hash匹配；各episode共同terminal NPZ分别验证翻转后RGB与终止指纹。没有把少量terminal快照说成补齐终止chunk所有动作后的缺帧。
- ON原生895行验证连续动作/goal与terminal图像hash；原始native接触只表示接触，In只表示区域包含，不自动变成稳定抓持/释放落稳奖励。
- 114对应chunk对分别比较完整action和video。独立CPU脚本再复核CSV/边界/命令/比较结果，以及721个去重输入/运行/raw/protected hash；模型source manifest及target9文件size/mtime保持合同，不冒充全权重content hash。
- 所有读取只在CPU上运行，没有加载policy或重新执行环境。当前execution.mode=self，主agent另编CPU复核脚本，不称为第三方独立审计。
- 没有锁定、回滚或写入另一线程正在更新的tile研究派生文件；只对本任务已结束runtime和不可变旧raw/protected做检查。Knowledge索引/compact增量写入时保留最新tile条目，不做全量重建。

- 结束后核对本任务12个client/server PID、supervisor与tmux daemon均已退出。另一个已获授权tile任务随后接手GPU；不将其显存占用误报为本任务未结束，也不停止或计入投机采样工作。详见cleanup.json。

## 6. 假设排序及结果

| 假设 | 证据/反证 | 当前结论 |
|---|---|---|
| 固定task/init仍可能有未控起点 | 前步查明client seed缺口、flattened init非完整snapshot；本轮统一seed/hash/指纹后重复成立 | 新受控合同有效，未唯一解释历史差异 |
| observer直接扰动执行 | 固定动作回放与本轮ON/OFF均未出现已测分叉；capture前后检查通过 | 当前测试范围不支持扰动，不做全域证明 |
| 模型/缓存/数值使同起点反复漂移 | OFF/OFF全部保存action/video输出一致 | 当前两init未观察到漂移；内部完整状态未取证 |
| 全部成功意味着训练已修复 | 本轮只有同一S0，非候选训练对照；没有旧失败反事实 | 不支持训练改善或候选晋级 |
| 长轨迹只是观察器造成 | init4在两个OFF条件与ON均同样600步 | 该600步长轨迹不由开关差异解释，可作为后续诊断对象 |

## 7. 定期检查实际执行

本轮持续状态轮询，计划约2分钟间隔，工具/处理耗时会使实际检查间隔更长；期间没有自动追加实验。查询的当前工具列表没有可用automation_update/常驻调度接口，因此**没有虚构创建离线定时推送**；实际保持本轮跟踪至任务终止，现不再轮询已结束任务。

- 12:25:33：running_repeat_off_B，完成2/6。
- 12:30:16：running_repeat_off_B，完成3/6。
- 12:33:27：running_observer_on，完成4/6。
- 12:36:16：running_observer_on，完成4/6。
- 12:39:12：running_observer_on，完成5/6。
- 12:43:07：completed，完成6/6。

服务器supervisor本身独立tmux运行、自动检查完成/失败并写结果，HW自动结果链接随完成报告更新；本报告是在任务结束后CPU审核生成，不将自动报告时间等同人工核验时间。

## 8. 下一步（尚未启动）

1. **复现/观察器关卡本轮通过，可以进入成本账本。** 单独设计同phase/同并发需求的target-only T与A_off成本/target工作计数，用同一新seed/初态合同，原target/验收协议固定；不拿本次诊断耗时作clean性能结论。
2. **物理/训练线：优先CPU检查可复现的init4 600步轨迹**，对齐contact/pose/In与保存RGB，区分返工、重抓或阶段停滞；现有600步不是短轨迹目标已达成。若需新target-only反事实或新GPU评测，另行确认预算，不自动启动。
3. 再决定冻结target下任务相关反馈/困难状态监督的小模型训练。L2仍为局部辅助，不改变验收阈值；不因为观察器通过就直接跳到RL/KL或大小模型联训。

## 9. 证据

- [独立对账](../../review/OMISSIONS.md)、[汇总](../../review/OMISSIONS.md)、[逐局](../../review/OMISSIONS.md)
- [OFF/OFF与ON/OFF完整比较](../../review/OMISSIONS.md)、[完整张量比较](../../review/OMISSIONS.md)
- [实际动作](../../review/OMISSIONS.md)、[RGB索引](../../review/OMISSIONS.md)、[原生事件区间](../../review/OMISSIONS.md)
- [轮询记录](../../review/OMISSIONS.md)、[原始自动报告](../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/FINAL_REPORT.md)、[原始运行状态](../../投机采样/results/runtime/p0-repro-observer-task0-6ep-20261011-120724/status.json)
- [持续研究入口](../research/speculative-sampling/RESEARCH.md)
