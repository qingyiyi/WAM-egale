# TF 收口前的 teacher 数值诊断结果

固定基线：`b0dfac3a02f65de33a1217246b83003615166640`。本阶段已完成并清理；**TF64 固定为当前主 draft，当前权重/训练配方仍未验收**。本文件是实验报告，不是论文初稿或质量证书。

## 已观察

1. 原 task4/init26 的 T 与 TF repeat0 各一个完整首 policy chunk、各50次合法 scheduler 提交、actions、视频 latent 与解码物理动作全部逐位重现。它们是已有记录重放，**不是新完整episode**，不补齐M11未完成的第二TF。
2. 首次 teacher 分歧 query（scheduler索引1）两臂全部模型输入字段、30层完整有效KV及保存metadata、实际CPU/CUDA RNG一致。
3. 同一冻结query：B2普通与B2只读逐位一致；B14原生与已有shared-prefix逐位一致，且重现真实TF输出。这个case没有发现由共享prefix新增的方向差异，不能据此概括所有query。
4. 首处子算子差异为`condition_embedder_action.time_embedder.linear_2`。其实际输出FP32，max差`2.682209014892578e-7`；随后时间编码转BF16。不能将首处差异错误写成BF16 GEMM，或凭此声称已定位CUDA kernel的具体归约机制。
5. 原始根条件方向112维中65维不同，max`0.015625`。只替换B14根时间编码，以及把全部候选时间编码逐对按B2计算，两项反事实均未恢复最终方向逐位一致：112维中72维不同、max仍`0.015625`。替换后的时间编码确实与B2一致，但第一个block输出仍不同。**单改时间编码不足以解决完整模型的形状相关数值差异**，不集成所谓修复。
6. 本query实际根深度0；原GPU首边分数约`0.0259571/0.0493239`，均超过原阈值`0.007839680526450805`。CPU按BF16方向先分别转FP32再相减，重算B14两分数与保存GPU值在此例一致；B2方向下两分数约`0.0262596/0.0491447`。这是本例离线诊断，不是重新校准接受器或更改阈值。

## 边界与执行决定

这是对已知首query的数值来源定位，**没有证明整局task4失败的因果贡献，也没有证明TF结构不可用**。投机本身是近似方法，后续不把B2/B14全模型逐位一致当成TF质量的新增验收前提，不继续围绕此问题展开大型算子重写。

用户进一步明确TF64为主模型：暂缓此前建议的新增MLP风险对照；MLP只保留为最终论文baseline/ablation。当前资源应优先用于TF数据覆盖、递归监督与闭环效果诊断，再按证据触发唯一一次新FIT/CAL和TF-cont/TF-on等预算比较。不得将“发现batch数值差异”直接写成“已证明递归分布偏移、允许训练”。现有日志中具体分布/监督缺口仍需核验。

四权限、原retention、阈值、旧FAILED及全局A_off全部保留。无120、独立TEST、正式性能、新架构或新GPU租用。是否通过质量仍按原成功数、matched MLP和长度门决定；结构确定不等于结果预定为PASS。

## 资源与终态

| Attempt | 实际监督作业墙钟 | target forward | 终态 |
|---|---:|---:|---|
| r1 两完整记录chunk及四路径query对照 | 211.860秒 | 117 | COMPLETE_DIAGNOSTIC |
| r2 时间子算子及两项因果替换 | 165.180秒 | 4 | COMPLETE_DIAGNOSTIC |

两作业监督生命期合计377.039秒；从首次监督窗口开始，含中间CPU分析/准备累计703.318秒（11.72分钟），低于原20分钟帽，未重置。实际CUDA活跃时间UNKNOWN；不是性能测试。新增证据约0.45GiB，低于1GiB；保留真实fixture一次，不复制30层历史到每个query。

两个supervisor及17个观察到的子进程身份均退出，未发送清理信号，无自有GPU任务或waiter。AutoDL未连接。491项旧保护记录size/mtime未变；当前配置与固定公开快照逐字节一致。CPU审查的旧模块同进程重名保护和字段读取错误已修正，不消耗GPU；CPU派生norm初版BF16减法错误保留并由v2明确替代，未修改原GPU分数或科学终态。

## 原始证据

- [r1事前登记](../../../投机采样/results/runtime/p0-draft-teacher-path-diagnostic-v1-20261016-r1/PREREGISTRATION.json)、[r1实际结果](../../../投机采样/results/runtime/p0-draft-teacher-path-diagnostic-v1-20261016-r1/WORKER-RESULT.json)。同目录`REAL-QUERY.pt`保存一个完整可重放fixture，`REAL-STAGES.pt`保存对应root阶段输出。
- [r2事前登记](../../../投机采样/results/runtime/p0-draft-timestep-causal-v1-20261016-r2/PREREGISTRATION.json)、[r2实际结果](../../../投机采样/results/runtime/p0-draft-timestep-causal-v1-20261016-r2/WORKER-RESULT.json)。同目录`CAUSAL-STAGES.pt`保留反事实输出。
- [CPU方向及分数正确版本](../../../投机采样/results/runtime/p0-draft-timestep-causal-v1-20261016-r2/CPU-ROOT-ANALYSIS-v2.json)、[初版派生字段替代说明](../../../投机采样/results/runtime/p0-draft-timestep-causal-v1-20261016-r2/CPU-ANALYSIS-v1-SUPERSEDED.json)、[资源/保护/清理复核](../../../投机采样/results/runtime/p0-draft-timestep-causal-v1-20261016-r2/FINAL-CPU-AUDIT.json)。
