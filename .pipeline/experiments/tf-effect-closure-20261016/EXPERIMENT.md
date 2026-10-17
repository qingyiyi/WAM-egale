---
kind: experiment
id: tf-effect-closure-20261016
cycles: [wam-research-20261016]
status: paused_by_user_partial
---
# 固定 TF64 的唯一一次新分布训练与完整 DEV

> 10/17早用户明确暂停。新30/固定训练完成，DEV72/96，当前权重风险门不满足；GPU清理完成。最新只读结果见[RESULT](RESULT.md)，不按下方历史继续启动。

用户已同意继续完成诊断→等预算训练→完整任务验收，当前结构固定 TF64。执行清单见 [PLAN](PLAN.md)，来源固定 `b0dfac3a02f65de33a1217246b83003615166640`，原接口及 FAILED、阈值、四权限均不改。未启动120、TEST或正式性能；AutoDL不连接。

## 1：监督/覆盖诊断终态

[CPU登记及结果](../../../投机采样/results/runtime/p0-tf-supervision-alignment-cpu-v1-20261016-r1/RESULT.json)：57.14秒、CPU2核、0GPU、没有teacher调用或训练。完整已知失败DEV的904树/2712 query，仅作诊断，不能进入新FIT/CAL。

按同task、scheduler step匹配，当前真实到达parent平均局部误差0.002568，target FIT参考0.001734，约高48%；中σ为0.004525对0.002905。当前20.52%的到达parent联合标准化state/previous最近FIT距离超过旧CAL对应step的P95；root为19.03%。这是描述性覆盖信号，不是新增质量门。CPU batched预测只用于离线比较，不替换GPU实际BF16接受。

旧14997混合池并非当前TF on-policy；到达parent的旧混合参考误差0.002483，与当前接近，因此**不能声称新分布严重失控，也不能据此解释task4失败**。历史纯target前一方向来自teacher；当前递归parent前一方向来自draft，且实际B14教师与原B2标签存在已确认数值差异（目前只有单query因果定位）。简单RMS并未爆炸，联合覆盖诊断补充了前次审计缺口。

## 有限训练准入解释

本轮允许检验“用当前策略真正到达的同query教师补训练，能否改善TF局部拟合及闭环质量”。依据是已观察的递归输入语义、联合覆盖与同step监督误差缺口，而非假定失败根因成立。唯一可证伪控制是同child2500、同500更新的TF-cont/TF-on；若新数据无优势，停止本轮TF tuning，不迭代采更多数据或换结构/阈值。

这只触发一次受限研究实验；不修改前两次 `training_trigger=false`，也不开放旧质量/性能权限。

## 2：数据与训练方案

最初对已完整记录的TF task4/init26全部50个policy及cache请求设计了两遍OFF/ON且逐请求复制全部KV，r2因诊断复制成本预期超原预算主动停止，原PARTIAL保留。新r3复用原M11完整OFF基线，只重放collector ON完整50个policy/cache：逐位核所有原actions/latents/接受路径/teacher和逐policy前后RNG，reset与最后核全部有效30层KV及RNG。它是不同且明确登记的观察范围，**不声称原两遍/每请求KV门通过**；所有Attempt共用原21:49:55绝对清理截止，不扩45分钟帽。不重新跑机器人，不作为新episode数；原O3保持启用，额外teacher forward=0。r3及独立完整CPU字段审计都通过才准新30采集。

通过后一次性采task0–9×init30/31（20 FIT）和init32（10 CAL）的完整episode，当前TF_CONT3000固定生成，root+实际到达parent入训练列；反事实节点另留。20/10事前隔离，不按成败选局。训练从child2500同起点，不把生成策略3000误作初始化权重；旧normalizer/loss/lr/batch/seed/dtype/API固定。主seed23固定末500步；matched MLP-on仅为必要公平对照，后置于TF实现。各GPU作业另登记有限cap、输入、产物及自有清理。

资格r3及[完整CPU审计](../../../投机采样/results/runtime/p0-tf-effect-collector-reuse-audit-cpu-v1-20261016-r1/RESULT.json)均已完成，904树/2712行。原r1/r2失败保留，r3 GPU作业监督793.33秒；原累计45分钟帽内结束并清理。首条完整新episode的数据通路审计975全node/607reached、CPU六score最大差1.49e-8通过，不替代最终30局审计。

**实际采集已完成**：[collect30](../../../投机采样/results/runtime/p0-tf-effect-collect30-v1-20261016-r1/COLLECTION.json)，30/30完整终态，9383.734秒、23685个target forward、约132.3MiB；原4h/2GiB内，root/clock和全部观察到的worker已退出。生成池29成功/1完整失败均保留，不能作新TF质量结果。

[全量pack审计](../../../投机采样/results/runtime/p0-tf-effect-collect30-pack-cpu-v1-20261016-r1/DATA-AUDIT.json)49.367秒PASS：9797树/29391原始query，主列18221到达行（FIT11416/CAL6805），11170反事实另留；六score CPU重算最大差2.98e-8、0接受翻转，实际GPU分数不替换，额外teacher0。

**固定三条件训练及完整CPU1校验已完成**：[新终态](../../../投机采样/results/runtime/p0-tf-effect-onpolicy-train-cpu1-v2-20261017-r1/FINAL-RESULT.json)，CONT复用已完成500更新、ON/MLP首次各500，新作业1061.888秒，0GPU；原CPU作业1765.165秒FAILED保留，累计2827.053秒。训练仍2线程、完整逐行校验1线程，不继承旧CPU2数值结果；生产GPU完整旧/新CAL在新arm开始前执行。**DEV96已实际登记并运行**：`p0-tf-effect-dev96-v1-20261017-r1`，不是已获新质量结果。

接续见[RUNBOOK](RUNBOOK.md)。96局清单在新数据前冻结；按首条270.5秒任务将DEV上限事前改为7h，原活跃作业08:11:37清理帽不变。旧全轮12h截止10/17 09:04:55及原5h预测保留；用户最新授权至10:00只允许另登记一次固定96缺项接续，不改旧帽、不重跑完整失败、不扩训练/阈值/TEST/性能权限。

## 3：完整 DEV 与门

权重固定后重新登记完整任务，不复用旧T作为新的严格配对基线，不把训练/调参数据称独立TEST。必须报告全部任务成功数、双方成功步长P50/P90/max、实际步数及失败800罚分、已知风险回归、matched MLP与真实配对初态/RNG。沿用原成功数/长度/风险条件；结构偏好不放宽门。未满足则TF研究候选NO-GO，系统可研究、最终闭环claim不可写。
