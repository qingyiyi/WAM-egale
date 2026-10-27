# libero_goal 仅三臂共同成功的5090速度 v1

按用户2026-10-27补充要求，从原始计时重新统计；0 GPU调用。筛选使用测速观测来源的原DEV episode成功身份，不使用后来全量重复的结果替换。target、TF、TF+PF三臂共同成功，保留集合内所有快慢样本。

|边界|共同成功源组N / 原N|计时对N|target均值ms|TF均值ms|加速比|条件性95%区间|
|---|---|---:|---:|---:|---:|---|
|完整动作去噪|20/20|300|61.086|50.602|1.20718×|[1.18868, 1.22625]|
|完整policy chunk|20/20|300|176.144|165.721|1.06289×|[1.05800, 1.06785]|

|policy trace回放|共同成功N / 原N|TF加速比|TF+PF加速比|
|---|---|---:|---:|
|固定target观测、等长|4/4|1.06403×|1.04212×|
|各自自然长度|4/4|1.04471×|1.02040×|

同一比较集合内按总target时间/总候选时间计算。自然trace含真实长度差异，但仅计policy推理回放，不含仿真步进。上述N不是500组全量测速，也不是5090新闭环成功率。加载和首次编译仍在原收据中单列；成功筛选不抹除冷启动成本。

全部失败、PARTIAL、完整质量分母及原始计时保留；只替换后续加速比汇报口径，四科学权限不变。

源组筛选：

|task / init|T / TF / PF成功|纳入|
|---|---|---|
|0 / 47|True / True / True|是|
|0 / 48|True / True / True|是|
|1 / 47|True / True / True|是|
|1 / 48|True / True / True|是|
|2 / 47|True / True / True|是|
|2 / 48|True / True / True|是|
|3 / 47|True / True / True|是|
|3 / 48|True / True / True|是|
|4 / 47|True / True / True|是|
|4 / 48|True / True / True|是|
|5 / 47|True / True / True|是|
|5 / 48|True / True / True|是|
|6 / 47|True / True / True|是|
|6 / 48|True / True / True|是|
|7 / 47|True / True / True|是|
|7 / 48|True / True / True|是|
|8 / 47|True / True / True|是|
|8 / 48|True / True / True|是|
|9 / 47|True / True / True|是|
|9 / 48|True / True / True|是|
