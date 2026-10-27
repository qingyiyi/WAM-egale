# libero_10 仅三臂共同成功的5090速度 v1

按用户2026-10-27补充要求，从原始计时重新统计；0 GPU调用。筛选使用测速观测来源的原DEV episode成功身份，不使用后来全量重复的结果替换。target、TF、TF+PF三臂共同成功，保留集合内所有快慢样本。

|边界|共同成功源组N / 原N|计时对N|target均值ms|TF均值ms|加速比|条件性95%区间|
|---|---|---:|---:|---:|---:|---|
|完整动作去噪|16/20|240|61.437|49.502|1.24109×|[1.22031, 1.26118]|
|完整policy chunk|16/20|240|183.246|170.995|1.07165×|[1.06639, 1.07706]|

|policy trace回放|共同成功N / 原N|TF加速比|TF+PF加速比|
|---|---|---:|---:|
|固定target观测、等长|3/4|1.06433×|1.04348×|
|各自自然长度|3/4|1.07527×|1.05507×|

同一比较集合内按总target时间/总候选时间计算。自然trace含真实长度差异，但仅计policy推理回放，不含仿真步进。上述N不是500组全量测速，也不是5090新闭环成功率。加载和首次编译仍在原收据中单列；成功筛选不抹除冷启动成本。

全部失败、PARTIAL、完整质量分母及原始计时保留；只替换后续加速比汇报口径，四科学权限不变。

源组筛选：

|task / init|T / TF / PF成功|纳入|
|---|---|---|
|0 / 47|True / True / False|否|
|0 / 48|True / True / True|是|
|1 / 47|True / True / True|是|
|1 / 48|True / True / True|是|
|2 / 47|True / True / True|是|
|2 / 48|False / False / False|否|
|3 / 47|True / True / True|是|
|3 / 48|True / True / True|是|
|4 / 47|False / True / True|否|
|4 / 48|True / True / True|是|
|5 / 47|True / True / True|是|
|5 / 48|True / True / True|是|
|6 / 47|True / True / True|是|
|6 / 48|False / True / True|否|
|7 / 47|True / True / True|是|
|7 / 48|True / True / True|是|
|8 / 47|True / True / True|是|
|8 / 48|True / True / True|是|
|9 / 47|True / True / True|是|
|9 / 48|True / True / True|是|
