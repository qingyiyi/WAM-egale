# 同init受控T/S0原生对照（最多4局）

2026-10-11T13:34:48.591934+08:00：S0先通过与旧ON轨迹逐步已测指纹/20初始数组/原生物理字段/114 action-video完整张量复现关卡，才运行T。四独立首局，条件内两个EGL worker，条件顺序S0→T。无teacher/profile/训练/RL/联训/bonus或公共源码变化；不是clean timing。

| 条件 | init | success | env steps | chunks |
|---|---|---|---|---|
| S0 | 3 | True | 293 | 19 |
| S0 | 4 | True | 600 | 38 |
| T | 3 | True | 293 | 19 |
| T | 4 | True | 300 | 19 |

- init3：T/S0步数 293/293；第一In→第二双指 82/69；首次命令差异step 1。
- init4：T/S0步数 300/600；第一In→第二双指 79/359；首次命令差异step 1。

详细第一分叉、第二罐闭合段与阶段数据：matched-target-s0.json；逐完整张量：matched-target-s0-tensors.csv。In是区域包含，不等于释放；contact不等于持稳。小样本不同算法轨迹对照不证明L2/z位姿唯一机制、正式非劣性或自然E2E加速。

下一步：先CPU独立复核结果与困难阶段；同负载T/A_off成本包装/诊断、冻结target小模型训练均需另行明确，不自动追加局数或训练。
