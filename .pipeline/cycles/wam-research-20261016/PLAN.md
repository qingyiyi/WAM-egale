---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-19T17:50:34.115962+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# WAM投机质量与算子研究

| ID | 工作 | 状态 | 当前边界/证据 |
|---|---|---|---|
| M16 | Long500组三臂首轮 | completed | T483/C484/P473；描述性，原配对/曝光问题保留 |
| M17 | Object/Goal/Spatial既定评估 | in_progress | Object 1166/1500；T500/C333/P333为完成数，非成功数，来源2026-10-19T09:46:38Z；原同卡队列继续 |
| M28 | 两份完整native block | completed | 六actual30有限逐位、双进程九点动态性能；非强B14/Graph或E2E |
| M29 | 相关工作与创新边界 | completed | 多项投机直接先例，新增调度仍待证 |
| M30 | 结构/shape/DAG/MegaMoE思想交接 | completed | CPU-only，9签名/27参数/21节点；无新persistent kernel |
| M31 | 一个有界tile常驻worker原型 | planned | 尚未登记GPU作业；先显式tile坐标FFN1，再FFN2门与行块DAG |

## 当前可见步骤

- [x] 完整native数学链、实际30层/有效状态和两进程动态block有限门，M28终态。
- [x] 核验既有强baseline/profile/旧负结果，固定MegaMoE官方源码只读。
- [x] 真实输入/参数/GEMM形状与归约DAG，综合HW给出tile/persistent设计。
- [x] Maintain保存用户方向、协作交付与MegaMoE思想偏好，不把具体策略写成已确认结论。
- [ ] 下一包先有界登记当前强B14边界、显式tile worker与FFN2精度诊断；逐项增加常驻和warp分工。
- [ ] 子图→完整block→30层verifier→完整chunk正确性→质量放行后的正式性能。

[M30交接](../../experiments/transformer-tile-handoff-20261019/RESULT.md)、[具体设计](../../experiments/transformer-tile-handoff-20261019/DESIGN.md)、[工作大纲](../../experiments/transformer-tile-handoff-20261019/WORK-OUTLINE.md)。旧M18–M27及原完整经过见[归档](history/M30-before-handoff-20261019/PLAN.md)。M25程序工作已结束，旧严格性能/NVML缺口仍在；旧表非标准status不作为新科学状态使用。

## 预算与准入

M30自10/19 17:23:28起：CPU4/5≤2、GPU0、墙钟≤3600秒、存储≤32MiB。新设计建议的两工作日/≤7 GPU生命周期小时尚非运行账本，不借旧余额或自动启动。M28已清理：1088.541/3600生命周期秒；后继不重置旧截止。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1。BAAI仅~/cxz，AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲，精度可共享；不杀他人/改服务/时钟。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

投机源码/配置/运行依赖与原tile只读；不覆盖state/rules、不建缺失cycle.yaml、不升级或旧sync覆盖HW。当前只维护baseline/optimized；M28完整native baseline作为对照，后续优化限定optimized。 不自动新训练、120/独立TEST、新TF正式45chunk、第二环境或30层大型融合；中文论文仍会话交飞书，不新建本地初稿。

10/22 go/no-go、10/23冻结候选/API/engine、10/28主表、11/01 18:00中文证据包；证据不足则缩小主张，不制造PASS。发布只用main当前镜像，历史固定commit保留。
