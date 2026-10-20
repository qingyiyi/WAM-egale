---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-20T14:20:35+08:00'
current: M40
next: null
---
# 当前研究进度

更新：2026-10-20T14:17:31+08:00。当前自有GPU作业均已结束；M40因环境阻塞停止，不是机器人实验完成。

|ID|工作|状态|终态说明|证据|
|---|---|---|---|---|
|`M17`|四套既定6000局|in_progress|计算齐；科学审计未放行|[质量](../../experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)|
|`M31`|静态tile子图|completed|有限通过|[结果](../../experiments/transformer-tile-subgraph-20261019/RESULT.md)|
|`M32`|阶段内persistent|completed|更慢，负结果保留|[结果](../../experiments/persistent-tile-remote-20261019/RESULT.md)|
|`M33`|退化profile|completed|无新增正收益，NCU未知|[结果](../../experiments/single-block-persistent-profile-20261019/RESULT.md)|
|`M34`|原完整单kernel|completed|旧9输出FAIL/速度NO-GO|[结果](../../experiments/single-layer-warp-specialized-20261019/RESULT.md)|
|`M35`|新同卡工程计时|completed|330/360，原PARTIAL保留|[结果](../../experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)|
|`M36`|RoboTwin资料核验|completed|官方模型/任务与接口核验|[调研](../../experiments/robotwin-migration-survey-20261019/RESULT.md)|
|`M37`|设备契约/成本|completed|旧四原语FAIL；任务成本归因|[结果](../../experiments/device-contract-20261020/RESULT.md)|
|`M38`|原慢点测量|completed|GEMM主导、旧版本慢5.09–11.14倍|[结果](../../experiments/whole-layer-bottleneck-20261020/RESULT.md)|
|`M39`|单层数值/原语/ready|completed|本次有限精度PASS、速度NO-GO，完整矩阵缺|[最终](../../experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)|
|`M40`|RoboTwin有限迁移|blocked|原渲染阻塞，已停止，0policy/teacher|[止损](../../experiments/robotwin-pilot-20261020/ENVIRONMENT-STOP-RESULT.md)|

## 本次实际结果

M39在optimized内完成原LayerNorm设备体、cuBLASLt有限BF16分段/CTA归约与bias顺序对齐；增大大M GEMM tile，加入FFN行组release/acquire交接，最后删除Params复制并使用grid_constant。9种真实输入×阶段/ready两路径输出与可配对DAG边界逐位通过；每路径18有效KV转移直接对Torch通过，输入/RNG/重复/tile审计通过。

最终默认仍慢native2.68–5.02倍，ready额外慢1.2%–6.0%，默认关闭ready。108CTA、255 registers、121376B shared、1880B localSizeBytes；实测occupancy/流量UNKNOWN。最终27trace各1compute kernel。不是最强cache/Graph、完整verifier/chunk/机器人E2E，不制造g或H3/H4。1380/1500 block、43owned根进程回收，GPU worker生命周期324.748秒；CPU编译/总体墙钟另列。[资源与全部失败](../../experiments/single-layer-async-gemm-20261020/FOLLOWUP-STOP-REPORT.md)。旧M39 registration240/PLAN900 FAILED及原数值失败保留。

M40已准备固定官方源码、Target/公共组件、资产与私有SAPIEN/CuRobo环境，所有写入限~/cxz。原OIDN2.0.1和私有2.3.3均在首个真实场景非法访存；495.321秒计原24h帽，0完整policy、0teacher、0新训练/CAL/DEV。未改光追/去噪/相机来绕过，根因未独立锁定为驱动。原Target首6局前置未过，后继不自动提交。[质量空表](../../experiments/robotwin-pilot-20261020/QUALITY-RESULT.md)。

14:12北京现场两卡0MiB/0%、无compute进程，自有guard/worker无残留。不是保证后续无人占用；AutoDL未连接。

## 原LIBERO事实与边界

四套每臂500成功数：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。八组合P50/P90在旧界内，max全部>1.30；失败/800罚分、首两S严格配对FAILED、训练/DEV曝光、组件失败及缺匹配500局MLP保留。不是独立TEST或非劣。

M35计时重复不增加独立质量N；Object1入口0episode实施失败不自动重跑，330/360终态。Long全尝试1.107×/共同成功1.344×、Object1.216×为原工程观察，不借新算子精度通过签发算法质量。实际环境命令796与失败罚分800分列。

## 下一步、授权与日期

先解决原语效率再扩大调度；完整1/3/7×cache强baseline矩阵仍缺，B6未准入。RoboTwin需可运行官方原渲染的兼容环境，单局/174局耗时UNKNOWN，不能从LIBERO外推。已向用户询问是否有已跑通环境；未购买/改系统。

[新执行授权](../../memory/global/requirements/decision-执行RoboTwin迁移与算子有限收口.md)替代此前仅规划。GPU0原8h、GPU1原24h/准备10月20日18:20北京截止不重置；当前没有待自动启动任务。所有旧FAILED、retention、四项false/default A_off保留。baseline、原tile/HW、既有投机代码/配置和state/cycle/rules不改，只main发布源码/HW/小证据。

10/22质量/系统go-no-go、10/23冻结引擎/claim、10/28主表、11/01 18:00老师中文证据包。CVPR官方日期/作者profile见[日期报告](../../experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)，不保证投稿或录用。中文正文由用户飞书整理。

本轮更早完整checkpoint保留[历史进度](history/ordered-followup-20261020/PROGRESS.md)，前轮见history/M34-complete-20261020，不默认全量载入。
