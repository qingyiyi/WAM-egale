---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: '2026-10-20T16:24:50+08:00'
current: M40
next: null
---
# 当前研究进度

更新：2026-10-20T16:24:50+08:00。固定专家初态114/114齐；首局Target已完整成功，8policy/240命令、184.197秒；GPU1继续第二局stack_blocks_two。首局旧视频入口PARTIAL保留，用户已批准补跑。算子本次仅审查。

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
|`M40`|RoboTwin有限迁移|in_progress|114/114初态；首局Target成功8policy/240命令；GPU1第二局运行，旧PARTIAL保留|[接续](../../experiments/robotwin-pilot-20261020/RESUME-FIRST6-PLAN.md)|

## 本次实际结果

M39在optimized内完成原LayerNorm设备体、cuBLASLt有限BF16分段/CTA归约与bias顺序对齐；增大大M GEMM tile，加入FFN行组release/acquire交接，最后删除Params复制并使用grid_constant。9种真实输入×阶段/ready两路径输出与可配对DAG边界逐位通过；每路径18有效KV转移直接对Torch通过，输入/RNG/重复/tile审计通过。

最终默认仍慢native2.68–5.02倍，ready额外慢1.2%–6.0%，默认关闭ready。108CTA、255 registers、121376B shared、1880B localSizeBytes；实测occupancy/流量UNKNOWN。最终27trace各1compute kernel。不是最强cache/Graph、完整verifier/chunk/机器人E2E，不制造g或H3/H4。1380/1500 block、43owned根进程回收，GPU worker生命周期324.748秒；CPU编译/总体墙钟另列。[资源与全部失败](../../experiments/single-layer-async-gemm-20261020/FOLLOWUP-STOP-REPORT.md)。旧M39 registration240/PLAN900 FAILED及原数值失败保留。

M40旧两版OIDN失败保留。本次同脚本/同原场景对照实际加载库：系统535两次完整专家与四路RGB成功；继承的CUDA13.2 compat595在读取原相机观察时重现非法访存。已仅修改RoboTwin客户端LD_LIBRARY_PATH，驱动仍535、rt/32spp/path8/oidn保持。新增119.345秒计入原24h，累计614.666秒；0policy/teacher/训练/DEV。最终两卡0MiB、无owned残留。[修复原件](../../experiments/robotwin-pilot-20261020/DRIVER-INTEROP-REPAIR-RESULT.md)。A100可运行这个原场景，当前无需为该故障换4090；全任务稳定性待首6条Target验证。

[当前MegaMoE差距](../../experiments/single-layer-async-gemm-20261020/MEGAMOE-GAP-REVIEW.md)：单kernel/warp分工/双缓冲/行组ready已有；K片段ready、有界中间槽及更独立的角色流水未完成。宽GEMM有32KiB仅小M用的partial_y预留，是B14专用化机会，尚未改码或测收益。旧M38阶段比例不冒充当前版本profile。

M40 control-6专家None抓取断言790.555秒及旧FAILED保留，按官方规则从下一seed接续后，control-7已完成114/114固定初态并清理。first6-2构建108条完整512-token BF16 bank、首case初态逐位通过，但可选视频目录未彻底关闭，调用不存在ffmpeg，在0teacher/0动作处PARTIAL；70.743秒和原件保留。用户明确允许补跑，独立wrapper只移除该导出目录，first6-3首局已完整成功，8policy/240实际命令、184.197秒，初始物理/观察逐位通过；16:24 GPU1第二局stack_blocks_two正在运行。原首六局17:46:42北京deadline不变。[入口修复与补跑](../../experiments/robotwin-pilot-20261020/FIRST-CASE-RETRY.md)。text bank张量逐位检查通过，完整任务缓存等价仍未实测。后继TF采集/训练/DEV未完整实现。

## 原LIBERO事实与边界

四套每臂500成功数：Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。八组合P50/P90在旧界内，max全部>1.30；失败/800罚分、首两S严格配对FAILED、训练/DEV曝光、组件失败及缺匹配500局MLP保留。不是独立TEST或非劣。

M35计时重复不增加独立质量N；Object1入口0episode实施失败不自动重跑，330/360终态。Long全尝试1.107×/共同成功1.344×、Object1.216×为原工程观察，不借新算子精度通过签发算法质量。实际环境命令796与失败罚分800分列。

## 下一步、授权与日期

用户最新[接续迁移](../../memory/global/requirements/decision-修复后接续RoboTwin有限迁移.md)。已实现专家清单→首6条Target有界运行器；后继TF适配/训练/DEV未完整实现，不声称自动排队。已闭合GPU账本2266.544秒，运行中另计。当前[填尾波方案](../../experiments/single-layer-async-gemm-20261020/TAIL-FILL-SCHEDULING-REVIEW.md)为审查建议，未启动新算子测量。

原始action路径按用户要求保留原算子。仅看B14，默认自有单层短/中/长KV为12.9445/10.8687/11.2424ms，native为2.5778/2.4073/2.8418ms，即耗时5.022/4.515/3.956倍；是uncached eager单层工程结果，不是最强cached/Graph或E2E。进一步研究应先解决原语效率再扩大调度；完整1/3/7×cache强baseline矩阵仍缺，B6未准入，本次问答未新增GPU任务。

[原执行授权](../../memory/global/requirements/decision-执行RoboTwin迁移与算子有限收口.md)与本次接续决定合用。GPU0原8h、GPU1原24h/准备10月20日18:20北京截止不重置。旧FAILED、retention、四项false/default A_off保留。baseline、原tile/HW、既有投机源/配置和state/cycle/rules不改，只main发布源码/HW/小证据。

10/22质量/系统go-no-go、10/23冻结引擎/claim、10/28主表、11/01 18:00老师中文证据包。CVPR官方日期/作者profile见[日期报告](../../experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)，不保证投稿或录用。中文正文由用户飞书整理。

本轮更早完整checkpoint保留[历史进度](history/ordered-followup-20261020/PROGRESS.md)，前轮见history/M34-complete-20261020，不默认全量载入。
