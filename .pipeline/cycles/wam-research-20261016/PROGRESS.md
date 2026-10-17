---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: 2026-10-17T18:42:29+08:00
current: M17
next: produce_assets_and_complete_suite_target_collection_then_fixed_TF_training
execution_status: remote_shared_controllers_waiting_Long_bank_and_asset_rotation
---
# 当前研究进度

M17已批准三小套件各自同配方TF及Target/TF/PF2全套对照。四套权重独立，不拿社区发布者成功率代替实测；不把Long draft当作已适配。见[实际状态](../../experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md)。

| ID | 工作 | 状态 | 实际进展 | 证据 |
|---|---|---|---|---|
| `M16` | Long全量三臂 | in_progress | 18:42新285闭合局＋原69＝354；另有运行中局，非全量 | 原v2 status，旧源不动 |
| `M17` | 三套独立TF与全量三臂 | in_progress | 8eaf130远端共享资源控制器32543/31852/31853等待（Object结果改OS盘）；新模型/episode/训练0 | [实际作业](../../experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md) |

Object/Goal远端权重完整，本机Object也已完整；Goal/Spatial完整bank完成（2455秒CPU），Object远端bank v3 r3完成（1220.5秒CPU）。原本机3600秒bank超时、传输3200秒超时、缺收据/启动参数失败及旧控制器替换全部保留。

用户要求本机减负并暂不扩容，已在新namespace将Object迁远端，保持17:11起点和10/21截止；Long清理后才核查可重建Long远端Transformer缓存轮换Spatial。Object新增结果放远端OS盘，独立新namespace仍GPU0/原5GiB及5GiB空闲线；归档Long到本机后才退远端重复payload/COUNTER，归档128d435/PID443492已启动等待，复制0/删除0。本机本次清理7个失败下载残片985,044,916B，剩约94.9GiB；没有删除权重、原始结果或失败记录。LIBERO90仍缺可信同代Target，未启动。

## 历史质量不覆盖

固定4f04 paused72永久72/96、MLP0、PAUSED_BY_USER_PARTIAL_NOT_QUALIFIED；本机live77另列。旧TF_ON fresh20/20不覆盖风险4/8对Target6/8，三条新增风险(1,29)/(3,26)/(4,26)保留。旧22清理缺证、所有FAILED和暂停/预算中断保持。

M15 PF2六对完整：C3/6、P5/6，救回2、无lost；risk4/26仍败，控制1/36 260→405，共同成功P50/P90/max=1.0043/1.4470/1.5577，投入NO_GO。旧四权限false；新大面板只补描述证据，不能追改原门。

结构TF64及API暂冻不等于权重通过。唯一Long训练机会已用，新增三套是用户新批准的对应teacher独立训练，不无限调参。accept tau/selector/树/bonus固定，PF2=SST，T是canonical50 scheduler更新/51 action调用，非bonus/物理回滚。

## 性能与资源边界

并行质量计时仅探索性；正式完整任务E2E需质量与独占。旧capture重复字段/有效KV桶缺口均保留UNKNOWN，尚无可宣称合格的新TF全状态replay。系统H1/H2有机制证据，H3有限pilot，H4未成立；原tile不变。

AutoDL无smoke/独立测试/SHA，包括隐式loader；所有臂完整512-token BF16优化reset。只清理本项目确认可重建/无引用内容，原失败与唯一权重保留，不杀他人。各job有绝对deadline和PID出生证明；脚本退出不等于关机。

原M01–M16完整进度见[历史原件](history/PROGRESS-through-M16-20261017.md)，原结果链接仍在；当前短状态只更新这里，不追加整篇旧总结。
