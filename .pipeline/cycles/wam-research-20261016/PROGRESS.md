---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: 2026-10-17T22:08:59+08:00
current: M18
next: observe_new_shard_resume_then_migrate_at_complete30_boundary_when_new_SSH_arrives
execution_status: original567_backup_complete_four_remote_workers_READY_running_whole_episodes
---
# 当前研究进度

M17已批准三小套件各自同配方TF及Target/TF/PF2全套对照。四套权重独立，不拿社区发布者成功率代替实测；不把Long draft当作已适配。见[实际状态](../../experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md)。

| ID | 工作 | 状态 | 实际进展 | 证据 |
|---|---|---|---|---|
| `M16` | Long全量三臂 | in_progress | 旧阶段189组/567局完整归档；311未开始组正由新namespace分片接续，4模型READY、首30局分片完整任务运行中；原PARTIAL保留 | 原v2 status，旧源不动 |
| `M17` | 三套独立TF与全量三臂 | in_progress | Object本机首局超时/PARTIAL，0完整局已清理；Goal/Spatial按换机请求暂停，训练0 | [实际作业](../../experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md) |
| `M18` | 阶段备份与换机准备 | in_progress | 全部37349文件/1.84GiB原件已备份核对/CLEAN；新机未接入，新作业按30局边界可暂停 | [换机计划](../../experiments/autodl-stage-migration-20261017/PLAN.md) |

Object/Goal远端权重完整，本机Object也已完整；Goal/Spatial完整bank完成（2455秒CPU），Object远端bank v3 r3完成（1220.5秒CPU）。原本机3600秒bank超时、传输3200秒超时、缺收据/启动参数失败及旧控制器替换全部保留。

最新授权允许本机空闲GPU分担，Object在零新增数据时改为本机SXM4单模型/nice10/server2-client2；CPU1导入完整bank128.9秒，未重做T5/teacher。远端Object32543已按PID/出生/UID退役，原FAILED_OR_PARTIAL终态保留；新本机Object装载749.6秒成功，首局1200秒超时，20:24终态FAILED_OR_PARTIAL_NO_RETRY、0完整局；guard/owned组CLEAN，无自动重试。Goal/Spatial已停止排队、租约释放；缓存轮换0删除终止。17:11起点和10/21截止、5GiB/套及四false不改，硬件分层计时。原全1500归档443492因换机请求退出，RPC实际exit0、复制/删除0；原604568逐文件方式过慢，实际SIGINT退出/SSH闭合；635195单tar在650.7秒完成/CLEAN，37349文件、源1970600995B；旧源0删除。2b1f039新接续69459/runner69462/guard69461四模型READY，首30局分片运行中，沿原10/19截止；新结果迁移前另备份，当前不可关旧机。本机本次清理7个失败下载残片985,044,916B，剩约94.9GiB；没有删除权重、原始结果或失败记录。LIBERO90仍缺可信同代Target，未启动。

阶段189组描述结果（不是验收）：Target183/189、TF连续185/189、PF2 181/189；平均失败800罚分284.81/274.76/284.22。相对Target共同成功最大长度比TF1.7004/PF2 1.5465；原组件审计FAILED仍保留（C15/P27）。含FIT/CAL暴露，不签独立TEST或正式速度，四false不改。见[阶段汇总](../../experiments/autodl-stage-migration-20261017/STAGE189-RESULT.md)。

## 历史质量不覆盖

固定4f04 paused72永久72/96、MLP0、PAUSED_BY_USER_PARTIAL_NOT_QUALIFIED；本机live77另列。旧TF_ON fresh20/20不覆盖风险4/8对Target6/8，三条新增风险(1,29)/(3,26)/(4,26)保留。旧22清理缺证、所有FAILED和暂停/预算中断保持。

M15 PF2六对完整：C3/6、P5/6，救回2、无lost；risk4/26仍败，控制1/36 260→405，共同成功P50/P90/max=1.0043/1.4470/1.5577，投入NO_GO。旧四权限false；新大面板只补描述证据，不能追改原门。

结构TF64及API暂冻不等于权重通过。唯一Long训练机会已用，新增三套是用户新批准的对应teacher独立训练，不无限调参。accept tau/selector/树/bonus固定，PF2=SST，T是canonical50 scheduler更新/51 action调用，非bonus/物理回滚。

## 性能与资源边界

并行质量计时仅探索性；正式完整任务E2E需质量与独占。旧capture重复字段/有效KV桶缺口均保留UNKNOWN，尚无可宣称合格的新TF全状态replay。系统H1/H2有机制证据，H3有限pilot，H4未成立；原tile不变。

AutoDL无smoke/独立测试/SHA，包括隐式loader；所有臂完整512-token BF16优化reset。只清理本项目确认可重建/无引用内容，原失败与唯一权重保留，不杀他人。各job有绝对deadline和PID出生证明；脚本退出不等于关机。

原M01–M16完整进度见[历史原件](history/PROGRESS-through-M16-20261017.md)，原结果链接仍在；当前短状态只更新这里，不追加整篇旧总结。
