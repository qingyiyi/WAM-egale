---
kind: progress
cycle: wam-research-20261016
plan: PLAN.md
status: active
updated: 2026-10-17T17:22:00+08:00
current: M17
next: produce_assets_and_complete_suite_target_collection_then_fixed_TF_training
execution_status: controllers_waiting_inputs_Long_four_workers_running
---
# 当前研究进度

M17已批准三小套件各自同配方TF及Target/TF/PF2全套对照。四套权重独立，不拿社区发布者成功率代替实测；不把Long draft当作已适配。见[实际状态](../../experiments/libero-four-suite-20261017/ACTUAL-RUN-STATUS.md)。

|工作|实际状态|证据|
|---|---|---|
|M17 Object/Goal资产|两份各10,177,831,668B已在远端完整下载；Object传本机中|asset r2 RESULT；本机import r3 status|
|M17 reset bank|本机Object CPU7；远端Goal/Spatial CPU4 affinity、Torch7线程，生产中|bank各job，不声明在线数学等价已重验|
|M17 三条pipeline|source9aa66fb，控制器已启动等待；新模型/episode0|master与三个campaign status|
|M16 Long|四worker/两卡继续；17:20闭合约80组三臂/240局（含旧23组69局），另有运行中局|原v2 status，旧源不动|
|LIBERO90|没有已核实同代Target；训练未启|原可靠性检索|

## 历史质量不覆盖

固定4f04 paused72永久72/96、MLP0、PAUSED_BY_USER_PARTIAL_NOT_QUALIFIED；本机live77另列。旧TF_ON fresh20/20不覆盖风险4/8对Target6/8，三条新增风险(1,29)/(3,26)/(4,26)保留。旧22清理缺证、所有FAILED和暂停/预算中断保持。

M15 PF2六对完整：C3/6、P5/6，救回2、无lost；risk4/26仍败，控制1/36 260→405，共同成功P50/P90/max=1.0043/1.4470/1.5577，投入NO_GO。旧四权限false；新大面板只补描述证据，不能追改原门。

结构TF64及API暂冻不等于权重通过。唯一Long训练机会已用，新增三套是用户新批准的对应teacher独立训练，不无限调参。accept tau/selector/树/bonus固定，PF2=SST，T是canonical50 scheduler更新/51 action调用，非bonus/物理回滚。

## 性能与资源边界

并行质量计时仅探索性；正式完整任务E2E需质量与独占。旧capture重复字段/有效KV桶缺口均保留UNKNOWN，尚无可宣称合格的新TF全状态replay。系统H1/H2有机制证据，H3有限pilot，H4未成立；原tile不变。

AutoDL无smoke/独立测试/SHA，包括隐式loader；所有臂完整512-token BF16优化reset。只清理本项目确认可重建/无引用内容，原失败与唯一权重保留，不杀他人。各job有绝对deadline和PID出生证明；脚本退出不等于关机。

原M01–M16完整进度见[历史原件](history/PROGRESS-through-M16-20261017.md)，原结果链接仍在；当前短状态只更新这里，不追加整篇旧总结。
