---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-21T11:02:47.052904+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前有界计划

| ID | 阶段 | 期望结果与验证 |
|---|---|---|
| `M17` | LIBERO证据收口 | 保留6000局质量/长度/成本与科学缺口，不追加TEST |
| `M39` | 简易单层先对齐 | 真实Torch/剥离原语/整层条件对照，有限精度与性能证据 |
| `M40` | RoboTwin迁移 | 固定Target48→parent18→current18→既定DEV90，预算/分区不变 |

## 当前todo

- [x] 真实Torch GELU向量化/调制入口核验，三KV同输入独立剥离和FFN2写回/分段对照。
- [x] table/GELU vec4入整层：九签名有限逐位、两独立进程完整层改善2.7%–3.1%。
- [x] FFN2并行K独立接近Torch，入整层未受益，保留负结果并恢复较快源/DSO。
- [x] 用户授权补齐parent18，完整监督审计，CHILD/CONT固定训练与GPU部署接口CAL。
- [x] 全部本轮进程退出，原始失败/预算/截止保留，短HW更新。
- [x] M39：官方来源冻结、12小算子同输入核对、point/RMS整层集成、全29阶段及CTA等待表完成。
- [x] CPU补齐短/中/长KV、12原语/两进程与完整层汇总；包括退化和波动。
- [x] 同前序控制与Cross/FFN设备函数控制完成，保留两项修改。
- [ ] 下一步：先FFN剩余原语/整层资源差距，再按适配tile和真实DAG设计最小ready调度。
- minor-ops本轮已结束，GPU207.379/1800秒，M39累计1611.324/36000秒，旧截止保留。
- [x] M40最新授权：GPU1启动固定CURRENT18；原09:30截止保留，新窗口独立登记。
- [x] 已部署自动CURRENT完整审计→ON3000→部署CAL队列。
- [ ] 完成固定DEV90三臂入口/首两S审计核查，再接既定完整评估。

M39前轮累计1403.945秒；本轮新增207.379秒，累计1611.324/36000秒。M40累计5.7624/24小时。
BAAI仅~/cxz，AutoDL已归还。保护state/cycle/rules、baseline、原tile、LIBERO生产、四项false不变。

历史准备：[GEMM同前序条件诊断](../../experiments/single-layer-async-gemm-20261020/gemm-context-control-20261021/PLAN.md)，GPU≤600秒、CPU2，该准备时尚未恢复机器人；现已由新用户优先授权接续。

- [x] 同前序Cross KV/FFN两进程诊断完成（首失败保留），下一步设备函数/编译资源单因素。M39当前累计1649.673秒。
- [ ] 当前优先M40，CURRENT18正在GPU1运行；新截止与原24h余额见current18登记，四权限false不变。

## 最新执行：先做Cross KV和FFN

- [x] Cross KV固定几何设备函数专门化：CPU编译资源及原9签名门、三KV两进程完整层。
- [x] FFN1/2按Cross结果处理设备函数/执行资源；固定原数学和tile。
- [x] 汇总所有负结果与实际收益，保留有稳定收益的版本；本轮不执行ready调度。

本轮沿M39证据，GPU≤1800秒、CPU2、准备≤2400秒、存储≤1GiB；累计从1649.673/36000秒接续。GPU1 RoboTwin独立继续。

本轮终态：Cross/FFN两项完成；实际GPU120.995秒，累计1770.668/36000秒，CPU准备523.226秒。12:19核验owned退出。见[结果](../../experiments/single-layer-async-gemm-20261020/gemm-function-control-20261021/RESULT.md)；历史预算起点保留。

## 当前接续：FFN与整层资源原因

- [ ] 当前同源码独立/整层与相同前序控制；限定三KV、两进程。
- [ ] 一次专用阶段FFN2分段并行候选，有限精度及完整层保留/回退。
- [ ] 解释已确认原因和未知，更新证据；RoboTwin继续后台。

沿原M39累计1770.668秒接续；本次GPU≤1800秒、CPU2、构建≤2400秒、存储≤1GiB，绝对截止2026-10-21T14:07:22.760819+08:00。不执行跨阶段ready。
