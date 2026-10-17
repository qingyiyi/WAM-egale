---
kind: plan
experiment: tf-effect-closure-20261016
cycle: wam-research-20261016
status: paused_by_user_partial
---
# 用户授权的 TF64 效果收口 1→2→3

2026-10-16晚用户同意继续测试，完成三步后汇报和下一步。固定公开科学基线 `b0dfac3a02f65de33a1217246b83003615166640`，新增入口独立保存，旧源码/FAILED/四权限不变。TF64主模型，不做architecture sweep；MLP后置为公平baseline。

- [x] 资源与当前Cycle恢复。本机约34GiB可用、lhw有进程；质量可共享，正式性能不运行；AutoDL不连接。
- [x] 1. CPU监督对齐、按scheduler step匹配的误差与联合特征覆盖诊断；实际57.14秒、2核、0GPU，无新teacher或训练。原已知DEV只作诊断。
- [x] 审阅具体机制证据：支持一次当前到达分布补数据实验；不证明严重偏移/整局失败根因，见EXPERIMENT解释与历史mixed反证。
- [x] 2. 已由诊断触发并完成一次20FIT/10CAL完整episode、O3旁路teacher核验和等预算TF-cont/TF-on/MLP-on；生成池不作为新权重质量证书。
- [x] 新collector完整记录资格与2712行CPU核验；r1/r2原失败保留，r3新范围真实通过且清理，未声称全部中间KV证明。
- [x] 一次30完整数据采集及全量pack审计完成并清理：FIT11416/CAL6805实际到达行，原始反事实保留、额外teacher0。接续见RUNBOOK。
- [x] 固定TF-cont/TF-on/MLP-on各500更新完成：原CPU r1校验中止FAILED/1765.2s保留；独立CPU1完整校验续接1061.9s PASS，CONT复用新增0更新、ON/MLP首次各500。三条件全旧/新CAL CPU接口通过，GPU接口仍待各arm实际校准。
- [ ] 3. 原96仅72完整，用户已暂停、剩24；旧/接续PARTIAL均保留，不能称96完成。历史登记：`p0-tf-effect-dev96-v1-20261017-r1`，root3787524/clock3787527，7h/3GiB，08:11:37清理硬帽。已知风险和MLP对照均预登记；不自动启动120/TEST/正式性能。
- [x] 当前72终态描述/50严格组件/22旧缺口、四风险配对证据、全部预算与自有清理归档，见RESULT。
- [ ] 按用户新要求发布paused72 GitHub分支和网页版文献/开源Prompt；不自动恢复实验。

诊断与实现估计1–2h；若触发，完整采集/核验/训练/DEV另6–10h，合计约7–12h；共享争用/失败长尾使墙钟不确定。每个GPU作业另登记实际有限cap，未登记前不启动。小模型训练分钟级，完整机器人episode是主要成本。

初始CPU证据：[登记及结果](../../../投机采样/results/runtime/p0-tf-supervision-alignment-cpu-v1-20261016-r1)。本页维护可见todo，不复制旧全量计划。

用户最新明确本轮截止为10/17 10:00，替代先前09:00用户上限；原活跃DEV的7h/08:11:37预算保持。仅当原预算自然终止且严格清理通过时，允许一次独立接续补固定96的未完整终态任务，GPU09:35停工/09:40清理、CPU09:55汇总。全部已完整成败冻结，原forward/空间累计；见RUNBOOK和USER-1000-DEADLINE-AMENDMENT。双卡AutoDL仅在最终报告评估，不自动租用或重连。
