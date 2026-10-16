# 共享KV与原torch：完整verifier性能结果

> 2026-10-12T20:54:43+08:00 | 投机采样 | zh-CN | Asia/Shanghai

共享KV完整verifier六组R/S交叉对照完成：原torch中位5.170557秒、共享版3.359750秒，S/R=0.649785，耗时变化-35.02%；稳定正向 pilot。额外显存峰值891.035→257.085MiB。全部348个计时输出与原torch逐位一致，history/metadata/RNG及资源门通过；尚未验证任务成功率或E2E。监督器在已完成后的退出阶段FAIL，原失败保留，测量完成已独立CPU复核；收尾修复10项CPU检查通过，尚未GPU集成。

## 实际结果

| 组 | 执行顺序 | 原torch R（秒） | 共享 S（秒） | S/R | R额外峰值（MiB） | S额外峰值（MiB） |
|---|---|---:|---:|---:|---:|---:|
| 1 | R→S | 5.038685 | 3.233190 | 0.641673 | 891.035 | 257.085 |
| 2 | S→R | 5.154355 | 3.485837 | 0.676290 | 891.035 | 257.085 |
| 3 | R→S | 5.186759 | 3.347632 | 0.645419 | 891.035 | 257.085 |
| 4 | S→R | 5.322236 | 3.297285 | 0.619530 | 891.035 | 257.085 |
| 5 | R→S | 5.314525 | 3.371868 | 0.634463 | 891.035 | 257.085 |
| 6 | S→R | 5.148758 | 3.412535 | 0.662788 | 891.035 | 257.085 |


原R MAD/median=1.486%，共享S=1.715%，预声明门限均≤7%。正式性能测量稳定状态为`PASS`，信号为`positive_pilot`；稳定只是测量门，不等于正向收益。正向pilot要求中位S/R≤0.95且六组均<1，负向pilot要求≥1.05且六组均>1，其他为flat/mixed。

CUDA event整体span中位为R 5170.194ms、S 3359.527ms；span包含主机提交间隙，不是kernel busy。额外peak采用`max_memory_allocated - baseline_allocated`，不是总驻留显存；lease结束后索引和历史引用释放。原两行KV共享访问无需history14/fullKV14物化，但本结果不证明DRAM读量减少七倍。

## 数值、输入及测量口径

金标准一直是原torch SDPA的实际FlashAttention路径。使用相同冻结模型/A_off、T-init3-chunk14、selector/threshold/seed，原cache、text CPU/VAE CUDA/FSDP关闭。新进程capture 52次重新得到同一29变化输入，原29输出hash全部匹配既有gold；没有完整policy terminal或环境episode。

R/S各完整29query预热；六组交叉顺序预先固定，无删除或重跑样本。每臂完整30层/29query的输入staging、lease/setup/keys、写前和返回后fsync journal、release均在原计时内。历史恢复、CPU输出拷贝、字节比较和持久输出保存在计时外。OFF无profiler/归因hook；算法、batch14、CFG2/七候选不改。

每个正式输出仍逐位匹配原torch，独立CPU检查12份实际输出、共2,338,560个元素；原29gold也再次逐字节匹配前轮。每段all30 history/metadata/CPU+CUDA RNG未变，额外peak≤12GiB、free≥8GiB，最后R/S再检查并释放。复用的全30首候选QKV/attention、全29输出、12失效及R/HC/S隔离PASS来自上一轮，所有相关原CUDA/source/binary和adapter字节冻结，本轮没有把这些门删除。

## 条件诊断

六组满足预声明稳定正向pilot条件，因此没有触发额外profile，不用受干扰计时替代正式结果。

## 预算与清理

**监督器原状态为FAIL，保持不变。** 模型20:43:48已经完成六组OFF、finish及`CLOSE`，461意图/461返回/0failed/0inflight；20:43:51最后一条监督采样把唯一已登记的自有PID 2644614列为foreign，触发父进程guard。190条采样仅最后一条有该自有PID，之前所有样本没有foreign或额外GPU PID，最后报错不在性能采样内。

原`FAILED.json`、`STOP_REQUESTED`和源码保留；未向失败job补写`FINAL_RESULT.json`，也未重跑GPU。新的[CPU核验结果](../../../review/OMISSIONS.md)只记录已经完成的模型证据、计数、预算及实际清理。独立CPU逐字节复查全部348输出、2,338,560元素，NVML确认已无进程/已用显存，端口释放，因此本轮性能样本可以单独报告，不能把父监督器写成PASS。

[CPU复现](../../../review/OMISSIONS.md)证明：同一PID/start_ticks/UID/PGID的已退出僵尸进程会清空argv，原ownership guard依赖完整argv会判为foreign。这与现场完成后仅自有PID误报一致；现场没有保存当时完整/proc字段，故不声称已唯一证明哪一个字段改变。新增默认关闭的[退出分类修复](../../../review/OMISSIONS.md)只把经过两次状态/身份复查的确切自有root僵尸态返回为terminal，供后续监督器回收/审计；活进程、PID复用、UID/PGID变化、未知同组进程仍拒绝。10项CPU回归（包括真实自有CPU子进程）通过；尚未接入GPU监督器，不修改本轮测量源码。

用户明确同意累计target上限由900调至1000；回执绑定本轮plan哈希，旧900计划与日志保持原状。新计划最多463target、1worker/1启动/无retry、含加载IO同步清理900GPU秒；累计GPU仍2700秒。实际461次写前意图及相同主机返回、0failed/0inflight、0直接attention；GPU本轮379.375312秒，累计968/1000及1636.324347/2700秒，余32次与1063.675653秒。

性能包装9项CPU检查通过，包括原计时Runtime重放、持久计数、default-off及六组统计失败门；收尾修复另10项通过，合计19项。独立CPU审计2322个既有文件/9个新增包装文件；所有旧失败、公共模型/server/config、TILE来源/结果和受保护state/rules/cycle未改。一个自有GPU进程已经退出，端口20051释放；不向其他进程发信号。

0episode、0EGL、0完整policy chunk、0训练。实现仍默认关闭的私有实验实例。执行和审查为同一主代理，CPU复查使用独立审计脚本，不宣称独立代理复核。

## 下一步及剩余问题

先将已通过10项CPU检查的退出身份修复接入新私有监督包装，不复用或重跑失败job。先设计task0固定条件的小规模闭环，用冻结T/A_off及共享版本比较成功率、接受率、环境步数和E2E。当前回执只覆盖此次verifier测试，0episode上限与累计2700秒保持；闭环须制定新的明确预算后执行。

这轮回答了固定完整verifier的速度、显存和稳定性；尚未证明任务成功率、接受率、环境步数或E2E改善。当前总状态`VERIFIER_POSITIVE_PILOT_TASK_PENDING_PARENT_EXIT_GUARD_FAILED`。

## 复查入口

- [模型测量原结果](../../../投机采样/results/runtime/p0-shared-prefix-torch-perf-v1-20261012-203252/MODEL-RESULT.json)、[父监督器原FAIL](../../../投机采样/results/runtime/p0-shared-prefix-torch-perf-v1-20261012-203252/FAILED.json)、[CPU核验结果](../../../review/OMISSIONS.md)、[六组OFF统计](../../../投机采样/results/runtime/p0-shared-prefix-torch-perf-v1-20261012-203252/OFF-RESULT.json)、[逐字节/预算审计](../../../review/OMISSIONS.md)。
- [冻结计划](../../../投机采样/results/runtime/p0-shared-prefix-torch-perf-v1-20261012-203252/plan.json)、[预算确认回执](../../../投机采样/results/runtime/p0-shared-prefix-torch-perf-v1-20261012-203252/BUDGET-AUTHORIZATION.json)、[CPU门](../../../投机采样/results/runtime/p0-shared-prefix-torch-perf-v1-20261012-203252/preflight/CPU-GATE.json)、[具体方案](../../../投机采样/experiments/shared_prefix_torch_perf_v1/PLAN.md)。
- [条件profile归因](../../../review/OMISSIONS.md)、[原torch数值对齐](SHARED-PREFIX-TORCH-ALIGNMENT-RESULT-20261012.md)。
