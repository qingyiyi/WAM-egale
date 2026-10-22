# 执行检查点

## 2026-10-22T18:16:14.446638+08:00 M39-FFN-PIPE完成与清理

R1与R2分别通过9签名/36结构有限逐位与状态核验。R2保留两K分段并行、仅最终提交等待；两进程原Torch同缓存整层比值1.1646/1.6808/2.7386×，比同轮此前阶段版快约0.33%–0.36%，小幅信号，实际应用显著性未知。debug四组计数完整、提前消费实证，最后生产/消费组均3。

R2首次build-lib目录错误致0调用FAILED。保留原失败；R2产物正确安放，R1从原对象恢复且SHA256与首次测量一致，之后同截止新retry完成，不改数学代码。18:10:40自有14身份均退出；GPU生命周期136.105秒，保守896.121秒，累计9844.678/36000秒。

两机研究入口采用R2，本机未另测；ACT保留、warm数学任务不变、grid29→27。四科学权限false。[结果](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESULT.md)、[资源](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESOURCE-AND-STOP-REPORT.json)。


## 2026-10-22T18:04:28.022111+08:00 M39-FFN-PIPE有界执行

用户批准FFN1到FFN2四组依赖流水。R1已完成9签名/36结构有限逐位与状态核验、两进程三KV计时；debug确认组0/2提前消费，但完整层略慢。R1人为将第二段MMA也等到第一段提交，已在R2改为仅提交等待，并让96任务全被领取后的空闲CTA进入出口屏障。R2正在CPU编译，未采用新DSO；R1保留。

每CTA先完成全部静态FFN1生产任务，最多48个第二段CTA持有shared部分和等待；固定108常驻CTA下其余CTA仍可推进第一段。两版各200种抽象时序模拟无死锁；实际内存/数值由GPU门验证，不以模拟替代。

[JOB](../../experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/JOB.json)：本轮45分钟，截止2026-10-22T18:40:44.193684+08:00，累计起点8948.557秒、总36000秒；R1 GPU生命周期65.188秒计入本轮。只用原Torch外部性能baseline，previous/R1仅作内部回归控制。baseline/投机/原tile/保护文件不改。


## 2026-10-22T17:44:23.726236+08:00 M39-FFN-FUSE完成并清理

直接在optimized完成FFN1+GELU、FFN2+残差3写回融合；保留BF16边界。9签名81调用与36结构144调用有限逐位/有效KV/RNG通过。两独立计时进程各5646完整层调用，相对原Torch1.1646/1.6893/2.7445×；融合本身退化1.09%/1.14%/0.83%，没有新阶段profile或warp重叠结论。

17:37:30 owned进程全部退出；GPU生命周期65.147秒，保守全窗口598.644秒，累计8948.557/36000秒。首CPU编译失败及原截止保留，r2未重置预算。两机研究入口采用已验证DSO，原版本可恢复；本机未另测。四科学权限false。

[结果](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESULT.md)、[资源](../../experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESOURCE-AND-STOP-REPORT.json)。修复当前Cycle派生导航的frontmatter和稳定ID表，不改保护文件或历史证据。

## 此前检查点（原文保留）

# 最近执行检查点（2026-10-22T16:37:20.064075+08:00）

1. 用户明确批准预计算后公平对照与完整逐阶段表，Maintain已保存；M39原累计预算不增。
2. 直接修改optimized，加入context_kv，warm跳过两投影与K RMS；baseline用外部适配器同等缓存。
3. 两项CPU构建准备失败已修复并保留；正常版编译后9签名/27结构/三KV缓存baseline逐位与状态通过。
4. 两独立进程七臂计时完成，warm自有0.957/1.124/1.140ms；强guarded仍快7.94%–11.70%。
5. 首profile因插桩遗漏入口/出口计数退出；原FAILED保留。补齐同两处计时helper，修复诊断版在同一截止内完成；正常GPU计时未重跑。
6. 两机研究入口采用；完整阶段、扰动和grid等待分别归档。最终owned清理，累计8349.913/36000秒。

[结果](../../experiments/single-layer-async-gemm-20261020/cross-kv-precompute-20261022/RESULT.md)，原预算/失败/四科学权限保留。

7. 2026-10-22T16:53:49.249001+08:00 用户指出聊天表遗漏tile数量及工作量；已按当前源码补齐原STAGE-TABLE/STAGE-MAP，新增TILE-WORKLOAD.csv。区分FFN2的48输出/96分段、attention两4warp组与内部循环及实际尾块；仅CPU文档核查，无GPU消耗，原计时/失败/版本不改。

8. 按最新反馈新增STAGE-SPEEDUP.md：同轮完整层相对原Torch及强guarded比值；逐阶段可比参照缺失明确标注。仅CPU汇总，无GPU启动，Maintain更新展示要求。

9. 用户将后续对照改为仅原Torch。CPU核对GELU元素/tile数、既有原Torch grid，以及FFN epilogue融合的BF16边界和K归约依赖；已记录只读分析，无GPU消耗、无计算源码修改。

10. 新授权FFN写回融合已实现；JOB登记45分钟、18:12:32截止、原累计8349.913秒。首CPU编译因bp变量遮蔽失败（0 GPU作业），改为显式buffer指针后r2同截止继续。旧运行和源身份保留；先9签名/36结构，再原Torch三KV两进程整层对照。
