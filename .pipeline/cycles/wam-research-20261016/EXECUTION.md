# 执行检查点

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
