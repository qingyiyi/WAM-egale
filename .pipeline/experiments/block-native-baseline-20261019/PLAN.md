# 完整原生block交付

1. **已完成：固定完整边界和上游原语。** 固定Torch原语/许可证，归档旧baseline；数学无ATen高层转调，cuBLASLt依赖公开。
2. **已完成：顺序native action baseline。** 六真实block每点23关键量/输出与六原语逐位过；12个write1→write2→read0状态门过，含long驱逐。新FFN2 cuBLASLt过，旧CUTLASS失败不改。
3. **已完成：两份代码接入和全模型。** Q128与24连续写读门、optimized12状态门过；最终六action query实际30层输出/全部有效KV/输入/RNG逐位过。追加保留原模型839个requires_grad=True标志的18次模型调用也过，不改参数地址。原视频缺字段和首次混合dtype拒绝保留。
4. **已完成：同口径性能。** 最终vector版两进程各9点，504条原值与A/A齐；baseline/optimized所有点中位不慢于两个原Torch对照。A/A最大8.63%偏差保留；不是最强shared-prefix/Graph、完整chunk或E2E。

终态与源码：[RESULT.md](RESULT.md)、[CODE-GUIDE.md](CODE-GUIDE.md)。23个GPU子作业1088.541/3600生命周期秒、全部owned退出。288模型/5114独立block/144原语；旧失败和预算不重置。本轮GPU工作结束。

CPU≤2，GPU作业生命周期累计≤3600秒，绝对墙钟≤8小时，新存储≤1GiB，详见REGISTRATION.json。共享GPU仅做已授权精度，正式计时仍自然空闲。生产投机入口不切换；用户允许改算子baseline，投机/原tile仍禁止修改。
