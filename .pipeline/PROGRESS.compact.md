# WAM 短状态

更新：2026-10-25T21:45:23.600887+08:00。动态状态以实际job/status为准。[当前计划](cycles/wam-research-20261016/PLAN.md)、[当前完整进度](cycles/wam-research-20261016/PROGRESS.md)、[现行约束](memory/CURRENT-CONSTRAINTS.md)。

## 当前资源与截止

- **5090算子**：S版FFN1分项核验完成，4296次完整block调用、保留S，GPU已清理。主体tile约95.6%在GEMM加载/计算/内部等待包络；GELU/写回非主项。原Torch中长KV实际448CTA×4warp/158regs；S120任务/170CTA、255regs。硬件stall/HBM仍未知。[新结果与完整阶段表](experiments/sm120-ffn-next-20261024/FFN1-GEMM-GAP-20261025/REVIEW_RETURN.md)。
- **5090质量**：固定篮子valid49完成T/OLD/Q75/Q90四臂，成功0/0/1/1，严格同卡初态通过。仅1组暴露DEV，未定tau；完整下一局预算不足后停止，未延长原21:56:53截止。已归档212文件并删除5090临时权重/环境/资产，算子目录保留。[有限结果](experiments/suite-acceptance-calibration-20261025/rtx5090-robotwin-20261025/RESULT.md)。
- **双A100 RoboTwin**：20:48实查，598/600完整三臂组、1794/1800局，T/C/P成功487/485/491，125.250/160GPU小时。两worker均`NO_UNCLAIMED_PAIRS`正常退出；旧281/406 PARTIAL缺口保留，不自动重试。[快照](experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/ROBOTWIN-SNAPSHOT.json)。
- **本机Goal**：resume05已闭合，81attempt、54完整均成功、4合格四臂组；未配齐/INVALID保留。resume06的初态复现诊断因可信本地PyTorch加载兼容缺项退出，0teacher/动作，未取得新初态，后续迁移5090。本机无自有GPU作业，20:55有他人GPU任务。

## 结论、限制与下一步

M39当前S：FFN1＋GELU写global ACT后grid同步，再FFN2七段BF16递推。B14/Q16三真实KV、有限逐位状态通过；本轮同场原Torch/S为1.284/1.791/2.237×，双方warmCrossKV/eager/noGraph，**非机器人E2E**。S的阶段历史FFN1约0.80×Torch、FFN2约1.03×。

W、共享B、共享A、A2及旧DSM8/C4/C2均未采用，负结果保留。A2虽把48次交接减到24次，源码中也失去step+2装载重叠，整层比旧共享A又慢3.44%–4.35%。没有NCU，不能将差值全部归因DSM带宽或宣称HBM节省。已按用户确认完成S的FFN1成本诊断；下一步对齐实际Torch主循环，不扩大DSM设计。

M42五套83030行CAL来自CONT主池，不是当前ON on-policy。Goal4组覆盖3task，全部4臂成功，不能据此定最终阈值；26旧初态无效的物理状态一致、仅图像3–24通道差1–2灰阶，因果/影响未定，不放宽旧门。Long旧tau来自task8/init0 B1局部误差q90，不是当前TF闭环最优证明。固定OLD/Q75/Q90完整DEV、PFoff/full512 BF16 reset，耗时仅探索性。

M41五环境44组三臂/132主表局已收口：Long E2E约1.121/1.062×；Goal/Robo新增失败、长尾和Robo首刷新前分叉FAILED保留。[分母与账本](experiments/rtx5090-spec-speed-20261025/PERFORMANCE-RESULT.md)。

`engineering_quality_screen=false`；`independent_test_complete=false`；`noninferiority_established=false`；`formal_performance_permission=false`。原FAILED/retention/M17配对与FIT-DEV暴露不变，不启动独立TEST、正式45chunk或无限训练。

本轮迁移DEV与算子诊断均收口；下一步核对实际Torch主循环与嵌入成本、整理Robo队列两缺口和新旧Target差异；不擅自补跑或扩大质量面板。10/28主表、11/01 18:00北京时间中文证据包仍为规划目标，10/22–23不补称完成。中文逐段[Prompt](prompts/WAM-PAPER-COLLABORATION.md)。H02已核验释放120.619GiB，临时目录`/rjs/cxz/tmp`，原tile/失败保留；[HW精简历史](maintenance/hw-compact-20261025/PLAN.md)。
