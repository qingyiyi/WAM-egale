# WAM 短状态

更新：2026-10-25T20:55:39.677397+08:00。动态状态以实际job/status为准。[当前计划](cycles/wam-research-20261016/PLAN.md)、[当前完整进度](cycles/wam-research-20261016/PROGRESS.md)、[现行约束](memory/CURRENT-CONSTRAINTS.md)。

## 当前资源与截止

- **5090算子**：A2共享A双K64交接已完成，4842次block/11.62分钟生命周期，精度通过但三KV整层比S慢25.92%–31.73%；保留S。owned清理、迁移IO恢复已闭合。[结果/18阶段](experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/REVIEW_RETURN.md)。
- **5090质量迁移**：用户改为RoboTwin阈值优先。迁移已进入完整DEV：首个篮子valid49 Target完整失败，OLD接续，尚无完整四臂组。实际单server峰值reserved约22.13GiB，另有模拟器；采用单worker。沿原本机接续窗口**21:56:53**硬截止，不能自动加6小时；每局1800秒＋600秒清理余量。原5090单worker实测占用20.8GiB，不能在32GiB上盲开双worker。[登记](experiments/suite-acceptance-calibration-20261025/rtx5090-robotwin-20261025/PLAN.md)。
- **双A100 RoboTwin**：20:48实查，598/600完整三臂组、1794/1800局，T/C/P成功487/485/491，125.250/160GPU小时。两worker均`NO_UNCLAIMED_PAIRS`正常退出；旧281/406 PARTIAL缺口保留，不自动重试。[快照](experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/ROBOTWIN-SNAPSHOT.json)。
- **本机Goal**：resume05已闭合，81attempt、54完整均成功、4合格四臂组；未配齐/INVALID保留。resume06的初态复现诊断因可信本地PyTorch加载兼容缺项退出，0teacher/动作，未取得新初态，后续迁移5090。本机无自有GPU作业，20:55有他人GPU任务。

## 结论、限制与下一步

M39当前S：FFN1＋GELU写global ACT后grid同步，再FFN2七段BF16递推。B14/Q16三真实KV、有限逐位状态通过；本轮同场原Torch/S为1.286/1.789/2.229×，双方warmCrossKV/eager/noGraph，**非机器人E2E**。S的阶段历史FFN1约0.80×Torch、FFN2约1.03×。

W、共享B、共享A、A2及旧DSM8/C4/C2均未采用，负结果保留。A2虽把48次交接减到24次，源码中也失去step+2装载重叠，整层比旧共享A又慢3.44%–4.35%。没有NCU，不能将差值全部归因DSM带宽或宣称HBM节省。建议先对齐S的FFN1 GEMM，未启动新搜索。

M42五套83030行CAL来自CONT主池，不是当前ON on-policy。Goal4组覆盖3task，全部4臂成功，不能据此定最终阈值；26旧初态无效的物理状态一致、仅图像3–24通道差1–2灰阶，因果/影响未定，不放宽旧门。Long旧tau来自task8/init0 B1局部误差q90，不是当前TF闭环最优证明。固定OLD/Q75/Q90完整DEV、PFoff/full512 BF16 reset，耗时仅探索性。

M41五环境44组三臂/132主表局已收口：Long E2E约1.121/1.062×；Goal/Robo新增失败、长尾和Robo首刷新前分叉FAILED保留。[分母与账本](experiments/rtx5090-spec-speed-20261025/PERFORMANCE-RESULT.md)。

`engineering_quality_screen=false`；`independent_test_complete=false`；`noninferiority_established=false`；`formal_performance_permission=false`。原FAILED/retention/M17配对与FIT-DEV暴露不变，不启动独立TEST、正式45chunk或无限训练。

先完成现有RoboTwin迁移及剩余预算内完整DEV，归档全量Robo队列与两缺口；算子本轮已收口。10/28主表、11/01 18:00北京时间中文证据包仍为规划目标，10/22–23不补称完成。中文逐段[Prompt](prompts/WAM-PAPER-COLLABORATION.md)。H02已核验释放120.619GiB，临时目录`/rjs/cxz/tmp`，原tile/失败保留；[HW精简历史](maintenance/hw-compact-20261025/PLAN.md)。
