# 最近执行检查点

## 2026-10-19T12:07:24.496573+08:00：本机有限复测完成，远端四worker自动接续

新授权计时接续独立登记。两个进程自然exit0，六真实block/六实际30层、write1/2及输出/有效KV/输入/当前RNG/安装还原共108检查逐位过；432full30/444block，GPU生命周期131.338秒，自有GPU进程退出。102次资源采样未见外来PID。完整模型baseline/原版0.9539–1.0058，optimized0.9535–1.0218；严格±2.5%非全点过，空卡NVML24–25%异常未解释，正式无争用仍UNKNOWN。B2 full30为root CFG派生非顺序轨迹；1875投机源/配置复核无变化。原三轮失败和预算保留。[全表](../../experiments/operator-stable-timing-20261019/RESULT.md)。

用户新要求远端两卡各并行并确认自动提交后聚焦本机算子。经PID出生/实际GPU UUID/设备FD/teacher返回/NVML唯一上下文及集合delta核对，在旧CPU dispatcher休眠检查点交接运维；现有model/client/guard均未暂停。四worker独立评估、端口/CPU/输出隔离，约28.5GiB/卡。匹配Target200完整；12:05去重435/1500、77组三臂齐，剩余固定0_99/100_199自动排队。原M17预算/截止不改；旧CPU交接信号和非自然退出另存收据。[证据](../../experiments/remote-eval-parallel-20261019/RESULT.md)。

四项false、原FAILED/retention/default A_off保持。下一步讨论block分解与单子图，只改optimized。更早四文见[本次归档](history/M25-before-stable-timing-20261019/EXECUTION.md)。

## 2026-10-19T12:45:00.444350+08:00：网页方案评估

已重读网页原方案及固定ca5f37e入口，形成[待执行计划](../../experiments/operator-stable-timing-20261019/NEXT-PLAN.md)。本轮仅CPU审查，未启动新GPU；建议不记作已采纳长期决定。
