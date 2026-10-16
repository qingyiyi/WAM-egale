# 6episode原生诊断完成

完成：2026-10-11T08:23:31.062407+08:00；投机采样，非tile。3臂串行，每臂2独立EGL worker、init3/4各1完整episode。没有teacher采集、训练、RL、target/verifier/gate/bonus修改或自动成本复测。

| 臂 | init | success | policy步数 | soup最终In | sauce最终In | 与旧命令首次差异步 |
|---|---|---|---:|---|---|---|
| T | 3 | True | 293 | True | True | 1 |
| T | 4 | True | 299 | True | True | 77 |
| S0 | 3 | True | 298 | True | True | 1 |
| S0 | 4 | True | 310 | True | True | 77 |
| B_on | 3 | True | 280 | True | True | 1 |
| B_on | 4 | True | 309 | True | True | 77 |

6episode全部原生记录/连续步数/goal合取/native success/terminal RGB hash通过。native_steps.csv为对象pose、EEF、夹爪qpos、左右指接触与各In谓词；episode_diagnostics.csv仅描述统计，contact不等于持稳，In是区域包含。诊断耗时不是clean E2E，不据此晋级。

## 复现边界

LINGBOT_SEED只在server启动时设置，reset不重设RNG。旧init3在init2后运行，本次init3是新worker首局，未恢复旧CUDA RNG/模型状态；同init/启动seed不保证复现旧失败。各新臂相同worker历史按本次合同比较；若旧失败未复现，报告为未复现，不删除数据、不自动增加episode。完整WAM observer on/off数值等价仍未证明。

## 下一步

人工/CPU对齐contact、object pose、grip qpos、In事件与命令/图像时间线，区分抓取未建立、抓后失去接触、入篮未满足及返工。先检查数据/解释有效性，不将未经验证的持稳标签当reward，不直接RL/大小模型联训。之后另行设计T/A_off同负载成本计数对照；本任务不自动启动。
