# RTX5090 LIBERO Goal有限完整任务观察

OBSERVED：10任务各init49，T/TF/PF2共30条完整自然终态；当前各自冻结权重/normalizer/tau，已有强noFSDP引擎、full512 BF16 reset bank、eager/no Graph，未接入单层自研算子。

| 指标 | Target | TF | TF+PF2 |
|---|---:|---:|---:|
| 成功 | 10/10 | 9/10 | 10/10 |
| action成本/有效更新加速 | 1× | 1.1612× | 1.1374× |
| 完整policy/chunk加速 | 1× | 1.1007× | 1.0933× |
| 全部完整尝试暖任务E2E | 1× | 0.8069× | 1.1583× |
| 全部完整尝试client生命周期E2E | 1× | 0.8281× | 1.1231× |
| 双方成功client生命周期E2E | — | 1.1856×（9局） | 1.1231×（10局） |
| 双方成功有效步比P50/P90/max | — | .9854/1.0267/1.0333 | .9996/1.0603/1.2026 |
| 全任务800失败罚分相对Target变化 | — | +500 | -113 |

OBSERVED：task9/init49 T成功153有效步/10chunks，TF失败796实际命令/50chunks/罚分800，PF2成功184步/12chunks。TF该局client生命周期160.134秒，T40.091秒、PF244.081秒。原实际796与罚分800分列，不补造四条命令。TF其余9局成功，不能用成功子集1.1856×替代包含失败的0.8281×。

OBSERVED：279完整policy/13950合法scheduler更新，有限记录路由/计数门通过；20对reset前RNG记录exact，10case的首两S保存分数/路由/计数均exact。初态物理和双相机检查由原client新wrapper在首动作前执行。没有保存全部latent/KV/teacher velocity，不能称全数值等价或根因证明。

RECOMMENDATION：记录一次本轮真实C失败/P成功的观察；PF2调度有继续核验价值，但仍是暴露DEV上单初态/每任务的有限工程面板，不是500组成功率、独立TEST或非劣证明，不能外推至RoboTwin或更改全局preset。四科学权限仍全部false。

证据：RESULTS.json、EPISODES.csv、AUDIT-RESULT.json、raw/goal/server-01及panel-01。回传GOAL-RETURN-ACK.json与临时Target删除GOAL-WEIGHT-DELETION.json；源输入预分配中断/续传单列，不修改原失败或预算起点。
