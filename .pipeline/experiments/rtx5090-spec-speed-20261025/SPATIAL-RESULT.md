# RTX5090 LIBERO Spatial有限完整任务观察

OBSERVED：10任务各init49，三臂共30条完整自然终态；同冻结Target/TF/PF2、full512 BF16 bank、已有noFSDP引擎、eager/no Graph，未接入自研单层算子。

| 指标 | Target | TF | TF+PF2 |
|---|---:|---:|---:|
| 成功 | 9/10 | 9/10 | 9/10 |
| action成本/有效更新加速 | 1× | 1.1753× | 1.1167× |
| 完整policy/chunk加速 | 1× | 1.1126× | 1.0777× |
| 全部完整尝试暖任务E2E | 1× | 1.0766× | 1.0168× |
| 全部完整尝试client生命周期E2E | 1× | 1.0645× | 1.0131× |
| 双方成功9局client生命周期E2E | — | 1.0602× | .9929× |
| 双方成功有效步比P50/P90/max | — | 1/1.0232/1.0252 | 1/1.1502/1.6975 |
| 全任务800失败罚分相对Target变化 | — | -2 | +74 |

OBSERVED：task4三臂都自然失败，实际796命令、罚分800分列；不得把相同horizon的失败计为成功任务。task7/init49三臂成功，T119步、TF122步、PF2202步，即PF2成功长尾1.6975倍，超过旧max1.30工程条件；TF总体最大1.0252。PF2客户端E2E仅约1.3%改善且缺独立重复，不声明统计显著。成功吞吐已计入失败尝试成本；原始逐局表见EPISODES.csv。

OBSERVED：339完整policy/16950合法更新，有限路由/计数检查通过，20对reset前RNG以及首两S保存分数/计数/路由exact；全KV/velocity/输出未全部持久化，不能作普遍数值等价证明。

RECOMMENDATION：本有限面板中TF在成本和长度上优于PF2；不据此修改全局默认、旧证书或独立TEST权限。四科学字段均false。本轮已有受控输入预传并行，正式性能仍需独立隔离与重复。

证据：RESULTS.json、EPISODES.csv、AUDIT-RESULT.json、raw/spatial/server-01及panel-01；SPATIAL-RETURN-ACK.json和SPATIAL-WEIGHT-DELETION.json。临时Target在核验回传后删除，整目录清理另列最终收据。
