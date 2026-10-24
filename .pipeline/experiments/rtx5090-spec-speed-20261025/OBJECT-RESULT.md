# RTX5090 LIBERO Object有限完整任务观察

OBSERVED：10任务各init49，T/TF/PF2共30条完整自然终态，三臂均10/10成功。冻结Target/TF权重、normalizer、原tau与SST调度；full512 BF16 reset bank、已有noFSDP、eager/no Graph，未接入自研单层算子。

| 指标 | Target | TF | TF+PF2 |
|---|---:|---:|---:|
| 成功 | 10/10 | 10/10 | 10/10 |
| action成本/有效更新加速 | 1× | 1.2032× | 1.1309× |
| 完整policy/chunk加速 | 1× | 1.1309× | 1.0870× |
| 全尝试暖任务E2E | 1× | 1.0823× | .9896× |
| 全尝试client生命周期E2E | 1× | 1.0751× | .9898× |
| 有效步比P50/P90/max | — | .9997/1.0643/1.2214 | .9926/1.2510/1.8931 |
| 全任务罚分相对Target变化 | — | +15 | +119 |

OBSERVED：task5/init49三臂都成功，T131步/36.075秒client生命周期，TF160步/40.070秒，PF2248步/56.100秒。PF2长尾达到1.8931倍，P90也超过旧1.20工程条件。不能用PF2中位数接近1掩盖它；本面板中PF2额外target刷新与长度成本抵消了局部计算收益。

OBSERVED：server01在加载bank时因原A100绝对符号链接失效而退出，0teacher/0episode。保存readlink和原失败后，native rsync-aL复制原46144386B bank、fsync，新server02/panel02才开始首条真实任务；没有已执行episode重跑或缓存值重算。bank dtype/512形状与checkpoint绑定沿原加载断言，未删断言。

OBSERVED：完整policy与路由/计数检查、20对reset前RNG及首两S保存分数/计数/路由exact，见AUDIT-RESULT.json。全KV/velocity/输出未全部保存；有限记录通过不证明所有输入。四科学字段保持false，未将本10组描述成500组质量测试。

证据：RESULTS.json、EPISODES.csv、raw/object/server-02及panel-02；原server01/panel01失败保留；OBJECT-BANK-PRE-REPAIR.json与transfer-object-bank-repair.json记录修复；OBJECT-RETURN-ACK.json、OBJECT-WEIGHT-DELETION.json和最终整目录删除收据。
