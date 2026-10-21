# 当前执行证据（2026-10-21T21:17:23.588246+08:00）

**六小投影固定形状阶段已保留：完整B14单层三KV、两进程均减少1.88%–2.08%。** 仍是一层一个kernel，tile/数学/claim/sync不变；不是新DAG。Self-Q慢约6–7µs，新独立版本也慢于旧独立，故只称部分改善。

- [完整结果](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/RESULT.md)、[完整层逐格表](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/WHOLE-LAYER-COMPARISON.csv)、[旧新阶段对照](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/STAGE-BEFORE-AFTER.csv)。新短/中/长约2.513/2.719/2.969ms；没有混入旧Torch计时推新倍数。
- [有限精度门](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/SIMPLE-GATE-RESULT.json)：9真实签名输出/23中间量/18有效KV转移及输入/RNG逐位；240阶段比较通过。不是任意输入/30层/闭环质量证明。
- [源码](../算子研究/wam_ops/optimized/csrc/whole_layer.cu)156–175行为`action_projection`与路由；源e91e09eb、DSO23238b37两机一致。baseline、原tile、投机生产不动。
- [资源](../.pipeline/experiments/single-layer-async-gemm-20261020/small-projection-function-20261021/RESOURCE-AND-STOP-REPORT.json)：GPU109.306秒，累计2483.116/36000秒；一个候选已结束，owned退出。Self-Q编译边界→FFN原语→DAG为后续顺序；没有自动第二候选。

**RoboTwin90/90完成**：6任务×5初态(valid14–18)×3臂，各30局，T25、TF连续24、TF+PF2 23。30组C/P初态/首两S审计通过，失败保留。不是每任务1初态，也不是与LIBERO每任务50初态相同规模。只属于迁移DEV，非完整RoboTwin/独立TEST，不支持质量保留主张。[终态结果](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/TERMINAL-RESULT.md)、[逐对数据](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/TERMINAL-SUMMARY.json)、[清理](../.pipeline/experiments/robotwin-pilot-20261020/dev90-20261021-r2/TERMINAL-CLEANUP-CHECK.json)。累计50006.380/86400秒，原deadline不变；两GPU自有作业均结束。

将六任务扩到各50初态共900局仅为建议，需明确独立分区和有界预算，未启动。已曝光5局不能改称独立TEST，且Target microwave本身1/5须核查。

四科学权限false；旧paused72/retention/FAILED及review/RESULTS.json保持历史身份。原始输入/DSO/大JSONL留服务器，网页镜像不能用于声称独立重算未公开张量。前派生快照commit `7bf2a4b4c0031220049408505ecc42e29759f3b8`。
