# 当前研究进度（2026-10-22T18:16:14.446638+08:00）

M39四组FFN依赖流水R2完成：9签名/36结构有限逐位及状态通过。两进程相对原Torchwarm完整单层Graph，短/中/长KV为1.1646/1.6808/2.7386×；比同轮此前阶段版延迟降0.342%/0.357%/0.332%，仅小幅正向信号。debug确认部分FFN2在FFN1全部完成前启动；最后组串行依赖仍在。ACT保留，数学tile3400/3472/3400不变，warm grid29→27。

[结果](experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/RESULT.md)、[阶段表](experiments/single-layer-async-gemm-20261020/ffn-four-group-pipeline-20261022/STAGE-MAP.md)、[优化索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。R1额外串行化负结果保留；R2目录错误0调用FAILED保留，原R1二进制完整恢复后同截止新retry核验。两机研究入口采用R2，本机未另测。

18:10:40 owned全部退出；本轮保守896.121秒，累计9844.678/36000秒，原18:40:44截止不变。下一步先评估最后组关键路径与同步成本，未开新GPU作业。四科学权限false，baseline/投机/原tile/保护文件不改。

M17四套LIBERO6000局齐，非独立TEST，首两S/暴露/长尾缺口保留。M40原DEV90完成，旧10/22 01:26截止过期；1800局/160h未获准。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。

[PLAN](cycles/wam-research-20261016/PLAN.md)、[完整状态](cycles/wam-research-20261016/PROGRESS.md)。
