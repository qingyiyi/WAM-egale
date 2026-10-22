# 当前研究进度（2026-10-22T17:44:23.726236+08:00）

M39：FFN1+GELU、FFN2+残差3写回融合完成；9真实签名/36结构有限逐位与状态通过。两进程warm完整单层Graph，相对原Torch短/中/长KV为1.1646/1.6893/2.7445×，但比自身融合前慢1.09%/1.14%/0.83%。当前采用为流水研究入口，未获得融合提速；仍阶段同步，无MMA/epilogue重叠。warm3400/3472/3400逻辑任务、29grid。

[结果](experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/RESULT.md)、[阶段表](experiments/single-layer-async-gemm-20261020/ffn-epilogue-fusion-20261022/STAGE-MAP.md)、[优化索引](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。首CPU编译失败和旧源码/DSO保留；本机仅采用远端验证二进制，未另测。

2026-10-22T17:37:30+08:00 owned全部退出，保守累计8948.557/36000秒；原本轮18:12:32截止不变。四科学权限false；baseline/投机/原tile/保护文件不动。下一步先定位两处融合开销，再讨论流水，不自动开新作业。

M17四套LIBERO6000局齐，非独立TEST，暴露/首两S/长尾缺口保留。M40原DEV90完成、10/22 01:26旧截止过期，1800局/160h未获准。10/22 go/no-go、10/23身份冻结、10/28主表、11/01 18:00中文证据包；证据不足收缩claim。

[PLAN](cycles/wam-research-20261016/PLAN.md)、[完整状态](cycles/wam-research-20261016/PROGRESS.md)。
