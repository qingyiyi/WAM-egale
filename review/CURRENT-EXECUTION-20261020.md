# 当前后继执行结果（不改paused72科学身份）

固定本文件所在commit阅读。旧RESULTS.json/paused72、FAILED、retention与四项科学权限false原样保留。

单层optimized已完成协作epilogue、point/cache任务2048元素，并比较context64/128行（action保持64）。选择context128。两个独立进程中，B14短/中/长完整层相对同轮native耗时比1.866–1.868 / 1.836–1.875 / 1.732–1.737。本轮≤2倍工程门通过，尚未Torch等速；uncached/eager，非强cached/Graph、verifier、chunk或E2E。逐进程绝对延迟波动见原始表。

两种实现各9真实签名/23中间量/18有效KV/重复/输入/RNG/tile有限逐位通过；12条profile各一个compute kernel。共372block，GPU worker生命周期69.379秒；原1497/1500账本不改。本轮所有owned退出、两GPU现场空，无后继自动队列。

[详细结果](../.pipeline/experiments/single-layer-async-gemm-20261020/larger-tile-followup/RESULT.md)、[源码说明](../算子研究/wam_ops/optimized/WHOLE_LAYER.md)、[终态](../.pipeline/experiments/single-layer-async-gemm-20261020/larger-tile-followup/FINAL-AUDIT.json)。仍255register、98848B shared、1608B localSize；阶段grid.sync尚在，旧FFN ready本轮未重验且默认关闭。下一步当前版本profile，再做独立context K/V ready任务填尾波；不把普通原语/tile增大包装成已证MegaTransformer创新。

RoboTwin首六FIT Target仍5成功1完整失败；83policy/6557teacher。actual1520/penalty1500分列，完整任务bank等价未知。TF采集/训练/DEV尚未完成，本轮只核终态，无新增机器人任务。

此前[执行快照](history/larger-tile-followup-pre-20261020/CURRENT-EXECUTION-20261020.md)保留，旧速度不能与本轮直接相除作严格AB。
