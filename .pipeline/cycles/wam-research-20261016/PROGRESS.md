---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-24T14:01:25.329383+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前进度

M39采用SELF-QKV-TMA-4MMA：Self Q/K/V迁移现有W0 TMA、W1–3 epilogue、W4–7 MMA；原288输出任务、64×128 tile、9×K320+K192十段、BF16合并与K16顺序保持，其他数学/FFN不改。两个独立进程三KV完整层均改善，相对旧A延迟下降2.942%/2.808%/2.249%；同轮原Torch/新版1.2475/1.7553/2.1872×，Guarded/新版1.1021/1.0850/1.0433×。全eager、无Graph、同CrossKV缓存，单层重放而非机器人E2E。

三份真实B14/Q16/block0、KV288/1152/2160的27中间量/输出、有效KV、输入、RNG、六投影计数有限逐位通过；公开默认及非默认流均通过。正常静态资源170CTA/每SM1CTA、100352B shared、255寄存器、784B localSize/thread不变，localSize不是动态访存计数。正常整层与独立插桩阶段分列；完整17行表本轮重测并在对话打印，FFN融合一行。同步补充：TMA函数内部3次CTA同步，领取/完成各1次，成功tile显式共5次，另有末次空领取；不是只有3次。

用户已明确取消后续现有RTX5090的原十小时累计上限。原35926.41251785547秒账本、旧截止/失败不改；本job20分钟内完成，含等待/回传生命周期434.383秒、保守480秒，跟踪累计36406.413秒但不再受旧上限阻塞。2871/12000完整block，两个构建、五GPU进程均成功，无timeout。源码/DSO六文件已回本机byte一致，无SHA；自有9个PID+出生身份和进程组均退出，5090 compute context清零，实例未关闭。每个后续作业仍单独有界，不能据此租新卡或扩大科学权限。

M40双A100持续原RoboTwin固定队列。13:55:53 CPU巡检RUNNING_FIXED_QUEUE，298/600有效完整三臂组（894/1800局）、2组执行中、299未领取；两worker均运行、无新HALT。原STATUS.tmp竞态与pair281 C中断保留，尾部一次补齐已登记，跨namespace严格核对通过才计有效组。完成条件600有效组，不是worker退出；原160GPU小时/10天预算不变，当前5090追加不扩张A100预算。

未知：新算子未重验其他层、在线接受、完整模型及机器人任务，未获取5090硬件计数器。融合FFN仍慢于Torch约15%–16%，三个96tile小投影仍慢约16%–19%；下一优先项是有限归因，不预设更多warp、更大tile或完整DAG一定更快。M17旧证据缺口、所有FAILED保留，engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission与production_admitted均false。

[本轮结果](../../experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/REVIEW_RETURN.md)；[本轮完整阶段表](../../experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/STAGE-TABLE.md)；[固定源码及结果入口](../../experiments/rtx5090-block-port-20261023/projection-warp-specialized/self-qkv-tma-migration/SELECTED.json)。

[前次主循环诊断](../../experiments/rtx5090-block-port-20261023/projection-warp-specialized/mma-mainloop-diagnostic/REVIEW_RETURN.md)；[双卡接续与失败恢复](../../experiments/robotwin-pilot-20261020/eval100-20261022/resume-after-storage-race-20261024/PLAN.md)。
