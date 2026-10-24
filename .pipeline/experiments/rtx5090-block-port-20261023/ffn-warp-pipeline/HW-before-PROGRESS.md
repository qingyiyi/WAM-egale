---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-23T22:25:11.629514+08:00'
plan: PLAN.md
current: M39-SM120-TMA
next: 与用户核对当前M64结果；后续优先剩余FFN2和三个小投影，双A100原队列继续。
---
# 当前进度

5090本轮有界两候选已收口。TMA三KV逐位通过，但完整层相对A退化5.0%–5.8%，不采用；原同名扩展两次误载计时判无效、原件和预算保留。当前研究采用GRAIN-M64-FFN2：输出tile从48个128×128变96个64×128，保留四MMA warp、七段K2048及BF16提交次序。

两独立进程同轮相对A完整层延迟降18.34%/17.00%/15.91%；原Torch/自有短中长1.0864/1.5430/1.9498倍，仍比强guarded慢4.50%/5.36%/7.83%。三方全eager无Graph，冷CrossKV单列。当前融合FFN对Torch约0.75倍；三个小投影约0.68–0.70倍，余下瓶颈仍未消除。

三真实B14/Q16、block0 fixture的26中间量＋输出逐位/有效KV/输入/RNG通过；两进程前后重复、实际public默认与非默认流均通过。仅有限研究入口，四科学权限及production_admitted=false。源码/独立二进制/结果已回本机并byte核对，A保留；owned自然退出、5090显存0。完整block共22104/26000次，含7356次无效计时；保守本轮34分钟，累计27466.413/36000秒，余额2.370小时。不自动使用余额。

A100双卡继续原RoboTwin队列，22:22只读见两模型context，CPU巡检继续；本轮不重算SR、不改队列。20:58的148组/444局只是历史快照。

下一步建议先隔离剩余FFN2计算与七段提交成本，再处理三个小投影；不自动组合第三候选、扩DAG或修改数值标准。当前15计算阶段＋布局/退出，全表按实际融合边界显示，插桩分段不冒充可加的整层时间。

[回传](../../experiments/rtx5090-block-port-20261023/blackwell-tma/REVIEW_RETURN.md)、[完整表](../../experiments/rtx5090-block-port-20261023/blackwell-tma/STAGE-TABLE.md)、[固定源码入口](../../experiments/rtx5090-block-port-20261023/blackwell-tma/SELECTED.json)、[原状态](../../experiments/rtx5090-block-port-20261023/blackwell-tma/HW-before-PROGRESS.md)。
