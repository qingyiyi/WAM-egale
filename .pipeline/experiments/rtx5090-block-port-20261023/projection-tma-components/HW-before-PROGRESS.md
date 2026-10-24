---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-24T10:09:30.649000+08:00'
plan: PLAN.md
current: M39
next: M39
---
# 当前进度

连续TMA输入单候选已完成并不采用：两个独立进程完整层相对A慢1.384%/1.125%/1.065%。三投影保持96输出任务、8MMA warp；双K64输入槽跨十个数值段，原BF16合并不变。26中间量＋输出/KV/输入/RNG及每tile48A/48B/48等待/10提交有限通过。共享内存仅增256B至100352B，驻留及静态寄存器/local属性不变；实际重叠比例与各开销贡献未知。后处理warp重叠尚未实现。

当前实际保留SPLITK-M64-FFN2，五源文件/public入口已恢复；默认/非默认流通过，六份源/DSO回本机byte一致。当前原Torch/完整层1.1512/1.6240/2.0417×，guarded短快1.485%、中慢0.300%、长慢2.979%。当前17阶段本轮重测，候选另存完整表；全eager无Graph、同缓存，阶段诊断不冒充可加完整层。

两个构建、六GPU进程成功，无失败/timeout；11469完整block，性能拒绝1个。16owned身份退出、5090显存0，实例未关机。保守计15分钟，累计33286.413/36000秒，余45.2分钟，旧预算/截止保留。四科学权限与production_admitted=false。

下一步建议先分量核对TMA输入等待、局部同步与后处理，再决定输出交接；未启动新候选。双A10010:02只读见原RoboTwin模型/模拟与巡检正常，不改队列、不重算成功率。

[结果](../../experiments/rtx5090-block-port-20261023/projection-tma-continuous/REVIEW_RETURN.md)、[当前全表](../../experiments/rtx5090-block-port-20261023/projection-tma-continuous/STAGE-TABLE.md)、[固定入口](../../experiments/rtx5090-block-port-20261023/projection-tma-continuous/SELECTED.json)。
