# 最近执行

## 2026-10-21T20:09:29.566093+08:00 当前源十阶段核对收口

- 三KV、两进程主表完成；短KV首次升频1095/1290MHz，原记录保留，固定600预热补测，未改GPU频率。
- Cross/FFN独立与整层内差约±1.5%；小投影仍有11%–58%集成增量。SASS支持固定形状编译组织的下一对照，未给spill/同步硬分账。
- 地址枚举与旧NCU写wavefront/冲突完全相符；没有新NCU或数学改动。
- 320输出比较逐位、8组有效状态检查；GPU68.215237秒，累计2373.809395/36000秒；owned退出/GPU0释放。CPU缺cuobjdump失败和短KV条件缺口保留。
- RoboTwin 20:07:50完整89/90，GPU1自动继续、原deadline不变。
- Maintain同主题约束已压缩更新，只保存用户确认的顺序与交付解释要求。

[结果/资源](../../experiments/single-layer-async-gemm-20261020/remaining-gemm-audit-20261021/RESULT.md)，[前完整执行](history/remaining-gemm-audit-before-20261021/EXECUTION.md)。
