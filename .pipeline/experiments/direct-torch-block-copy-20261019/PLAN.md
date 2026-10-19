# 先完成直接Torch复现

当前范围：固定ca5f37e的custom_sdpa、完整WanAttention及WanTransformerBlock逐字复制；保留Torch/diffusers基元、dtype、layout、cache、输入与RNG语义。只新增可撤销替换入口，不改生产模型/旧tile。

- [x] 逐定义AST/源码核对，原位替换/还原和参数/cache地址核对（CPU）
- [x] 六真实action block0，B2/B14×短中长KV数值/有效状态与配对耗时
- [x] 结果与计时不确定分列，更新HW/main

预算承接08:48:46开始、10:48:46清理的本次M24阶段，不重置起点。此前已118次block，剩余最多4882；本子项最多294次、生命周期≤900秒，总新增≤1GiB、CPU≤2。仅无其他GPU进程时开工，一次尝试不自动重试；deadline前不足300秒不启动。固定三轮ABBA/BAAB/ABBA，每臂5次warmup，每段3次重复。单module、参数与缓存地址相同，在计时外原位切换复制类。uncached原始Torch执行；不加入静态缓存/G1/G2/融合/新kernel。

这是新的直接复制范围，不回写旧A/A>2.5%失败；原矩阵与未运行fixed3 audit均保留。若NVML空载利用率异常，只报告探索性延迟/波动，不签正式稳定性能；收集原始值而不是为PASS修改旧门。数值不逐位一致则立即停止，无容差放宽。原四false保持。

保存六组全体数值、有效KV/元数据、输入/当前RNG、历史GPU gold、每轮原始event/wall延迟；输入原CUDA stride/alias/RNG缺失仍UNKNOWN。只测试完整block重放，非完整机器人任务/30层验证。等待、失败、清理计入原生命周期。

首次登记：2026-10-19T09:51:21.786856+08:00
