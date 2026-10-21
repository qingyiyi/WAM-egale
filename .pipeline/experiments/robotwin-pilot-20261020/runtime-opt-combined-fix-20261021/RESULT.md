# RoboTwin两项工程优化：有限完整任务结果

OBSERVED：两项均实现，单项与修复后的组合均通过同一完整曝光DEV任务的等价核验。组合客户端耗时**243.634→225.782秒，减少7.33%，比值1.079×**。这是评测工程收益，不是投机算法收益，也不是六任务正式性能结论。

## 全部完整对照

固定open_microwave/valid16，原Target canonical infer、同512-token BF16 bank、冻结初态/eval RNG0；同一GPU1，server/client各2CPU，固定顺序每臂一次。模型已常驻；完整客户端含导入/场景初始化/任务执行/外部审计。

|配置|客户端秒|相对原版耗时下降|setup秒|全部get_obs秒|take_action秒|
|---|---:|---:|---:|---:|---:|
|BASELINE|243.634|0.00%|15.150|47.081|66.198|
|FRONT_ONLY|233.767|4.05%|15.289|37.963|66.024|
|LAZY_BATCH_ONLY|237.787|2.40%|10.599|47.003|66.084|
|COMBINED_FIXED|225.782|7.33%|10.466|37.945|65.910|

get_obs可能嵌套在take_action内，不能把列直接相加；完整逐项RPC/命令/时段见ALL-VARIANT-COSTS.csv。四臂均自然成功：96次观察、368提交调用、363真实环境推进、12policy、948teacher。旧actual_commands调用语义与真实推进分开，不改旧罚分/记录。

## 实现与核验

1. FRONT_ONLY：保留原四相机对象的构造（含随机数消耗），仅将未输入模型的front从拍摄/取回枚举移除。受用两腕/head保持顺序、RT/32spp/OIDN、分辨率与图像处理。
2. LAZY_BATCH_ONLY：保留单目标MotionGen构造/预热/求解，延迟未使用batch MotionGen的warmup到第一次plan_batch。构造仍保留；本局两套预热延期、0次触发。不跨episode复用planner，不减少求解迭代或物理步。
3. COMBINED：两项同时开启。所有96次受用观察与物理/RNG摘要、368次命令及每命令状态、全部reset/policy/cache RPC输入输出/RNG/计数、初态与终态，相对原版一致。

持续保存的是摘要和完整命令，未保存所有原始图像/完整KV；不宣称可离线重算所有原始tensor，不将单case逐项一致外推全部任务/任意输入。原固定源码文件未原地修改；进程内隔离helper负责开关。

## 原组合失败保留

原组合因闭包变量original被后续planner初始化引用覆盖，在0teacher/0动作处退出，耗时17.682秒。已保留原PARTIAL及原作业FAILED_OR_PARTIAL，未计作算法失败。修复为独立camera_load引用；CPU对实际install AST做回调核验；用户明确批准本namespace一次补跑，未自动重试，未重置原累计/截止。

## 解释、未知与下一步

- OBSERVED：相机部分主要省get_obs，预热部分主要省setup，模型RPC和动作执行保持相近。仍有约66秒动作执行以及约100秒模型RPC；工程改善有用，但远未消除RoboTwin的主要计算负担。
- OBSERVED：组合两项主要阶段约省13.82秒；客户端总省17.85秒，RPC102.901→102.830秒基本相同。剩余差异没有单独profile，不能全部归于这两个机制。
- UNKNOWN：不同任务/失败长尾上的收益、重复进程波动、TF/PF2下的全部等价，以及批量常驻client的额外空间。固定顺序一次对照不足以给稳定总体百分比。
- RECOMMENDATION：保留组合为可选工程候选，下一步覆盖其他任务与长失败轨迹，再修订1800局吞吐估算。不能据本case直接把原135GPU小时预算乘以该比例，也不自动改全局默认或启动扩大。
- 正式三臂比较须同等使用通过核验的工程条件；四质量权限仍false，新增160GPU小时未批准。

## 资源与证据

原作业781.254秒，本次获准补跑255.945秒，合计1037.199秒GPU生命周期（含准备/失败/清理，非CUDA active）；原M40累计52520.882/86400秒，原10/22 01:26截止不变。两作业owned均清理，无超时；一个实施失败完整保留。

来源：本目录COMBINED-AUDIT.json、STATUS.json、ALL-VARIANT-COSTS.csv、SELECTION.json、RESOURCE-AND-STOP-REPORT.json，以及../runtime-opt-20261021下的原版、单项与失败原件。核心源码engineering.py；补跑授权见PLAN.json。
