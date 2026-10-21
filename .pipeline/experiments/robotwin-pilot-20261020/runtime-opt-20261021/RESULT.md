# RoboTwin工程优化：两项通过，组合补跑待批准

OBSERVED：同一曝光DEV open_microwave/valid16，原Target、512token BF16 bank、冻结初态/RNG0，单GPU1、server/client各2核。三个完整自然成功任务与一个0动作实施失败。不是新独立质量样本或正式投机性能表。

|版本|终态|完整客户端秒|相对原版耗时|实际环境推进|policy chunks|
|---|---|---:|---:|---:|---:|
|BASELINE|COMPLETE_SUCCESS|243.634|+0.00%|363|12|
|FRONT_ONLY|COMPLETE_SUCCESS|233.767|-4.05%|363|12|
|LAZY_BATCH_ONLY|COMPLETE_SUCCESS|237.787|-2.40%|363|12|
|COMBINED|PARTIAL_OR_INVALID|17.682|不比较|0|0|

## 两项已观察收益

- FRONT_ONLY：保留相机构造和实体，只从拍摄/取回枚举去掉未使用front；受用三路顺序/参数不变。观察总耗时47.081→37.963秒（约−19.37%）；完整客户端243.634→233.767秒（约−4.05%）。
- LAZY_BATCH_ONLY：保留MotionGen构造、单目标预热与求解；只将未使用batch warmup推迟到首次plan_batch。两臂各延期一次，本局0次实际batch调用；setup15.150→10.599秒（约−30.04%）；客户端237.787秒（约−2.40%）。
- 每个成功臂96次观察、368次提交调用、363次环境推进、12个policy；全部受用观察/物理/RNG逐次摘要、完整命令、全部server RPC输入输出/RNG/counts和终态与BASELINE相同。摘要匹配不是可离线逐字节重放全部原始图像/KV；full30及TF质量不借此放行。
- 同轮一个固定case、固定顺序各一次，完整客户端包含导入/初始化/外部审计，模型已常驻；单项数值是有限工程观察，不外推六任务和失败长尾。不将两项百分比直接相加当成组合收益。

## 组合实施失败与修复

COMBINED在0 teacher/0命令处TypeError，17.682秒。install中Camera.load_camera闭包捕获变量original，后续同名变量被赋为planner初始化；调用相机时转到错误函数。错误属于包装器实现，不是算法失败。原PARTIAL和预算保留，guard清理完整。

已在相邻runtime-opt-combined-fix-20261021修复为独立camera_load引用，CPU用实际install AST及无GPU类替身核对组合回调和首次batch触发。真实组合尚未重跑，执行旗标PENDING；原无自动重试规则已向用户申请单局补跑。不能宣称组合已通过或已有组合加速比。

## 建议与边界

两单项值得保留为工程候选；先补组合，再覆盖其他任务/长失败轨迹，最后修订1800局吞吐预算。当前无需调整阈值、物理、成功判据、相机质量或模型权重。原160GPU小时未批准，扩大未启动，四质量权限false。正式比较T/TF/PF2时必须同等使用获准工程优化。

耗时阶段存在嵌套（get_obs可能在take_action内调用），不把各列直接相加。原因与旧profile相符：动作规划/模型仍占大头，两项改进不能声称消除了全部慢因。

来源：EPISODE-COSTS.csv、FRONT_ONLY-AUDIT.json、LAZY_BATCH_ONLY-AUDIT.json、STATUS.json、各case审计JSONL与server/ALL-RPC-AUDIT.jsonl；源码engineering.py/client.py/server.py/guard.py及EXECUTED-SOURCE.json。
