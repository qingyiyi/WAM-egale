# RoboTwin TF64迁移适配

新目录只适配第二环境，不修改现有LIBERO投机源、权重、normalizer或阈值。

当前实现：32 token，每token按官方通道顺序16 state＋16 previous velocity＋3 scheduler标量；hidden64/4heads/FFN256/单pre-norm层、学习位置、双16维center±delta残差头，56416参数。推理沿原adapter的FP32 normalizer/model/decode，方向返回输入BF16；不把BF16输出说成全部模型权重BF16。

官方通道顺序是`[0..6,28,7..13,29]`，不是排序后的0..15。完整teacher噪声仍30×2×16=960元素；compact512不改变teacher RNG。

该代码尚无RoboTwin训练权重、真实部署CAL或闭环结果；四项权限false。采集/训练/选择/接受器按本轮[执行计划](../../../.pipeline/cycles/wam-research-20261016/ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)与M40记录推进；未接入机器人服务。
