# 评测范围核查

建议本轮先完整收口Long，不把LIBERO90作为必做依赖。不是“只跑Long就保证可投CVPR”；如果主张跨套件/跨WAM普适性还需要相应证据。其他套件由主线结果和预算触发，不自动扩展。

LIBERO[官方README](https://github.com/Lifelong-Robot-Learning/LIBERO)将90用于预训练、10用于下游终身学习评测。LingBot-VA、CoT-VLA（CVPR2025）与FLASH常见的是Spatial/Object/Goal/Long四套各10任务，不要求补90。Seer（ICLR2025 Oral）的LIBERO部分聚焦Long，但另有CALVIN和真机，不能据此承诺Long唯一实验就足够。

[原文范围与建议](../../../投机采样/results/runtime/p0-libero-eval-scope-literature-cpu1-v1-20261017-r1/RESULT.md)；[可靠90权重检索](../../../投机采样/results/runtime/p0-libero90-checkpoint-reliability-deepsearch-cpu1-v1-20261017-r1/RELIABILITY-RESULT.md)。Long10每任务50初态三臂=1500局；90另13500局，不做90减少计划90%episode数，墙钟比例未知。

这是研究建议和证据口径，不把用户的询问写成已取消90。当前90无可信Target，训练/采集自然BLOCKED。当前四false及旧NO_GO不变。
