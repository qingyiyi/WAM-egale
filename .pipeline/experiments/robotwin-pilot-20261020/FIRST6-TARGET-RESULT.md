# RoboTwin原Target首六局终态

OBSERVED：六个预登记任务valid00（FIT身份），六局完整终态，5成功、1完整算法失败；每局初始物理及观察逐位核对通过，原A100/535、rt/32spp/path8/oidn配置。不是标准多任务完整评测，也不是新独立TEST。

| case | 终态 | 实际命令 | 罚分 | policy chunk | client生命周期秒 |
|---|---|---:|---:|---:|---:|
| place_object_basket-valid00 | 成功 | 240 | 240 | 8 | 184.197 |
| stack_blocks_two-valid00 | 成功 | 304 | 304 | 10 | 200.213 |
| handover_block-valid00 | 成功 | 272 | 272 | 9 | 184.284 |
| lift_pot-valid00 | 成功 | 112 | 112 | 4 | 105.316 |
| press_stapler-valid00 | 成功 | 112 | 112 | 4 | 91.951 |
| open_microwave-valid00 | 完整算法失败 | 1520 | 1500 | 48 | 914.900 |

open_microwave原task_step_limit1500；原完整chunk循环结束时实际发出1520命令，罚分1500独立保留，不删除实际20命令或篡改原循环。该失败不属于此前0动作视频导出PARTIAL，完整失败数据保留。

server累计83 policy chunk、6557次teacher attempted/returned；108条完整512-token BF16 text bank，12 hits、0 misses。仅bank张量逐位，完整任务cached/uncached等价仍NOT_YET_MEASURED。不能把bank应用当作完整等价验证通过。

本first6生命周期1742.194526秒，累计原GPU账本4008.738977/86400秒；原首六局17:46:42截止未重置，作业提前收尾。owned server及6client均退出，remaining_owned=[]，最终两GPU空。server局部JSON仍SERVING是终止前快照，guard终态和现场才是当前状态；账本旧顶层policy/teacher=0是过期累计字段，真实本job计数见server状态，不用0否认实际实验。

UNKNOWN：TF/RoboTwin collector/训练和三臂DEV尚未完成实现，没有自动排队，也无投机效果结果。HYPOTHESIS：迁移接口已具备原Target运行基础，不能据5/6外推套件成功率。RECOMMENDATION：后续先完成RoboTwin动作Q32及真实teacher标签/CFG/cache适配与有限数值门，再沿原任务/预算讨论训练DEV；当前按用户优先级先收口算子。

证据：[终态](first6-terminal-evidence/STATUS.json)、[六局原件目录](first6-terminal-evidence)、[server计数](first6-terminal-evidence/server/STATUS.json)、[原首局失败补跑依据](FIRST-CASE-RETRY.md)。原PARTIAL及专家异常均保留。
