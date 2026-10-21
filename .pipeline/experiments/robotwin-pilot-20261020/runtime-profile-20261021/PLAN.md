# 完整RoboTwin执行成本诊断

单个已曝光microwave valid16原Target完整局，保存cProfile并与之前两条原始路径诊断中该局比较所有命令与初态/RNG。并非重试失败或新增独立样本。不得缩短episode/修改成功判据/改变renderer或物理步长。

使用原M40剩余预算，GPU1生命周期最多1800秒、整局1200秒、最多8000teacher，新增512MiB，原10/22 01:26硬截止保留。只在前两条诊断完全结束、owned清理后启动。source/server数学同原Target，cProfile不属于正式计时；原日志90局账本作为无本轮profile的总体成本。

拆分：原server reset/policy/cache、client进程启动、setup_demo、双臂planner、take_action/physics、get_obs/render/图像取回、close/落盘。cProfile嵌套累计时间不相加伪造总计；单独标inclusive/exclusive。终态/cleanup见STATUS.json。
