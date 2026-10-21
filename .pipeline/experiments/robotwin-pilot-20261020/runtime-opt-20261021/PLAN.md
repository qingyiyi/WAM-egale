# 两项工程优化：有界完整任务核验

用户已授权实施，仍属M40原累计24h、10/22 01:26硬截止。单GPU1，server/client各2CPU，最多90分钟生命周期/4个完整episode，每局1200秒，留180秒清理，32000teacher、1GiB新增证据。不启动1800局，不做训练、少步数或改阈值。

固定open_microwave valid16、同冻结初态/原Target/full512bank/评估RNG0。顺序baseline、只front、只batch预热延迟；仅两单项均通过才运行组合。每个attempt只一次，无自动重试；独立变体不等于重跑失败臂。不同variant可自然完整终止，实施错误停止该臂并保留，不用更快失败算加速。

初态保留全物理与三路受用相机；front只从观察返回中省略，不创建伪造图像。全局记录每次受用obs及RNG、物理状态、全部RPC输入输出/RNG与teacher计数、命令和终态。有限同case通过不证明所有任务等价，不计独立质量N。耗时为相同外部审计下的完整客户端/各阶段，初次一轮仅工程信号。

不改固定官方文件：隔离helper修改本进程Camera.load_camera后的static相机枚举，以及CuroboPlanner中batch warmup调用为首次plan_batch触发；实际单目标planner配置/求解路径不变。若不一致不集成。
