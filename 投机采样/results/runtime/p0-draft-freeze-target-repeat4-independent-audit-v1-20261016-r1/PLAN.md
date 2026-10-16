# 今日首四条完整 target 重复任务审核

事前固定今日manifest首4条：T/task2、task8/init33，各rep0/1。只读取这4条已完成完整原horizon任务与T-READY/原plan元数据，不读TF、MLP或fresh/risk DEV结果，不改任何今日job/173源/规则/HW。

先用原真实episode/coverage helper核每局commands、初终保存状态/初图hash、RESET、COUNTER、FINISH与全部实际actions/latents/obs_data文件及finite。常驻T server尚在工作，不能把arm未退出当这4条未完成，也不伪称整臂/50终态通过。

按task比较rep0→rep1的初始物理保存字段、两相机原字节、RESET的目标CPU/CUDA RNGfingerprint、完整commands、success/env_steps；再按COUNTER真实完整policy/cache序列比较共同actions/latents/逐步obs，记录最早可见差异与未配对尾部。动作/观测时间按实际action[7,4,4]及每动作一帧绑定，不能再误写每4动作一帧。

这次同一常驻target server内重复与旧跨r3/r4差异分别描述，不混成额外独立质量样本。不能从像素差推因果、改原像素门或把有限相同重复写成保证。逐步完整sim内部状态/每请求RNG/KV、完整Python/NumPy RNG对象仍按实际保存边界报告UNKNOWN。

预算：CPU≤2、墙钟5分钟、新产物≤4MiB、GPU/SSH/forward/episode/训练0。只有新审核目录写文件；原FAILED与质量/正式性能许可均不变。
