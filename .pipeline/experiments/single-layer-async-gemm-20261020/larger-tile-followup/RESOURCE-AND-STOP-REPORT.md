# 本轮资源与止损

新登记[REGISTRATION](REGISTRATION.json)，原旧1497/1500历史账本未改。本轮GPU0完整block372/420，GPU worker生命周期69.379秒/900秒；CUDA active时间未单独汇总。两次CPU构建分别118.401和120.085秒。guard生命周期308.327秒；登记起点19:08:53到最后worker结束19:20:36约703.6秒，含实现/等待，低于3600秒，早于原20:20:28硬截止。

2个CPU build、6个GPU worker均returncode0并由guard wait回收；无失败/timeout、无自动重试、无robot新增episode。已登记剩余48block与GPU余额未继续使用，不自动变成新实验。

远端stage在worker结束时40,423,596字节，约38.55MiB，含构建/私有DSO/原source备份/trace，低于512MiB。后续小报告和source核对见[FINAL-AUDIT](FINAL-AUDIT.json)。不删除旧失败、模型或机器人证据。

最终选context128，remote optimized DSO与该已测二进制逐字节比较相同；源码默认构建值设128，原数学不变。新默认仍不接生产模型，不修改旧投机配置。自动队列已结束，GPU释放。

本轮止损点：不继续第三tile；不使用剩余48次拼不完整的ready验收；不把单层≤2倍当速度对齐/最终加速；不自动扩大RoboTwin或更改旧截止。阶段全局barrier、255寄存器和spill仍待下一轮定位。
