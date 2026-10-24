# 当前四MMA分工的 result_empty 有界诊断

状态：CLOSED_DIAGNOSTIC_ONLY；一次构建与完整block诊断完成，自有GPU进程已退出。属于现有M39，不新增训练、架构或生产候选。

- [x] 从当前采用源码生成独立观察副本，当前正常源码/二进制保持。
- [x] 单次构建；真实短/中/长KV完整block数值、有效KV、输入和RNG核对。
- [x] 比较原正常/观察副本OFF/ON完整层扰动，采集四个MMA与三个epilogue的时间戳。
- [x] 解释TMA搬运边界，保存结果与完整阶段表，关闭自有进程。

预算：原累计35566.41251785547/36000秒；本次最多420秒GPU生命周期，含远端准备、编译、运行、回传和清理，累计不重置。CPU准备不占卡；单GPU单worker，CPU≤4。新增持久化≤512MiB，完整block≤500次，构建最多1次，无自动重试或算法修改。T0在已核对可用资源后设置，并登记绝对清理deadline；预留60秒回传/清理，GPU任务最晚T0+360秒结束。

输入：已有block0/B14/Q16真实fixture，KV288/1152/2160；共同warm Cross K/V。基线是当前WARP-SPECIALIZED-4MMA独立二进制。数学、tile、十段K/BF16顺序、四MMA/一TMA/三epilogue、所有屏障不改。

只在诊断副本采集globaltimer：MMA的result_empty前后和结果store/发布后；epilogue的result_full前后及计算/释放后。每次观察tile编号模12的一个子集，12次覆盖96个tile，重复3次。四MMA leader、三epilogue leader分别记录，初段无empty等待作为读钟控制。时间是有扰动的软件观察区间，不是硬件stall、单纯内存或Tensor Core时间；分量不能相加解释完整层。完整层OFF相对正常、ON相对OFF扰动任一KV超过5%则标LIMITED_PERTURBATION，不据其决定扩warp；局部扰动无法单独排除时明示限制。

不增加warp，不实现TMA store，不改当前采用版本。失败/timeout原样保留，预算不足停止，未知不补造。四质量/正式性能字段及production_admitted保持false。结束打印并保存当前17阶段全表；各阶段旧比值明确历史，本次只新测正常/OFF/ON完整层与局部观察。
