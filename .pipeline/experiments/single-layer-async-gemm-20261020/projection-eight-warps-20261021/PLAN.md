# M39投影8warp候选

目标：Cross K/V把4计算warp改为8，输出tile128×128→256×128，WarpTile仍64×64×64，三级流水/数学K顺序不变；不改变FFN/FA2。所有8warp既加载/MMA也写回，thread0仅在tile边界领取任务。不能仅放宽compute谓词。

优先有限逐位及全部有效状态，再两独立进程三KV native/before/candidate；无稳定整层收益或数值门失败则回退。证据仅有限单层，不改变四权限。
