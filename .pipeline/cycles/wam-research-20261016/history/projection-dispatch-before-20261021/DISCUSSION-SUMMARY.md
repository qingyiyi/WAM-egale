# 当前讨论与授权

用户要求投影阶段原等待的W1/W2/W7也参与MMA，检查负载不均衡。已直接在optimized完成大tile8warp和固定tile8warp两个对照；均有限逐位通过但完整层退化，所以恢复原4warp并保留证据。没有以更多warp工作替代性能结果；未试小action投影，不外推。

原分工确有阶段性等待，不能直接推导性能损失。大tile同时改变warp与共享内存，固定tile对照排除了shared增大，但具体MMA/shared/local因果耗时仍未量化。后续先定位投影内部成本，不盲目把所有阶段改8warp，也不以调度掩盖原语差距；真实ready调度仍未实现。

沿M39原10h累计；用户允许optimized直接迭代、不另建工作容器。baseline/原tile/投机生产/保护文件及四权限不动。M40固定DEV90已获准接续，GPU1后台自然终态；原24h累计/10月22日01:26截止不重置，不自动重试后续失败。BAAI只在~/cxz工作。

[前讨论](history/projection-warps-before-20261021/DISCUSSION-SUMMARY.md)。

2026-10-21T16:03:04.038104+08:00：用户追问Torch真实CTA/warp、672CTA与108persistent的差别，以及固定/动态warp分工。只读核验原版6份trace与NCU；B14小投影96CTA/4warp、cross K/V672/8、FFN1112/8、FFN296/4。672是总CTA、非同时驻留或kernel launch次数；未做同计算体CTA数消融，不将原版收益归因CTA数量。分工方案仍为讨论建议，未获选择确认。见M39 projection-eight-warps-20261021/TORCH-WARP-ANALYSIS.md。

2026-10-21T16:45:00+08:00：用户“可以的我同意你说的，你先做着”确认同计算体三调度对照；直接在optimized增加诊断入口，原整层选用4warp不变。先精度再三KV独立/同前序/嵌入测量；原累计预算，失败保留。
