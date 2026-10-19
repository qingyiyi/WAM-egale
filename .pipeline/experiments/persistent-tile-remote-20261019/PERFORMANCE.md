# M32：单层完整block探索计时

同一真实输入、相同有效KV恢复，GPU0无其他可见context；另一GPU继续Object，CPU4/5。每进程七入口一次检查warmup、三轮旋转顺序、每次完整动态block；含KV管理/打包/同步。event与host值均保留。这里只展示静态与persistent同开FFN1的对照，七入口完整值见[CSV](BLOCK-TIMING.csv)及原始JSON。Q128不启用FFN1融合。

| 单层输入 | 静态 ms（进程1 / 2） | persistent ms（进程1 / 2） | 延迟变化（1 / 2） |
|---|---:|---:|---:|
| short-B2-action | 1.3998 / 1.3978 | 1.4377 / 1.4336 | +2.71% / +2.56% |
| middle-B2-action | 1.5258 / 1.5278 | 1.5677 / 1.5698 | +2.75% / +2.75% |
| long-B2-action | 1.6466 / 1.6486 | 1.6927 / 1.6937 | +2.80% / +2.73% |
| short-B14-action | 2.5948 / 2.5948 | 2.7116 / 2.7095 | +4.50% / +4.42% |
| middle-B14-action | 2.6982 / 2.8160 | 2.8262 / 2.9594 | +4.74% / +5.09% |
| long-B14-action | 3.1427 / 2.9655 | 3.3393 / 3.1293 | +6.26% / +5.52% |
| short-B2-video | 1.4817 / 1.4817 | 1.4858 / 1.4920 | +0.28% / +0.69% |
| middle-B2-video | 1.5759 / 1.5759 | 1.5841 / 1.5892 | +0.52% / +0.84% |
| long-B2-video | 1.7142 / 1.7080 | 1.7265 / 1.7265 | +0.72% / +1.08% |

新增persistent在本次两进程均慢于对应静态路径，B14约4.42%–6.26%。这不是完整模型/任务E2E，也不是最强B14 shared-prefix/O3/Graph。未证明稳定加速，不推广为全部形状定理。

原Torch A/A最大绝对差 10.07%（进程1短B2 action），不删除这个点；进程间B14中/长的绝对值也波动。三样本中位的精度有限，不能把小差值解释为确定收益。所有计时返回输出仍逐位核对。

计时attempt1因NVML宿主PID与容器PID不同在0次block调用时退出，原件保留。新守卫要求初始化前GPU空、初始化后唯一context、全过程context集合不变；这个现场关联不等于已证明宿主PID映射。两有效进程为attempt2/3，不挑选最快轮。

原始：[进程1](remote-evidence/jobs/block-timing-2/results/RESULT.json)、[进程2](remote-evidence/jobs/block-timing-3/results/RESULT.json)、[失败attempt](remote-evidence/jobs/block-timing-1/results/RESULT.json)。
