# 原 LingBot-VA baseline 对照与 reset/E2E 解释

2026-10-13T17:06:37+08:00（Asia/Shanghai）；只读取既有存档并追加 HW。0 新 episode、0 target forward、0 GPU/CUDA。

## 版本与结论

当前最佳私有版本是 direction 向量投机（K2/B2）＋shared self-attention 历史 KV＋O1/O2/O3。上一轮 1.1010× 是相对已含投机＋共享 KV 的 S0 的增量；原始 LingBot-VA 对照为 T（target-only，不投机、不共享 KV）。这两个分母必须分开。

T 保留原 Torch attention/cache、Long checkpoint、BF16/no-FSDP、CPU 文本编码/GPU VAE，不把它描述成不同部署配置的全体官方 baseline。本次 T 来自 08:40 批次，S0/O3 来自 15:25/16:04 批次；任务0、初态3/4、prompt、seed20260922和模型配置对齐，但批次及并行配对不同，T 对比只属历史参考，尚未在本轮新测 T。

| 初态 | 原 T 总秒 | 上一版 S0 总秒 | 当前 O3 总秒 | T→当前降时 | 环境步数 T/S0/O3 |
|---|---:|---:|---:|---:|---|
| 3 | 360.235 | 356.615 | 311.441 | 13.545% | 293/283/283 |
| 4 | 360.579 | 350.568 | 330.877 | 8.237% | 300/291/291 |

两任务累计 T 720.813 → 当前 642.318 秒，1.1222×，耗时下降 10.890%；环境步数 593→574，减少19步/3.204%。各版均2/2成功，只覆盖同一任务两个初态。初态3少10步/3.413%，初态4少9步/3.000%；chunk数19/19→18/19。时间是两条 episode 耗时之和，不是两个并行worker的批次makespan。

上一版 S0 707.183→当前 642.318 秒，1.1010×，下降9.172%；env_steps 574→574。O1/O2/O3与S0完整输出/实际命令/初终环境/RNG一致，轨迹缩短在原T→投机版本时已出现，不能归功于O1/O2/O3。本轮逐字节验证的历史原Torch K2（D、不共享）是数值oracle，不能与target-only T混淆。

## 为什么局部快而整局比例小

先前所谓“单chunk”部分数据实际是 action verifier replay：共享安装在计时外，输入staging/lease/release在计时内，未覆盖完整视频、动作与环境。新的完整chunk测量包含server.infer，但episode reset在计时外、恢复冻结prompt/KV，仍不含真实环境执行。短/中/长replay下降17.49%/23.42%/34.16%来自固定状态/单worker；短档5/6胜出仍为mixed，不能声称所有KV长度或每个完整chunk稳定同幅加速。

共享KV减少多候选物化/复制历史KV，但当前候选QKV/FFN、cross-attention、视频/VAE、动作输出、CPU同步和环境执行仍存在。7节点验证使用CFG batch14，target-only通常batch2；串行target调用减少不等于每次forward成本一样。direction投机并未把原50次动作scheduler更新删掉。O1复用Python attention适配的AST代码模板（不是verify树/每次重新编译GPU kernel）；O2减少CPU读回；O3固定7行缓冲/批量构造。三个优化不改变接受规则或最终动作。

| 两episode累计项 | 上一版S0秒 | 当前O3秒 | 节省秒 |
|---|---:|---:|---:|
| server reset/T5 RPC | 280.985 | 280.727 | 0.258 |
| 完整policy RPC | 362.840 | 299.832 | 63.008 |
| 观察/KV RPC | 21.582 | 20.241 | 1.341 |
| 环境执行/客户端余项 | 41.777 | 41.519 | 0.258 |
| 完整任务总计 | 707.183 | 642.318 | 64.865 |

policy下降17.365%，但它原来仅占整局51.308%；其余阶段不变时，整局算术降时约8.91%，实测9.17%。节省的63.008秒policy时间实际传递为64.865秒总节省；这里是百分比被固定项稀释。当前reset占总计43.705%。

历史T→当前还存在跨批reset波动：reset 233.115→280.727秒，增加47.611秒；非reset部分487.698→361.592秒，下降25.857%，最终完整任务只下降10.890%。这是观测分解，不证明新增算法导致reset变慢；CPU负载/调度是可能解释，唯一原因未测定。扣除reset的比例只作诊断，不能代替完整E2E。

## reset究竟做什么

调用顺序是环境reset→设置固定初态→5步稳定→一次server reset→循环[policy chunk→执行真实动作→观察/KV更新]。客户端reset在while循环之前，每episode一次；普通chunk不会再次清空历史。见[调用位置](../../../投机采样/scripts/p0_libero_multichunk_client.py)。

[server._reset](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)把frame归0/init_latent置空，清Transformer与streaming VAE缓存，创建空历史KV池，准备动作mask/归一化数据，对本任务prompt编码，设置输出目录并empty_cache。它保持已加载模型权重，不是重新加载模型、生成动作或GPU编译。它使新episode不继承上一次场景的时序记忆；首chunk从空历史开始，随后积累/共享本episode有效KV。初始图像VAE编码在[首chunk _infer](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)，不在reset RPC。

[text编码](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)按512 token padding，在CPU T5编码后把embedding移到GPU；视频CFG启用时[encode_prompt](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)会计算正prompt和空负prompt两个embedding。reset明确传入embeds=None，每episode都重算；得到的embedding在本episode chunk间复用，但prompt/cross-attention投影KV的候选共享仍关闭。

“reset基本没变”严格指S0/当前两初态合计280.985→280.727秒，仅差0.258秒；单初态分别有约±9.5秒波动，并非每次reset完全一样，也不适用于历史T。每局135–145秒是整个server reset RPC，不能全归因于清KV或环境reset；T5是源码支持的重点嫌疑，但T5、clear/alloc与RPC各自占比尚未独立测量。模型构造/加载和环境构造另列，不应称作reset。

## 后续固定对照要求

用户原话：“以后对比除了和上一版对比，同时也要和这个baseline对比可以吗？”

每轮保留原T baseline＋上一版＋当轮候选三臂；冻结baseline实现/配置，但同轮重测其时间。不能永久沿用本次历史720.813秒作为分母。保持checkpoint、任务/初态/seed、动作/视频步数、设备/精度、EGL/并行负载相同并平衡顺序；报完整E2E、分项耗时、成功率、环境步数及chunk数，分别给两种加速比。历史数据只能标注参考。

接续原短/中KV完整chunk计划时加入这两个固定比较口径；原T target-only完整chunk与K2 replay原Torch参考应分开。再按同轮三臂重测E2E；reset/T5先拆分计时确认瓶颈，随后再决定缓存prompt embedding等优化，保持数值与任务结果门。轨迹控制仍按原计划留待性能归因和覆盖后。此次仅解释/记录，不启动新GPU或修改实现。

[逐项CPU计算及原始存档hash](../../../review/OMISSIONS.md)；[上轮完整episode报告](OPTIMIZED-E2E-O4-RESULT-20261013.md)。
