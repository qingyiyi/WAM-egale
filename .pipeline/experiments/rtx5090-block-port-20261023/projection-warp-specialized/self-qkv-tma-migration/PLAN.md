# Self Q/K/V迁移当前局部warp流水

本轮来源为用户2026-10-24明确批准迁移与追加现有RTX 5090持续使用。保留旧累计35926.41251785547秒账本；原十小时上限仅对后续5090作业被新授权替代，旧终态不改。双A100的RoboTwin仍按原160GPU小时/10天固定队列推进。

- 目标：仅将Self Q/K/V从八MMA warp backend改为当前已用于三个小投影的一TMA／三epilogue／四MMA backend。
- 输入：父目录`INPUT-MANIFEST.json`和既有三份真实B14/Q16/block0、KV288/1152/2160；不造合成任务、不读取新模型、不计算SHA。
- 保持：Q/K/V独立权重/输出、共享288任务队列、64×128输出tile、K9×320+192的十次BF16归并、每段K64/K16顺序；FFN、其他投影、attention及外部阶段同步不改。
- 唯一附带接口变化：TMA descriptor和debug计数从三个投影扩为六个，原前三个编号不改，Self Q/K/V为3/4/5；M1写入发布到async proxy。
- 基线：冻结当前研究采用A源码与独立DSO；原Torch、已有Guarded、A、候选四臂，同context缓存、同eager/no Graph，交错顺序、两独立进程。
- 验收顺序：所有三KV的27项中间量/输出、输入、全部有效KV、RNG、计数有限逐位；然后无插桩整层；阶段插桩只作定位。若通过且三KV完整层重复稳定获益，再更新optimized公开入口并验证默认/非默认流。
- 负结果：精度不符即停性能；完整层无稳定增益则保留A。最多一次明确的实现修复，不扩大architecture/warp/tile网格。
- 本job预算：GPU生命周期20分钟（含两次完整kernel编译、等待、失败、回传及90秒清理余量），单worker、4CPU、新增存储≤1GiB、完整block调用≤12000。公开入口验证仍使用此job原截止，不重置。
- 终态：`CLOSED_ADOPTED_RESEARCH_ONLY`或`CLOSED_REJECTED`；基础设施/超时为`PARTIAL`并保留真实失败，不自动重试。
- 清理：仅记录并核对自有PID+出生时间，确认退出和5090 compute context清零；实例保持开启。源码、独立DSO、结果与完整阶段表回同步本机。

执行清单：

- [x] 冻结A并生成单变量差异。
- [x] 原Torch三KV完整精度门、角色计数。
- [x] 两进程四臂整层与当前候选完整阶段表。
- [x] 采用或拒绝；公开入口两流（采用时）。
- [x] 原件回传、清理、HW更新与完整表聊天回报。

四项科学门和`production_admitted`全为false，本次单层重放不提供闭环证书。

终态：CLOSED_ADOPTED_RESEARCH_ONLY。详见REVIEW_RETURN.md。
