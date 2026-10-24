# RTX5090 RoboTwin：六组完整自然任务的成本观察

OBSERVED。固定六任务、每任务 valid49、Target/同一冻结 TF/TF+PF2 共18条均完整终态。使用原 FSDP2、video25/action50、Q32、冻结阈值0.007879490032792091、完整512-token BF16 reset bank、eager/no Graph；未接入自研单层算子。不是600组质量评估或独立TEST。

| 指标 | Target | TF连续投机 | TF+PF2 |
|---|---:|---:|---:|
| 成功 | 5/6 | 4/6 | 4/6 |
| 实际有效环境步总数 | 2521 | 3226 | 3010 |
| 原客户端提交命令总数 | 2592 | 3328 | 3104 |
| 原登记失败罚分总和 | 2572 | 3288 | 3064 |
| 按有效环境步另列的罚分 | 2521 | 3226 | 3010 |
| 完整policy chunk | 84 | 107 | 100 |
| action秒/合法scheduler更新 | 0.085104 | 0.048684 | 0.060164 |
| 完整policy秒/chunk | 6.583603 | 4.749288 | 5.328394 |
| 动作成本加速 | 1× | 1.7481× | 1.4145× |
| 完整policy/chunk加速 | 1× | 1.3862× | 1.2356× |
| 全部完整尝试客户端生命周期E2E | 1× | 0.9327× | 0.9359× |
| 暖自然任务E2E | 1× | 0.9211× | 0.9232× |
| 双方成功4局的生命周期E2E | 1× | 0.9274× | 1.0611× |
| 双方成功有效步数比P50/P90/max | — | 1.0054/2.2855/2.8333 | 1.0054/1.1400/1.1970 |

生命周期为模型server已常驻时的新client进程启动到退出；含导入、场景/规划器建立、reset、policy/cache、环境执行与关闭，不含首次模型装载/迁移。暖口径为首次reset RPC入口至首次env.close入口，排除client/场景初始化。不同口径不互换。

action主成本包含逐轮host范围、完整action的准备/终端/GPU排空残差和路由开销，除以全部合法推进g。三项加和与完整action观测闭合；不是单个异步round的host时间或GPU active时间。原始round与残差分别见RESULTS.json/EPISODES.csv。T含终端g=0的工作，不能删去它把51个调用凑成50。

OBSERVED。篮子任务T成功、C/P失败；微波炉三臂均1500有效步失败，原始提交1520另列；其余四任务三臂成功。TF订书机374步、Target132步，为2.8333倍成功长尾。更少的单位动作计算并未抵消新增失败、更多chunk及环境/初始化成本，不能称完整任务加速成立。

OBSERVED。291个完整policy共14550合法更新，S/S/T路由、50更新、7候选/14 CFG行与终端调用的有限记录检查通过；12组reset前CPU/CUDA等RNG记录exact。首16个真实Q32/history240共享attention调用与同输入原Torch packed参考逐位相同、maxabs0，仅限这些调用。原runtime的这16次inline比较发生于首个共享路径使用者，参考计算/同步/落盘计入该局，未单独测量或事后扣除；正式性能需要将数值审计与计时分开。

**PF2单因素隔离不成立。** C/P首个S的保存分数/路由和16条提交命令在六case都相同；第二个S在五case已有分数或动作差异，首个差异均在提交命令index16，早于第三chunk的PF2刷新。前48条命令最大绝对差约0.0055–0.0083；stack_blocks_two为0。见AUDIT-RESULT.json、ROBOTWIN-PREFIX-DIFFERENCE.json。不能把后续C/P差别纯归因于PF2，也不把相同初态/RNG当成后续全状态相同的证明。

UNKNOWN。第二个S分歧的具体根因未定位；缺完整中间观测/KV/velocity，不能追溯独立重算全部数值路径。模拟器、GPU数值或缓存等仍只是可能方向，未新增根因实验。历史A100同DEV身份的篮子T失败/P成功，本5090却相反；A100-SAME-DEV-BACKGROUND.json只作跨平台结果背景，不混算速度或成功率。

RECOMMENDATION。将本表保留为实际执行成本与质量信号，不推荐据此晋级TF/PF2；后续要做PF2因果结论，应先补分叉前隔离诊断，不靠重跑挑通过局。本轮不改权重/阈值、不自动重试或追加任务。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention不改。

原始入口：raw/robotwin/server-02、raw/robotwin/panel-03；逐局EPISODES.csv和RESULTS.json；回传ROBOTWIN-RETURN-ACK.json。核验后已删除临时Target资产10179092783B，见ROBOTWIN-WEIGHT-DELETION.json；最终整目录清理另有收据。
