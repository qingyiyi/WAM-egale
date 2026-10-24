# 5090 Long阶段结果

固定task2..9/init49，8组完整T/TF/PF2配对；三臂均8/8成功。既定task0/1中两条已执行C在0环境动作处因迁移/本轮观察器问题PARTIAL，全部保留且未重跑；原默认FSDP task0/T单列。task1只有T/P，作为补充，不混进共同三臂N8。不是500局、独立TEST或TF质量放行。

| 指标：原Target/对应投机成本 | TF连续 | TF+PF2 |
|---|---:|---:|
| 动作有效scheduler更新（Σaction / Σg） | 1.2328× | 1.1465× |
| 完整policy平均/chunk（自然历史分布） | 1.1444× | 1.0942× |
| 暖任务（初始化env后至原自然终态） | 1.1333× | 1.0690× |
| 客户端全生命周期（含imports/env构建/关闭） | 1.1207× | 1.0616× |

| 双方成功环境步数：投机/Target | TF连续 | TF+PF2 |
|---|---:|---:|
| P50 | 1.0173 | 1.0043 |
| P90 | 1.0269 | 1.0302 |
| max | 1.0314 | 1.0360 |

同卡RTX5090、Torch2.9.0+cu130、eager/no Graph、既有LIBERO noFSDP单worker强基线，共同full512 BF16 reset bank；Target权重、TF_ON、normalizer、tau/selector和50更新均冻结。单层MegaTransformer未集成。本轮coarse同步/event与原chunk日志仍有观察成本，只作工程计时，旧formal_performance_permission保持false。

有效g来自真实提交计数，不把反事实scheduler调用或7候选当合法推进。完整policy与自然任务均实际运行，无截断；没有用target调用数推速度。所有失败/准备时间另列，不拿更快失败当加速。两种任务E2E边界分开，不含权重网络传输/一次模型冷装载。

CPU原始数据审计：432条完整policy、21600次有效提交，route/SST、50更新/terminal、7/14行及提交分区全部通过有限日志核对；17组保存的reset前CPU/CUDA RNG逐值一致；8个C/P case的前两个S共16对保存的分数/路径一致。未保存完整KV/每个velocity，不能声称全输入/全状态逐位一致或独立重算teacher cost。

Long全部结果已native rsync回传，文件数/尺寸核验ACK后删除远端临时Target10179092850B（约9.480GiB）；见LONG-RETURN-ACK.json、LONG-WEIGHT-DELETION.json。其余公共资产只在后续任务使用期间驻留，最终整目录清理待完成。

原始：raw/long/server-05/requests.jsonl、raw/long/panel-06/STATUS.json及逐局episodes.jsonl/initial audit；汇总RESULTS.json、EPISODES.csv、AUDIT-RESULT.json。四科学权限全部false，旧FAILED/retention/预算/默认入口未改。
