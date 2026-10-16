# O4：O1+O2+O3 完整 LIBERO episode 端到端结果

2026-10-13T16:20:41+08:00（Asia/Shanghai）

**两初态均有端到端收益，保留 O1+O2+O3 为当前最佳私有版本。** 同一固定方案的旧 S0 与 O1+O2+O3 的两个完整任务累计耗时 707.183 → 642.318 秒，1.1010×，下降 +9.172%。这是 libero_10/task0 初态3、4各一次的配对 pilot，不能外推所有任务或稳定总体收益。

| 初态 | 旧 S0 总秒 | O1+O2+O3 总秒 | 加速比 | 耗时下降 | 旧/新环境步数 |
|---|---:|---:|---:|---:|---:|
| 3 | 356.615 | 311.441 | 1.1451× | +12.668% | 283 / 283 |
| 4 | 350.568 | 330.877 | 1.0595× | +5.617% | 291 / 291 |

两臂成功分别 2/2、2/2，动作/轨迹长度完全相同；没有通过缩短任务或截断换取耗时下降。

## 耗时分解

| 初态 | 臂 | reset/T5 秒 | 全 policy 请求秒 | 观察/KV请求秒 | 环境执行/客户端余项秒 | outer 总秒 |
|---|---|---:|---:|---:|---:|---:|
| 3 | S0 | 145.215 | 180.393 | 10.743 | 20.265 | 364.851 |
| 3 | O3 | 135.517 | 145.337 | 9.644 | 20.942 | 319.741 |
| 4 | S0 | 135.770 | 182.447 | 10.839 | 21.512 | 358.899 |
| 4 | O3 | 145.209 | 154.495 | 10.597 | 20.576 | 339.156 |

新组合的 policy 请求累计下降 +17.365%、减少 +63.008 秒；reset累计差 +0.258 秒。policy请求包含 video、draft、verify、接受/回退、输出拷贝和 RPC 传输；不把全部请求差额归因为单一 GPU kernel。客户端余项是总耗时减所有模型RPC，含真实环境执行等，未做受扰动 profiler 分摊。

主指标为原 client elapsed_sec：包含reset、prompt/T5、draft、验证、回退、动作提交和环境执行，保留原完整800步horizon。outer额外含环境构造与关闭。原server构造/模型加载、进程准备的Python模板、任务返回后的async保存drain不在task计时内，仍计入GPU预算。O2/O3 helper模板在服务准备阶段生成；O1前缀模板首次生成仍计入首个policy请求。S0每chunk原安装逻辑包含于policy时间；新臂保持新请求绑定与缓存生命周期，热chunk无AST重建。

## 精度、覆盖与运行安排

第一批两独立EGL worker：S0/init3、O3/init4；第二批交换：O3/init3、S0/init4。固定同checkpoint、训练draft A_off、seed20260922、global selector、threshold0.007839680526450805、video20/action50、CFG、VAE设备；profile/full trace/teacher关闭。prompt与cross-KV未共享，rowproof保持原规则。

新增CPU六组BF16完整50步helper对照覆盖首块frame0和后续frame4、全拒绝/全接受/生产阈值；真实tiny生产30层模型，30通道输出、全部KV、scheduler/RNG和接受计数一致。tiny CPU reset夹具硬编码float32 KV，首轮在0 GPU时拒绝；只修正可抛弃夹具的KV dtype，失败日志保留。

独立CPU读取本轮完整原始存档：74对action/video张量、927072元素、574条真实执行命令，以及80对初终环境/controller状态数组逐字节一致；初始观察、环境XML/init、终态与客户端RNG一致，全部target调用分布、接受深度、输出与完整任务步数一致。另外独立CPU读取历史原Torch K2（未共享）完整两局：本轮两臂与其148对action/video、1148条命令及初终环境/客户端RNG一致。历史数据只作为数值oracle；性能仍使用本轮实测S0，无历史耗时替代。

本轮未保存整局每个chunk的完整KV快照，整局结论来自全部输出/命令/环境/RNG一致；此前O3长KV完整chunk全30层有效KV字节/元数据一致的门仍冻结保留。不将整局输出一致描述成整局每个KV均被独立归档检查。

## 资源、退出与下一步

四个完整episode，保留第一批并补齐第二批；共八model启动、三个有限GPU窗口。第二批的两次启动在VAE/checkpoint加载阶段发现外部GPU进程，被原严格guard中止；0 target经constructor/无Runtime/无client证明，CUDA加载时间仍计费。旧FAILED原保守计数与清理失败保持不改，RECONCILED-CLOSED另记0前向/最终退出与稳定空闲。未启动的两任务通过有限续跑补齐；最后增加启动等待门，仅在client启动前等待外部占用消失并连续15秒确认，计数必须0；正式任务继续立即拒绝外部占用。没有重复任何已完成性能样本；计费包括首个窗口直到观察到稳定空闲的保守上界。target intents精确 3938，加载至清理 2069.849 秒；累计 33239/35000、11493.013/18000秒，余 1761 次与 6506.987 秒。旧失败保守消耗继续保留。

四客户端自然成功退出；全部async存档drain和计数闭合之后才按精确PID/start/UID/PGID/argv归属结束本次临时server。每批退出记录保留，终态GPU分配0、compute/graphics为空，外部服务未触碰。public源文件、已有worktree状态、protected state/rules/cycle与所有旧证据保持。

先补 O1+O2+O3 组合在短 KV288／中 KV1152 的完整 chunk 对照，固定同状态与原 Torch/旧 S0 口径；预算按本轮实际闭合计数预留。覆盖通过后再研究完整任务环境步数的控制，随后优先研究reset/T5编码开销，它已占新版整局约44%；不先追加更细碎的动作优化。

- [完整逐项结果](../../../投机采样/results/runtime/p0-optimized-e2e-resume-v2-20261013-160413/FINAL_RESULT.json)；[CPU预检](../../../review/OMISSIONS.md)；[方案](../../../review/OMISSIONS.md)；[GPU清理和资源](../../../投机采样/results/runtime/p0-optimized-e2e-resume-v2-20261013-160413/GPU-COLLECTION.json)。
