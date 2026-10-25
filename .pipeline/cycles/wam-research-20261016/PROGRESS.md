---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-25T12:37:58.807835+08:00'
plan: PLAN.md
current: M42-MULTI-INIT
next: M42-PAIRED-COST-RESULT
---
# 当前进度

M41已收口，表中加速为Target/投机成本；E2E含全部完整尝试与client启停，模型常驻，低于1即变慢。列内C/P=TF连续/PF2。

| 环境 | 完整三臂N | 成功T/C/P | action有效更新C/P | 任务E2E C/P |
|---|---:|---|---:|---:|
| long | 8 | 8/8/8 | 1.233/1.146 | 1.121/1.062 |
| goal | 10 | 10/9/10 | 1.161/1.137 | 0.828/1.123 |
| spatial | 10 | 9/9/9 | 1.175/1.117 | 1.065/1.013 |
| object | 10 | 10/10/10 | 1.203/1.131 | 1.075/0.990 |
| robotwin | 6 | 5/4/4 | 1.748/1.415 | 0.933/0.936 |

[本轮完整结果](../../experiments/rtx5090-spec-speed-20261025/RESULT.md)、[时间账本/长度](../../experiments/rtx5090-spec-speed-20261025/PERFORMANCE-RESULT.md)、[源码与原始数据入口](../../experiments/rtx5090-spec-speed-20261025/SOURCE-AND-RESULT-ENTRY.json)。

Goal task9出现C失败/P成功；Spatial PF2 task7为202/119=1.697倍，Object PF2 task5为248/131=1.893倍，不能只报中位长度。Robo篮子T成功/C/P失败，C订书机374/132=2.833倍；六case首S相同但五case第二S在首PF2刷新前已不同，隔离FAILED、根因UNKNOWN，不能作纯PF2因果结论。实际保存的是分数/路由、初态和所存RNG/命令，不是全部KV/velocity。

Long只有8共同三臂：两条0动作但已teacher执行的PARTIAL（21与23尝试/22返回）不重跑；旧不同FSDP完整T0和未配齐T/P1单列。141次client启动含4次0teacher准备失败，135自然完整终态；主表132局，不伪称138独立任务。

所有三臂full512 BF16 bank/eager/no外层Graph。Target reset现占总尝试时间约0.01%–0.03%；Robo与LIBERO时间边界分别登记。受控输入预传有CPU/IO重叠，Robo首16共享attention含inline数值检查；属于有限DEV工程观察，没有正式隔离重复。旧四科学权限全false。

本轮T0 01:15:33至06:01:46清理约4.770小时；95597 teacher尝试/95596返回。1664文件84.02MiB native传输/尺寸/fsync完成后删整个5090临时根（含新权重、bank、venv、sim/cache）。174个birth身份无活跃自有work，GPU compute context为空；原算子环境实际存在、实例未关。Goal输入一次EOF外预分配短时owned du36G违反34GiB cap，原失败保留；停自有rsync后回收并用exact-size preallocate修复，没有改cap或重置T0。Object0teacher bank绝对链接失效在新attempt复制原bank修复。

M40实查 2026-10-25T12:36:45.783774+08:00：520/600完整组三臂、1560/1800局，109.211/160GPUh；两guard alive/HALT空。原281/C授权恢复未闭合、406/T旧OOM不重跑，4条未配齐完整记录单列。[动态实查](../../experiments/sm120-ffn-next-20261024/DSM-STREAM/ROBOTWIN-PROGRESS.json)。

M39 DSM-STREAM已完成：8CTA各自产生两个64×128 GELU片，4生产/4消费warp，经DSM复制到消费者本地shared；28cluster job，保留七段K2048/BF16递推。B有限精度通过但整层慢49%–65%；一次R1提前W2 TMA使FFN约快22.5%，整层仍慢31%–45%，NO_GO保留A。默认/custom stream、三真实KV、27中间量、七段数值/有效KV/RNG有限逐位通过；不代表全部输入/闭环。

本轮A Torch/A约1.247–1.249／1.757–1.764／2.195–2.200倍；R1为0.909–0.910／1.215–1.219／1.674倍。warm CrossKV、eager、同轮无插桩两独立进程。R1消费者TMA等待约98→13μs，但DSM READY约72μs、驻留170→152CTA；不宣称动态HBM下降。5742调用、1889.25秒生命周期、87文件源码/二进制/原始结果回传核验/fsync、owned GPU已退出，实例未关闭。[结论](../../experiments/sm120-ffn-next-20261024/DSM-STREAM/REVIEW_RETURN.md)、[全阶段表](../../experiments/sm120-ffn-next-20261024/DSM-STREAM/STAGE-TABLE.md)。

H02完成：25564大文件逐项/闭合小批rsync校验和接收端fsync后释放120.619GiB，04:56结束，runner已退出；另有先前公共组件13.2068GiB去重与闲置/tmp清理，分别记账。默认临时目录/rjs/cxz/tmp。本机源码/HW/轻量证据保留，恢复见[结果](../../maintenance/storage-placement-20261024/RESULT.md)。

原state/cycle/rules、原tile、旧FAILED/retention、旧M17证据缺口与四false继续有效。此前完整Progress/Execution见同目录before-M41归档，不因精简删除。

M42五套83030行CAL已完成，CONT生成与暴露局限保留，OLD/Q75/Q90已冻结、PFoff/full512 BF16 reset。原Goal init49 Target成功126动作/9chunks/76.512秒自然episode，但OLD在teacher/action前腕相机9个通道差1触发严格初态门；原链PARTIAL、不重跑该case。

用户已确认Goal多初态优先，其余按原窗口剩余预算；用户要求多初态有效均值，固定Goal task0–9的init46/47/48，四臂轮换，原预算不变。首次多初态准备守护误识别相对路径进程而提前清理，0新episode，原件保留；prep02改为直接启动/登记子进程，当前取回Goal/VAE，自动接完整任务、汇总和清理。无新的有效配对均值。旧失效case不替换；固定新panel如初态无效，仅跳过该case未开始臂，按清单接续，不按结果选样。原15:45:32截止、每局1800秒、清理600秒；CPU争用未知，结果是探索性DEV成本。

当前路径：suite-acceptance-calibration-20261025/multi-init-20261025/prep02；SUPERVISOR-STATUS动态、RESULT.md/RESULTS.json/ATTEMPTS.csv由守护结束时生成。完整算法失败与慢成功保留；有效均值、总成本比、逐pair均值/中位/波动、双方成功补充和task分层分列。

## 2026-10-25T13:07:49.155783+08:00 M39单因素归因与交接细分

真实三KV、原A独立二进制、同设备体170/152CTA对照及152第二进程完成。152仅改变host grid，整层慢4.47%–15.48%，复核4.81%–15.31%；中KV Self attention +74.6%，Cross +66.8%–69.0%，源码168任务越过152驻留边界，支持波次导致退化。三KV27中间量/七段BF16/有效KV/RNG有限通过。

R1诊断拆开旧MMA计时内TMA提交：READY72–73μs、DSM复制/交接38–39μs、TMA提交14.5μs、TMA等待11–13μs、MMA/shared取数/组屏障92–93μs、前序等待<0.7μs。插桩local792B vs正常784B，不能直接当正式延迟；旧R1/B的86→107μs边界不同，不能称MMA变慢。28job/19cluster尾部9cluster=72CTA，生产/消费失衡另列。

保留A；A FFN尚慢约15%，整层仍快1.25/1.76/2.18倍。A25.8μs主循环/输入交接包络内部尚需定位，不将0.2μs提交等待当主因。5409本轮调用，7phase已真实exit0/child_exited，owned/GPU均空；原始数据回传进行中，尚未宣称归档闭合。报告/全17阶段快照：../../experiments/sm120-ffn-next-20261024/CTA-ATTRIBUTION-20261025/。

## 2026-10-25T13:10:37.103735+08:00 M39归档闭合

42文件/248170381字节native rsync与尺寸/fsync核验完成；5409调用、生命周期1139.08秒（约19分钟，包含编译/等待/归档；CUDA active未知），约238MiB本地存储，小于45分钟/8000调用/1GiB。7phase真实exit0、owned birth身份无活跃/GPU上下文空，实例未关闭。所有新源码/3个诊断二进制/6个数值trace均回传。采用原A不变。

M40 2026-10-25T13:09:10.105039+08:00：525/600完整组、1575/1800局，110.291/160GPUh，两guard alive。M42原Goal panel仍准备资产，13:10本机GPU compute为空，自动接续守护在运行，原15:45:32截止不变。

## 2026-10-25T13:47:40.038498+08:00 用户暂停本机A100

已向birth核验后的campaign发SIGTERM；其finally清理client和server，server真实返回-15，本机GPU compute列表为空。18attempt中10个已完整，暂停中的case2/init47/OLD保留PARTIAL，非算法失败。独立USER-PAUSE-RESULT解释原通用FAILED_OR_PARTIAL信号状态，不覆盖原记录。守护只作CPU归档/清理，不启动下一任务。本机须用户明确恢复；远端两A100/5090授权不变，原15:45:32截止不变。

## 2026-10-25T14:00:32.176134+08:00 本机恢复与门控主线澄清

用户明确恢复本机。resume03守护已启动，原T0/15:45:32截止与总预算保持；原收尾已删临时权重，正在重传，未声称GPU已恢复推理。固定23个剩余case/90臂，保留10个完整局、7个严格初态无效case与1个用户中断；仅中断局另attempt，其他只跑未开始臂。

当前10完整局全部成功：T3/3、OLD3/3、Q75 2/2、Q90 2/2；0完整四臂合格配对，不能判优阈值。逐字段CPU核查7个初态失败仅相机3–6通道差1–2，physical及其他保存字段一致；这只是观察，不证明渲染根因，也不放宽原门。

用户追问本机是否仍测试门控：主任务仍是CAL阈值的成功率/长度/接受深度比较；耗时只随完整任务附带记录计算代价，没有独立正式测速作业。先前汇报过度强调加速比已澄清。新运行：suite-acceptance-calibration-20261025/multi-init-20261025/resume03/SUPERVISOR-STATUS.json；初态差异：prep02/INITIAL-DIFF-COUNTS.json。


## 2026-10-25T14:18:57.297284+08:00 四CTA候选收口，本机仍以门控质量为主

B4仅cluster4启动，C4保留global ACT、局部DSM共享FFN2输入。三真实KV/27中间量/七段BF16/有效KV/RNG有限通过；完整层C4比A慢41.12%/48.11%/33.87%，NO_GO，原A源码/二进制保留。FFN阶段约0.548–0.550ms对B4约0.296–0.298ms；164CTA也使168attention任务增加波次。没有单独无插桩C4 FFN或动态HBM证据，不把所有差值归因DSM。完整17阶段表已在对话打印并保存。

初始c4-1/B4七段/状态专项检查晚于计时，原记录保留；最终c4-pre-gate在计时前完成门检，主表采用该轮。7122调用、28.98分钟含编译等待归档，56文件369215507字节尺寸/fsync验证；8phase exit0、owned/GPU为空，实例未关闭。结果：../../experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/REVIEW_RETURN.md。

M42 resume03仍恢复临时权重/VAE；tokenizer已完成。本机主目标是接受阈值对成功率、长度、接受深度与失败的影响，耗时附带记录，无独立正式测速。旧10完整/7初态无效/1用户中断保留，0合格四臂组，原15:45:32截止不变。M40 2026-10-25T14:17:15.662513+08:00实查535组/1605局、112.560/160GPUh，两guard alive。


## 2026-10-25T15:48:08.464854+08:00 双CTA实测与原窗口清理

M39：完成B2/C2/G2、各三KV完整数值先门检、C2/G2两进程无插桩整层及R0/R1成本观察器。C2比A慢12.27%–15.46%，G2仍慢8.71%–10.42%；C2比同轮C4快1.19–1.30倍但仍NO_GO。B2保持170CTA、启动近似无成本。输入二次swizzle、64K32步骤/128CTA屏障+16cluster交接及有序提交组合有开销；R1关键链提交等待约26–28 vs6–7微秒，W2就绪约23–24 vs10–11微秒，MMA包络接近。无计数器不能称DSM指令/HBM根因。实际保留A；完整17阶段表已直接打印并保存。

10413调用/17phase，105文件424775295字节回传核验/fsync，原A字节未改、owned/GPU为空；约52.62分钟含编译等待归档，小于原90分钟。入口../../experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md。

M42检查发现resume03在14:59 VAE路径缺config准备失败（0 teacher/新episode），supervisor finally把自己的OWNED收据纳入killpg，自终止后原状态停留RUNNING。本次独立RECONCILE-STATUS保留原异常，204结果文件归档字节/fsync核验，删除仅本轮临时资产13018066237字节；未恢复GPU/未改原15:45:32截止，原10完整/7初态无效/1中断、0合格四臂组不变。

M40 15:42实查549组/1647局，115.389/160GPUh，两guard alive；GPU0篮子092-C、GPU1微波炉091-P，旧281/406缺口保留。四false全保持。
