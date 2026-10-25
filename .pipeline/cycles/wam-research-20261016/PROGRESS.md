---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-25T19:36:51.428062+08:00'
plan: PLAN.md
current: M42-MULTI-INIT
next: M42-PAIRED-COST-RESULT
---
# 当前进度

## 资源状态（2026-10-25 20:55北京时间快照）

- 本机无自有GPU作业；Goal resume05闭合4合格四臂组，resume06初态诊断0teacher准备失败，旧记录保留。质量线已转5090迁移。
- 双A100队列无可领取任务：598完整组/1794局，T/C/P487/485/491，125.250/160GPU小时；两worker退出，旧2组PARTIAL不自动重试。[快照](../../experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/ROBOTWIN-SNAPSHOT.json)。
- 5090 A2已闭合NO_GO保留S；RoboTwin资产/客户端准备，服务端依赖安装中，GPU完整评估未启动。M42原21:56:53截止不延长，单worker避免32GiB OOM。

## M42质量与阻塞

五套83030行CAL已有限重放，主池CONT3000生成；不冒称当前ON的on-policy数据。固定Goal十task×init46/47/48×T/OLD/Q75/Q90，PFoff/full512 BF16 reset。旧18attempt中10完整均成功（T3/3、OLD3/3、Q75 2/2、Q90 2/2），7个严格初态无效case、1用户中断；**当时0合格完整四臂组**；该旧状态不覆盖resume05新4组。

resume03在0teacher/0新episode因Goal VAE路径缺config失败，收尾误向自身killpg造成过时RUNNING；[独立核对归档](../../experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume03/RECONCILE-RESULT.md)保留。用户要求修复再接续；resume04为0episode计划性传输交接，resume05沿同一新窗口，修复common链接/owned清理，不改严格初态门、tau配方或normalizer。

## M39单层系统

当前采用S（FFN两阶段同步版，原A170独立保留）：B14/Q16、KV288/1152/2160、warm CrossKV、eager/noGraph。只在FFN1+GELU后加grid同步，FFN2七段BF16递推与tile/warp不变。本轮三真实fixture、27tensor/七段/有效KV/输入/RNG、两个独立进程及公开入口全部有限逐位通过。

无插桩完整层原Torch/S约1.288/1.797/2.238倍，S相对A延迟下降2.25%/1.82%/1.90%；资源仍170CTA、100352B shared、255regs/thread、784B local。FFN1+GELU为0.797–0.805×Torch，FFN2+残差为1.027–1.033×；下一步先定位FFN1计算/后处理成本。2865调用、17.12分钟生命周期、约132MiB新证据，owned清理完成。[新结果/18阶段表/固定入口](../../experiments/sm120-ffn-next-20261024/FFN-STAGE-SPLIT-20261025/REVIEW_RETURN.md)。

后续有限实验均保留负结果，S仍采用：W（FFN1 1T/4M/3E、448个M64N128）整层慢约0.8%；共享B在W基础上让双CTA不同M/同N共享权重，整层慢49.89%–60.62%。[W](../../experiments/sm120-ffn-next-20261024/FFN1-WARP-PIPELINE-20261025/REVIEW_RETURN.md)；[共享B/等待证据](../../experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-WEIGHT-20261025/REVIEW_RETURN.md)。

用户追加的共享A（同M/不同N、8KiB DSM）已完成：同轮比共享B延迟减少18.76%–21.24%，但仍比S慢21.70%–26.48%，未采用。共享A FFN1约323μs，同场W约152μs；只分组/独立加载L_A约等于W。27tensor/七段/计数/状态、两进程三KV均通过；资源仍170CTA/100352B/255regs。观察器定位接收READY约1.03μs/K64、源EMPTY约1.76μs、MMA输入等待约1.35–1.37μs；相互重叠，不当纯DSM带宽。5400调用、10.96分钟生命周期、S公开入口与15文件字节回传/owned清理完成。[共享A结果/完整18阶段/源码](../../experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-INPUT-20261025/REVIEW_RETURN.md)。

最新A2把共享A相邻两K64合并16KiB交接，实际DSM/credit/ACK减半；三KV/两进程精度状态通过，整层却比旧共享A慢3.44%–4.35%，比S慢25.92%–31.73%，不采用。FFN1约355μs、同场独立加载W152μs，保留原S；4842调用、11.62分钟生命周期，owned清理/IO恢复闭合。[完整结果/阶段](../../experiments/sm120-ffn-next-20261024/FFN1-CLUSTER2-GROUP2-20261025/REVIEW_RETURN.md)。失去step+2预取与结果一致，但未独立量化各等待贡献。

下面是相对旧A的历史DSM负结果，不重写为对S测量。

| 已闭合研究路径 | 相对A结果 | 采用 |
|---|---|---|
| DSM8/R1 | R1整层仍慢31%–45% | NO_GO |
| C4混合DSM | 慢33.87%–48.11% | NO_GO |
| C2混合DSM | 慢12.27%–15.46%；G2绕过peer DSM仍慢8.71%–10.42% | NO_GO |

[历史DSM结果/17阶段表/源码](../../experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md)。三真实fixture/27tensor/七段/有效KV/RNG有限逐位通过，不外推全部输入或闭环。2CTA恢复170CTA并比C4快，新增swizzle/主循环/交接/提交仍有成本；R0/R1观察器有扰动，硬件流量与纯DSM指令根因UNKNOWN。10413调用、17phase、105文件回传/fsync、owned清理已闭合。

[FFN同步源审查](../../experiments/sm120-ffn-next-20261024/FFN-SYNC-SOURCE-REVIEW-20261025.md)：608/863、681/867可研究合并；C2单A槽974/983不能直接删。仅CPU判断，无新收益。每段FP32部分和先加前段BF16运行和，末段再bias后舍入；不存在额外独立BF16(partial)步骤。

## M41有限五环境结果（历史，非TEST）

44完整三臂组/132主表局，C/P为连续TF/PF2。E2E含完整失败与client启停、模型常驻；action列按有效更新计成本。不同N和边界不合并。

| 环境/N | 成功T/C/P | action加速C/P | 任务E2E C/P |
|---|---|---|---|
| Long/8 | 8/8/8 | 1.233/1.146 | 1.121/1.062 |
| Goal/10 | 10/9/10 | 1.161/1.137 | 0.828/1.123 |
| Spatial/10 | 9/9/9 | 1.175/1.117 | 1.065/1.013 |
| Object/10 | 10/10/10 | 1.203/1.131 | 1.075/0.990 |
| RoboTwin/6 | 5/4/4 | 1.748/1.415 | 0.933/0.936 |

[完整账本与边界](../../experiments/rtx5090-spec-speed-20261025/PERFORMANCE-RESULT.md)。Goal新增失败；Spatial/Object的PF2成功长尾1.697/1.893倍、Robo C长尾2.833倍及篮子新增失败保留。Robo五case第二S在首PF2刷新前已不同，隔离FAILED/根因UNKNOWN；Long仅8共同组、已teacher执行的PARTIAL不重跑。全臂优化reset、首次构建另列，有限探索性成本非正式计时。临时权重/环境回传核验后已清除。

## 权限、规划与历史

`engineering_quality_screen=false`；`independent_test_complete=false`；`noninferiority_established=false`；`formal_performance_permission=false`。原M17配对/FIT-DEV缺口、原FAILED/retention/协议保持；不启动独立TEST/正式45chunk或无限训练。

H02已归档核验后释放120.619GiB，源码/原tile/必要失败保留；[恢复路径](../../maintenance/storage-placement-20261024/RESULT.md)。中文稿逐段由用户整理飞书，证据包规划11/01 18:00北京时间；证据不足就收缩claim。

[精简前完整进度](PROGRESS.before-HW-compact-20261025.md)；[完整事件](EXECUTION.before-HW-compact-20261025.md)；[旧计划](PLAN.before-HW-compact-20261025.md)；[旧讨论](DISCUSSION-SUMMARY.before-HW-compact-20261025.md)。动态状态随时间变化，开新作业前重读实际status/登记，不从此快照恢复旧窗口。

更早的LIBERO全量结果、迁移与M17限制见[原完整进度](PROGRESS.before-M41-20261025.md)；本页M41是5090有限DEV，不替代原全量评测。
