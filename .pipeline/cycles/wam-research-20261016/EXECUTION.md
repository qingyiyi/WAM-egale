# 最近执行检查点

## 2026-10-18T21:24:15+08:00：M22数值对齐完成，性能仍待新窗口

独立去--use_fast_math的v2：108/108 attention、54/54真实block、六full30 B14 raw/CFG/全部有效KV/RNG、三短中长完整canonical policy配对返回/声明状态全部逐位一致。三组各两臂每臂72实际forward（video/action同canonical工作），总714 full target forwards。CPU独立重读24完整query输出与三对policy返回共27检查全一致，无CUDA初始化。原tile/旧失败不变，有限fixture通过不改四false。

r1/r2真实终态return0并按PID出生时间核实退出，累计615.057/1200秒，新增源/编译/证据约40.71MiB≤512MiB，CUDA active未知。source main NCCL未显式destroy warning保留，已核实实际context释放。收尾lhw PID2015053约1090MiB，故新性能不启动，旧二进制时间不借用；准备worker/复测方案，等待自然独占窗口。

[结果](../../experiments/operator-torch-alignment-20261018/RESULT.md)、[机器表](../../experiments/operator-torch-alignment-20261018/RESULT.json)、[自有清理](../../experiments/operator-torch-alignment-20261018/OWNED-CLEANUP.json)。


## 2026-10-18晚：M22新授权与独立修正

用户授权实际对齐Torch并同步GitHub。旧tile原件不改；独立v2只移除--use_fast_math，179个vendor文件及去命名空间后的CUDA源逐字一致。CPU无GPU可见编译完成（nvcc12.4/wheel12.6差异保留）。九真实保存输入新v2全部逐位/输入布局/RNG通过，旧版本同输入仅3/9逐位；生命周期22.950秒。完整54点/六30层query/三完整policy配对重放作业已独立登记并启动，尚无终态。M22新累计GPU帽1200秒，不借M21余额。

已将M21全部正负结果、HW与新源码/方案推送review/operator-torch-parity-20261018分支，固定20562e3ad46896409579fbb0ffeb8ad049c8b54a；网页版prompt已交用户。该公开提交在新GPU验证前冻结，不能声称已包含后续结果。


## 2026-10-18T20:43:58+08:00：算子性能与完整I/O证据齐，真实负结果保留

新性能r6于20:14:26真实结束，54固定点/684单元/18360Event，5分2秒。原r5长重复按用户不足20分钟窗口收尾31点PARTIAL；新单进程3轮不冒充原双进程完成。九输入370763331B已存。实际加载80.317秒、重建/保存106.117秒、性能/资源核查78.943秒。

用户随后授权共享GPU继续精度，不重开计时。r8 worker1969142/birth1359532935，54点/414比较、264 full target forwards、6×4完整B14模型执行，298.212秒。standalone完整六query非逐位，后段三例不满足旧容差，最大有效action CFG误差.1875；shared/guarded六query输出与全部有效状态/RNG通过。r9保存九输入直接I/O27比较，24.749秒；24份完整输出CPU独立审计通过、未初始化CUDA。

本轮全部记录PID按出生时间/UID已核实际退出，[收尾核验](../../experiments/operator-kv-matrix-20261018/OWNED-CLEANUP-FINAL.json)。累计1649.274/1800登记秒，CUDA active UNKNOWN，新增约397MiB≤512MiB。无新机器人episode/训练/SHA/smoke，未动原tile或保护源/四false。

原r1 CLI0forward、r2未准入0forward共124.138秒；r3 CPU等待设置1800实际1831.139秒的收尾超额保留；r4初始化缺失0forward126.311秒；r5收尾774.179秒；r7预算reserve算术错误0worker/0forward0.000612秒。全部在独立namespace保留，未伪造PASS或旧清理收据。

## 资源迁移已完成，远端队列不受本轮影响

本机Object CURRENT16/30自然终态后CLEAN，旧controller/server/handoff退出；620898534B原件迁新机。远端r1 CPU路径绑定失败0GPU已保留，r2仅绑定物理路径、原assert不改，T200审计FIT80605/CAL9065通过。19:08最新已核Object54413/birth691998396等待Goal/Spatial自然槽位；本轮未声称刷新后续计数。余14/ON/final500均在远端队列，本机400局不自动启动。

## 证据与历史

[算子统一报告](../../experiments/operator-kv-matrix-20261018/COMBINED-RESULT.md)、[原算子CPU复核](../../experiments/operator-parity-evidence-20261018/REPORT.md)、[Long最终](../../experiments/long-morning-20261018/FINAL-SUMMARY.md)。

本次前完整PLAN/PROGRESS/EXECUTION/DISCUSSION逐字归档至history/*-before-operator-final-20261018-204358.md；更早过程按既有history索引，不截断失败或重写登记协议。
