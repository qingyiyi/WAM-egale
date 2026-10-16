# 当前preset短中长完整policy chunk结果

2026-10-14，Asia/Shanghai。**三档测试已完成：短1.602×、中1.544×、长1.500×。** 45个完整chunk通过，正式计时18对全部提速，同算法RAW/current的输出、route、全30层有效KV和RNG现场检查通过。最终任务自有进程全部退出、GPU显存分配为0。旧失败尝试与费用保留。

## 主结果

本地A100-SXM4-40GB，PyTorch2.9.0+cu126，7个CPU线程/CPU0–6，单部署模型内AB/BA交替。候选为当前preset：A_off、K2/B2、bonus、prefix_then_cost、原WL2阈值、共享KV和O1/O2/O3；没有混入PF2、MSE/RES或新MegaTransformer。

| 档位 | 请求前history / action verifier总KV | 原T中位秒 | 当前投机中位秒 | 加速比 | 耗时下降 | T/current相对MAD |
|---|---|---:|---:|---:|---:|---|
| 短 | 144 / 288 | 6.332767 | 3.953222 | **1.601926×** | 37.575% | 1.079% / 1.293% |
| 中 | 1008 / 1152 | 6.275748 | 4.064502 | **1.544039×** | 35.235% | 1.433% / 0.983% |
| 长 | 2016 / 2160 | 6.674660 | 4.449593 | **1.500061×** | 33.336% | 1.676% / 1.900% |

各档各臂6个正式样本，所有样本保留，没有因快慢删除或追加。相对MAD为`median(abs(sample−median))/median`，六组均通过预声明≤7%检查。每档6/6配对更快，配对加速范围分别1.421–1.634×、1.484–1.601×、1.444–1.576×。中位数和较低MAD描述本轮重复稳定性，不是跨任务置信区间。

三档来自同一task0/init3的frame4/28/56。中档从原短snapshot和真实记录的frame4/8/12/16/20/24重建，实际18次cache前向，保留原30层slots/容量；没有裁切KV或生成替代轨迹。三个输入不同，不能把这张表当作只改变KV长度的纯因果消融，也不应要求T耗时严格单调。

## 测量包含什么

主指标是`server.infer`完整policy请求、前后CUDA同步的墙钟。包含video20/action50、draft、候选verify、接受/校正/bonus、原始输出CPU拷贝与后处理。50次action scheduler提交保留，没有通过减少去噪步数提速。

reset、快照恢复、数值核验、模型加载和编译在主计时之外，但计入GPU资源预算。HTTP/客户端序列化、环境执行，以及随后根据真实执行观察发起的独立cache更新请求不属于这次policy请求计时；不能称整个机器人控制周期或完整episode速度。原异步保存的CPU拷贝保留在请求内，后台落盘完成不计入请求时间，收尾等待写盘仍计资源预算。初始化/预热不能并入正式样本，也不能据此推断首个冷请求速度。

每档1次RAW_CURRENT oracle、T/current各1次warmup，再运行6对AB/BA，共15次；三档45次，其中36次正式计时。profile、teacher、full trace关闭，无EGL或训练。监测程序只做轻量状态读取；正式窗口无其他GPU程序。

## 数值、路径与计数

每次调用前恢复同一真实输入、30层有效KV/metadata、scheduler、VAE流式状态与RNG。同算法RAW_CURRENT与优化CURRENT比较完整输出hash、逐tree接受记录、phase/forward、RNG及现场完整有效KV字节；所有CURRENT调用均一致。T也按其自身oracle检查重复确定性。没有要求近似投机与原T动作相同。

O3每次均显式确认`enabled=True, ok=True, closed=True`；共享attention无泄漏、调用后恢复原方法。非共享臂明确标记未启用，避免继承上一臂metadata。独立CPU审计261项通过，重算记录、统计和计数；完整GPU KV字节由运行现场检查，不声称离线重新加载/检查了未保存的全KV。

| 档位 | T完整policy target调用 | 当前完整policy target调用 | tree数 | 接受深度0/1/2 | bonus更新 |
|---|---:|---:|---:|---|---:|
| 短 | 72 | 43 | 19 | 3 / 3 / 13 | 13 |
| 中 | 72 | 44 | 21 | 3 / 8 / 10 | 10 |
| 长 | 72 | 45 | 21 | 6 / 3 / 12 | 12 |

调用数包含video、action及terminal前向；候选批量验证一次调用含7逻辑状态/14CFG行，不能把调用次数下降直接等同算力下降。所有投机调用提交50次action更新。成功job实际2586次target intent/返回闭合，其中包含18次中档前缀重建；无失败前向或活动请求遗留。

## 失败保留、GPU占用与资源收尾

| job尾部 | 结果 | GPU阶段秒 | 完成chunk | forward收费 |
|---|---|---:|---:|---:|
| 182138 | 模型加载时出现外来GPU进程，中止 | 32.878859 | 0 | 保守5000 |
| 183552 | 新测试封装漏传冻结共享kernel启用标志，优化路径拒绝 | 71.790266 | 2个oracle/warmup，0正式计时 | 保守5000 |
| 183953 | 修正后完整三档采集通过 | 367.771439 | 45 | 实际闭合2586 |

第二次问题仅在新supervisor的child环境中修复，补充CPU测试验证两个标志分别缺失均拒绝，实际启动环境可到达冻结kernel入口；不修改共享kernel、模型或算法。修复前harness副本及失败日志保留，失败中的两次输出不混入成功job统计。

外来进程所属系统账号是`lhw`；原PID1801365退出后，曾出现同账号PID1820030，入口`.venv/bin/python -m benchmark_scripts.benchmark_gpu_batch`，父应用为VS Code Codex。只能确认账号与程序入口，未确认真实姓名/项目目录，权限拒绝后未继续读取其目录或环境，也未向其进程发信号。再次确认连续60个空闲采样后才创建后续job。

成功job监测74次GPU样本，foreign均为空；自有root正常exit0。最终NVML为0分配显存、0计算/图形进程、0%利用率。三个job共新增472.4405635353秒（约7.874分钟）；累计30550.4884022828秒/36000秒，剩5449.5115977172秒（约90.83分钟）。累计forward收费144550/174494，含两个未闭合失败预约各5000；这不是声称失败实际各算了5000次。

## 结论与下一步

当前工程与投机组合在所测三种真实阶段均降低完整policy请求耗时，长KV收益仍约1.50×，短/中也有收益。它补齐了此前的三档速度待办；不是全LIBERO、完整任务、成功率或trace长度保证。旧接受器质量问题仍需单独解决，当前preset保持不变。

MegaTransformer可以与tile统一为同一研究目标，具体范围见[整合决策](MEGATRANSFORMER-TILE-INTEGRATION-20261014.md)。先对冻结当前CFG14 verifier做独立profile，选择一个真实block内的执行/融合原型，再验收完整30层与chunk增量收益；原tile代码/HW不移动或重写，尚未产生新MegaTransformer性能。算法质量协议和候选分析可并行；同卡正式速度与其他GPU工作串行，reset继续后置。

## 复核入口

- [成功job与冻结plan](../../../review/OMISSIONS.md)
- [45次原始结果及全部样本](../../../review/OMISSIONS.md)
- [资源闭合](../../../review/OMISSIONS.md)
- [独立CPU审计](../../../review/OMISSIONS.md)
- [统计复算](../../../review/OMISSIONS.md)
- [启动标志修复CPU验证](../../../review/OMISSIONS.md)
- [第一次尝试的历史状态](CURRENT-CHUNK-LENGTHS-STATUS-20261014.md)
