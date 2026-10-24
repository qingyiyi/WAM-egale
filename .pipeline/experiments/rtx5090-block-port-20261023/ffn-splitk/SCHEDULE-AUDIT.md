# FFN2 split-K：领取与发布协议CPU审计

审查对象：本轮 `workspace/算子研究/wam_ops/optimized/csrc/whole_layer.cu` 新 `action_ffn2_splitk`。本报告未运行GPU，不等同于CUDA硬件验证、精度证书或性能测量。

## 结论与前提

在以下前提下，本领取协议不会因为“所有CTA等待尚未派发的前驱”发生死锁：

1. 每个CTA先完成其静态分配的全部FFN1工作，然后才进入FFN2领取循环。
2. 四个row队列各自单调领取0…167；任务号按 `partition * 24 + column` 排列，CAS不越过168、不重复。
3. 一个已领取任务在提交完成前不会丢失、撤销或由其owner退出；每CTA最多持有一个任务。
4. 所有参与CTA保持驻留并获得执行进展；源码以实际occupancy决定cooperative grid大小。
5. 发布/获取遵守下面的内存顺序，且浮点计算、barrier及硬件本身无异常。

证明：对于任意已领取 `(row, column, p)`，若p>0，其直接前驱任务号小24，必已被领取或完成。等待只能指向同一输出tile的更小p，严格下降至p=0，因而无环。p=0无需前驱，可提交；之后按归纳法每段可提交。若row尚未ready，对应FFN1生产者不会同时等待FFN2，因为每CTA先完成自己的静态生产任务。提前进入最终grid barrier的CTA已经没有未提交任务，因此不会成为缺失的前驱owner。此证明依赖实际驻留与执行公平性，不声称CUDA规范保证任意自旋程序在任意launch下均有进展。

**独立FFN2测量也必须遵守驻留前提。** 不能把672个普通非cooperative block直接启动后沿用此等待协议；本方案是驻留CTA领取672个任务，而不是要求672个CTA同时驻留。

## 源码逐项核验

| 项目 | 当前源码位置 | 核验 |
|---|---|---|
| ACT及完成组发布 | `ffn_publish`，458–469行 | 每个写线程threadfence，CTA barrier，leader device-scope acq_rel fetch_add，再CTA barrier |
| FFN1先于本CTA消费 | `action_ffn_pipeline`，643–647行 | 先完整执行静态FFN1，再进入split-K，末尾grid.sync |
| ready检查 | 590–595行 | rows0/1等待生产计数28、28；rows2/3等待28、36，均acquire读取 |
| 单调有限领取 | 596–603行 | relaxed CAS用于唯一任务ID，不用于发布ACT；每row上限168 |
| 无ready行不能退出 | 605–616行 | 仅四队列全claimed才退出，否则继续等待 |
| 当前K段计算 | 617–621行 | partition-major解码；计算完成后shared部分和由本CTA独占保存 |
| 前驱结果获取 | 625–628行 | leader device-scope acquire读commit，再CTA barrier传播到全CTA |
| 当前段结果发布 | 629–637行 | 全CTA提交、各写线程threadfence、CTA barrier、leader release store、CTA barrier；最终段再递增row输出完成数 |
| 初始化 | 1180–1190行 | 96个commit归零；原16状态初始化；后有grid barrier |
| 实际驻留上限 | 1381–1391行 | occupancy计算blocks，cudaLaunchCooperativeKernel |

leader的release不能代替其他写线程的发布义务。当前实现明确让所有写线程先fence并与leader汇合；后继leader acquire后再与其他线程汇合。各生产CTA的acq_rel RMW形成发布链，使读取最后完成数的消费者能够读取整个对应ACT组。领取CAS仅提供唯一所有权，relaxed足够；不应据此把ready与commit原子的acquire/release降成relaxed。

## 预期终态

- `ffn_pipeline_state[0:4] = [28, 28, 28, 36]`：120个FFN1生产tile。
- `[4:8] = [168, 168, 168, 168]`：672个FFN2 K段任务，CAS不超额。
- `[8:12] = [24, 24, 24, 24]`：96个最终输出tile。
- `[12:16] = [-1, -1, -1, -1]`：旧“首次领取时已生产数”当前helper没有采集。
- `ffn_splitk_commit`：96项全部7。
- debug `ffn_trace[4:8]`：当前helper没有设置首次消费时间，保持0；不得据此声称已测得首次消费时刻。
- 非debug最后一段写OUT，F2最后一次完整物化只在debug发生；不能用非debug F2当最终FFN2输出审计。

行号对应本次审查时源码；数学函数未由本审计修改。

## 有限随机CPU模型

`schedule_audit.py`模拟170个CTA，随机FFN1生产耗时、K段计算耗时、CTA轮转顺序、早退及有序提交。运行单Python进程约1.544秒，320个调度全部完成：256个交错生产/消费调度，加64个生产已完成调度。最长33轮，最多80个CTA同时等待。每个任务均检查前驱已领取、无重复owner和终态计数。

负控故意颠倒partition领取顺序，7轮后出现170个CTA全部等待未领取前驱，符合应检出的死锁类型。负控不是实际算法候选。

结果见 `SCHEDULE-AUDIT.json`。该模型没有模拟GPU弱内存、warp执行、真实指令流水、真实cooperative驻留、硬件公平性或浮点数值；其通过不能替代实际GPU验证。它只支持算法领取顺序的有限检查，结合上述单调依赖证明用于启动前CPU审查。
