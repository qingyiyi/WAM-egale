---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-25T06:29:00.133606+08:00'
plan: PLAN.md
current: M40-EVAL100
next: M40-MICROWAVE-FINAL-DIAG
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

M40实查2026-10-25T06:27:03.821629+08:00：459/600完整组（1377/1800局），T/C/P成功371/374/378；96.887/160 GPU小时，两worker alive，无HALT。原281/C授权尾部恢复继续，406/T OOM未授权重跑；聚合成功不能覆盖新增失败/长尾。原更早10/29 guard保留。[结束前实查](../../experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/M41-CLOSURE-CHECK-20261025.json)。

M39保留SELF-QKV-TMA-4MMA。上轮B/C交接小幅退化、M16 ACT驻留D与双CTA DSM E约2–3倍慢于A；新PAIR2048x2有限七段BF16递推/状态通过，正常整层两进程三KV慢6.25%–7.89%，NO_GO。当前A原Torch/A约1.251/1.758–1.761/2.191–2.201×，单层、同warm CrossKV、eager，不能称机器人收益。[结果/全表](../../experiments/sm120-ffn-next-20261024/REVIEW_RETURN.md)。baseline/原tile未改。

H02完成：25564大文件逐项/闭合小批rsync校验和接收端fsync后释放120.619GiB，04:56结束，runner已退出；另有先前公共组件13.2068GiB去重与闲置/tmp清理，分别记账。默认临时目录/rjs/cxz/tmp。本机源码/HW/轻量证据保留，恢复见[结果](../../maintenance/storage-placement-20261024/RESULT.md)。

原state/cycle/rules、原tile、旧FAILED/retention、旧M17证据缺口与四false继续有效。此前完整Progress/Execution见同目录before-M41归档，不因精简删除。
