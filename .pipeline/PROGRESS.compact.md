# WAM研究短入口（2026-10-25）

先读[约束](memory/INDEX.md)，再按需读[当前Cycle](cycles/wam-research-20261016/PLAN.md)。本文件是派生摘要；原始PROGRESS、保护文件、旧FAILED/retention、原tile不改。本镜像只包含WAM当前主题。

M41已收口：[结果](experiments/rtx5090-spec-speed-20261025/RESULT.md)、[五环境表与账本](experiments/rtx5090-spec-speed-20261025/PERFORMANCE-RESULT.md)、[固定源码/原数据入口](experiments/rtx5090-spec-speed-20261025/SOURCE-AND-RESULT-ENTRY.json)。三臂T/TF/PF2，有限暴露DEV工程面板，非独立TEST/正式性能。

| 环境 | N | 成功T/TF/PF2 | 动作加速TF/PF2 | client任务E2E TF/PF2 |
|---|---:|---|---:|---:|
| long | 8 | 8/8/8 | 1.233/1.146 | 1.121/1.062 |
| goal | 10 | 10/9/10 | 1.161/1.137 | 0.828/1.123 |
| spatial | 10 | 9/9/9 | 1.175/1.117 | 1.065/1.013 |
| object | 10 | 10/10/10 | 1.203/1.131 | 1.075/0.990 |
| robotwin | 6 | 5/4/4 | 1.748/1.415 | 0.933/0.936 |

任务E2E包含所有完整失败、每局初始化和关闭，模型已常驻；暖口径、完整policy/chunk、长度与罚分另列原表，不能混用。Goal task9 C失败/P成功；Spatial/Object PF2成功长尾1.697/1.893倍，Object P90也超1.20；Robo篮子新增失败与C订书机2.833倍长尾。Robo五case第二个S在首次PF2刷新前已不同，隔离FAILED、根因UNKNOWN；有限分数/路由通过不能作全KV/velocity等价证明。

Long因两条已执行PARTIAL只有8共同组，旧不同FSDP T0/未配齐T/P1单列；不以成功筛样。原21teacher计数滞后与23尝试/22返回保留，不重跑。Object bank绝对链接失效为0teacher/0episode准备失败，复制原44MiB后新attempt开始。

所有三臂沿优化full512 BF16 bank/eager/no外层Graph；reset实测约0.01%–0.03%总尝试时间。受控资产预传有CPU/IO重叠，Robo首16 shared attention含inline数值检查，尚无正式隔离重复。四字段engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission均false。

M41原T0 01:15:33到06:01:46清理4.770h；95597 teacher尝试/95596返回，1664文件84.02MiB核验/fsync回传后删5090整个临时根/新权重/env/cache。174个birth身份无活跃自有work，GPU compute contexts空；原operator/env存在、实例未关。Goal传输曾因EOF外预分配短时owned36G违反34GiB，已停自有rsync回收并修复，不改cap/旧日志。

M40双A100继续：459/600完整组、1377/1800局；T/TF/PF2成功371/374/378，96.887/160GPUh，两worker alive、HALT空（2026-10-25T06:27:03.821629+08:00）。281/C原授权恢复继续，406/T OOM不自动重试。微波炉100组三臂完整后再诊断，旧更早10/29 14:03 guard不变。[实查](experiments/robotwin-pilot-20261020/eval100-20261022/external-oom-20261025/M41-CLOSURE-CHECK-20261025.json)。

M39当前保留SELF-QKV-TMA-4MMA；四驻留/交接方案与PAIR2048x2均负结果。新PAIR正常整层慢6.25%–7.89%，不采用；原Torch/A三KV约1.251/1.758–1.761/2.191–2.201倍，单层同warmCrossKV/eager，未接M41任务，不乘算倍数。[结果/17阶段表](experiments/sm120-ffn-next-20261024/REVIEW_RETURN.md)。下一轮FFN/任务依赖须有界方案，未自动扩展。

H02本机轻量化完成：25564文件逐项校验/接收端fsync后归档释放120.619GiB，runner退出；另有公共组件13.2068GiB去重和/tmp清理分别记录。默认TMPDIR=/rjs/cxz/tmp，原tile/代码/轻量失败/代表输入保留。[结果与恢复](maintenance/storage-placement-20261024/RESULT.md)。A100质量/5090性能/本机轻量开发；5090本轮无权重常驻，后续需重传。

后续：M40完整覆盖与原微波炉诊断；Robo分叉前一致性定位；再据质量冻结候选/正式性能，不能盲目加训练。旧M17配对/FIT-DEV缺口仍在，M01–M04历史完成不代表整体完成。完整旧短入口在PROGRESS.compact.before-M41-20261025.md；Cycle四旧文同目录before-M41保留。
