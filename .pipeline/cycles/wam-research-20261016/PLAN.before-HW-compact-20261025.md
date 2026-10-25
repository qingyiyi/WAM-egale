---
kind: plan
mode: goal
cycle: wam-research-20261016
status: active
updated: '2026-10-25T14:18:57.297284+08:00'
progress: PROGRESS.md
execution: EXECUTION.md
---
# 当前计划：多初态门控质量、质量接续、单层算子收口

- [ ] M42-RECOVERY04：最新用户要求失败分析修复再接续；新六小时有界窗口，固定Goal剩余23case/90臂、同旧阈值/严格门。见../../experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume05/PLAN.md。


- [x] M39-CLUSTER2-ATTRIBUTION：三KV有限精度通过；2CTA仍慢A12.27%–15.46%，NO_GO保留A。B2启动近似无影响，G2绕过peer DSM仍慢；R0/R1支持输入就绪/有序提交开销，硬件流量未知。10413调用/17phase/回传清理闭合。[结果与全阶段表](../../experiments/sm120-ffn-next-20261024/CLUSTER2-ATTRIBUTION-20261025/REVIEW_RETURN.md)。

- [x] W02：持续逐段中文写作Prompt及必要轻量证据包已生成、入口链接已核验；main发布回执另记于maintenance/paper-collaboration-20261025。未生成论文正文，未启动新实验。

- [x] M39-CLUSTER4-HYBRID（测量/回传/清理已闭合）：三KV有限精度通过；C4整层比A慢33.87%–48.11%，NO_GO保留A。7122调用，无GPU失败/超时；原60分钟截止不变。[计划](../../experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/PLAN.md)。
- [x] M39-CTA-ATTRIBUTION（测量与归档闭合）：用户要求继续算子并解释慢因；A设备计算不变、只比较170/152 CTA，同轮A/R1/Torch精度与无插桩整层，完整阶段表。45分钟、8000调用、4CPU，原已收口DSM不重置。[诊断计划](../../experiments/sm120-ffn-next-20261024/CTA-ATTRIBUTION-20261025/PLAN.md)。
- [x] M39-DSM-STREAM：cluster8分片生产/消费及一次W2提前搬运R1完成；有限数值通过、整层慢31%–45%，不采用，保留A。[结果与全17阶段表](../../experiments/sm120-ffn-next-20261024/DSM-STREAM/REVIEW_RETURN.md)。5742调用、31.49分钟生命周期，源码/二进制/原始结果回传，GPU自有进程清理。不展开第二DSM设计。
- [ ] M42-MULTI-INIT（原窗口已收口，0合格四臂组；resume03准备失败及收尾自终止已独立归档，未重试）：用户要求多个初态取有效均值。固定Goal十task×init46/47/48×T/OLD/Q75/Q90，最多120新attempt；原init49不重跑；用户确认Goal优先，其他套件须在Goal收口后按原窗口余量推进。prep02守护直接拥有传输子进程，就绪自动接完整任务→原始归档→均值→临时资产清理。[当前计划](../../experiments/suite-acceptance-calibration-20261025/multi-init-20261025/prep02/PLAN.md)、[动态状态](../../experiments/suite-acceptance-calibration-20261025/multi-init-20261025/resume03/SUPERVISOR-STATUS.json)。原09:45:32起点、15:45:32截止/清理600秒不变，准备/等待算预算。
- [ ] M40-EVAL100：双A100固定600组三臂，535/600组、1605/1800局，112.560/160GPUh；双guard alive/HALT空。原281授权恢复、406/T OOM不重跑；原10天/更早10/29 guard不变。[实查](../../experiments/sm120-ffn-next-20261024/CLUSTER4-HYBRID-20261025/ROBOTWIN-PROGRESS.json)。
- [ ] M40-MICROWAVE-FINAL-DIAG：等微波炉100组完整后核官方Target，再查额外失败，原队列继续。
- [x] M41五环境5090有限面板与临时资产清理；[结果](../../experiments/rtx5090-spec-speed-20261025/RESULT.md)。
- [x] H02本机归档释放120.619GiB，源码/原tile/失败保留；[恢复](../../maintenance/storage-placement-20261024/RESULT.md)。
- [ ] M17旧配对/FIT-DEV缺口；不以本次DEV覆盖旧证书，不自动TEST/正式45chunk。

均值按初态/配对/完整终态/计时条件纳入，保留算法失败与慢成功，双方成功仅补充。全体attempt成本、有效样本N与排除原因、按task分层均报告。新panel是暴露DEV，非独立TEST，CPU独占未证明。

原四false、旧FAILED/retention/state/cycle/rules/原tile保护不变。投机重组→候选感知→质量约束成本仍待证；M39没有集成机器人。历史计划和已完成索引见PLAN.before-DSM-STREAM-20261025.md、before-M41版本。
