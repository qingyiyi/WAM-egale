---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-23T20:58:56.516598+08:00'
current: M40-DUAL-5090-HANDOFF
next: 双A100在原共享预算内自动接续RoboTwin，5分钟后台巡检；5090下一步重新核对guarded并重测Torch/guarded/自有三方eager阶段与整层，旧速度不继承。
---
# 当前研究入口

用户新资源决定：双A100持续接续RoboTwin；算子开发、精度与性能统一5090。阶段表恢复原Torch/guarded/自有三方同条件eager；guarded的5090数字未测。

5090精度修复已完成：当前ffn-bias-last在三份真实B14输入（KV288/1152/2160）上26中间量＋输出逐位一致；两独立进程默认/非默认流共156完整block重复核验通过，有效KV、输入、RNG和FFN队列终态一致。修复六投影10段、FFN2七段、末段bias次序，以及原B14预计算后去重缓存。仅独立SM120研究采用；修复后性能未测，旧速度不能继承，四科学权限及production_admitted=false。

输入为原block0 B14/Q16/D3072/H24/FFN14336，真实短/中/长KV288/1152/2160。最终三档最大绝对差和相对L2均0；26中间量＋最终输出均逐位一致。两个独立进程各78次完整调用，共156次；重复运行不等于更多独立输入。默认公开加载入口的普通调用每档每进程20次，默认流和自建CUDA流均通过，不使用CUDA Graph。本轮覆盖只读历史与预计算Cross KV，未验证B2/任意层/任意输入/在线写缓存/机器人闭环。

根因与修复：5090原Torch六小投影split-K10（九段320＋尾192）、FFN2七段2048，不同于原A100两段；每段BF16舍入与末段先合并前段后加bias均要保持。FFN2改为两完整ACT行块就绪后七段有序处理，仍四MMA warp/128×128输出tile。Python公共缓存入口先原B14投影/RMS，再去重输出。没有改原Torch数学、后端精度、阈值、模型或生产策略。

研究保留独立5090的ffn-bias-last；原A100采用入口保留。170CTA×8warp、shared100096B=97.75KiB、255寄存器、静态local784B。新版性能NOT_MEASURED，旧profile二进制也未更新。旧迁移eager1.22/1.72/2.14倍只属于数值未过版本，不可搬到新版本。

原非法访存FAILED（尾K迭代）和三个数值未过候选全部保存；最终三个fixture初验9次＋两进程156次确认通过。20:30:12所有owned PID＋出生身份退出，GPU显存0、无可见compute；未杀他人任务，租机未关。源码9文件与回传副本byte一致，DSO/结果已在本机，无SHA。本轮保守生命周期27.982分钟，累计24832.512/36000秒，余3.102小时；不自动消费余额，原截止不改。

[本次回传](experiments/rtx5090-block-port-20261023/precision/REVIEW_RETURN.md)、[完整阶段表](experiments/rtx5090-block-port-20261023/precision/STAGE-TABLE.md)、[固定源码/二进制](experiments/rtx5090-block-port-20261023/precision/SELECTED.json)、[资源](experiments/rtx5090-block-port-20261023/precision/RESOURCE-AND-STOP-REPORT.md)、[旧迁移失败与测速](experiments/rtx5090-block-port-20261023/REVIEW_RETURN.md)。

## 继续保留的边界

环境RTX5090 SM120、Torch2.9.0+cu130/nvcc13.0.88/driver580.105.08。counter受限，本轮无提权、改驱动或重启；只见自有context不能证明宿主级独占。利用率异常旧原因UNKNOWN，当前清理观测0%。

A100局部warpT0–T4：主线重叠净收益INCONCLUSIVE，副线LN1/LN3融合L13仅独立研究采用，旧A使用入口未替换。有限门与旧Torch比值不转借5090；109.75→97.75KiB两槽版本的首次GPU核验在5090，本轮修复后有限通过。原A100两槽未重测。[历史](experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)。

2026-10-23T20:58:56.516598+08:00核验：远端A100 GPU0新namespace已接续RoboTwin，GPU1原队列继续。共享队列完成148/600组、444/1800局；同148组T/C/P成功119/118/122，仅扩大DEV，四权限false。含活动租约累计32.867/160 GPU小时，global halt为空。 GPU0新namespace和GPU1共享原子pair队列及账本，优化reset复用；5分钟CPU巡检PID1028069。GPU0首新pair审计尚待完成；原无自动重试规则保留。原11/01 22:03登记截止及实际guard更早10/29 14:03上限均保留。 [实际交接](experiments/robotwin-pilot-20261020/eval100-20261022/dual-continuous-20261023/RESULT.md)。

M17四套LIBERO6000局历史数据齐；训练/DEV暴露、严格配对及长尾缺口保留，四科学权限false，不恢复旧队列。不将调用减少或单层加速称作E2E收益。

10/23冻结身份/claim，10/28主表，11/01 18:00中文证据包；不足收缩claim。protected state/cycle/rules、原tile与投机代码不动，固定v15，不全量sync。

下一步：双A100在原共享预算内自动接续RoboTwin，5分钟后台巡检；5090下一步重新核对guarded并重测Torch/guarded/自有三方eager阶段与整层，旧速度不继承。

[Cycle计划](cycles/wam-research-20261016/PLAN.md)、[完整进度](cycles/wam-research-20261016/PROGRESS.md)、[优化账本](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
