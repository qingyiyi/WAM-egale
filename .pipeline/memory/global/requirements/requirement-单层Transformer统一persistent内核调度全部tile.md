---
kind: requirement
name: 单层Transformer统一persistent内核调度全部tile
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
updated: '2026-10-21'
supersedes:
  - 将M32仅attention和FFN1阶段内常驻CTA循环视为用户整层persistent目标已完成的解释
sources:
  - 用户明确：整个单层transformer都做成一个算子，然后在内部tile放到persist warp上跑
  - 用户进一步明确：整层一个persistent kernel，切tile放到persistent warp，每个warp只负责自己对应的事情
---
# 整层是统一persistent执行器的工作边界

用户要求一层完整Transformer的GPU统一执行器，内部包含norm/调制、QKV投影、self/cross attention、输出投影、FFN1/FFN2及残差等tile，由常驻warp或必要的协作warp组执行并满足跨阶段依赖。单个Python/C++接口再依次启动多个独立kernel，不构成这一目标的完成。

已有M32只在attention和FFN1采用常驻CTA循环，其余仍独立kernel/host cuBLASLt；它是保留的局部原型，不能称为完整层persistent实现。不能将固定步长CTA循环描述成已实现warp领取ready任务或动态队列。

此要求固定实现目标，不指定必须使用某种原子队列、tile尺寸或warp数量；具体机制需依据真实DAG、SM80资源与数值门设计。只有已就绪任务可被执行；不得令消费者占满常驻资源而阻塞生产者。仍只做一层，不做30层大融合，不降低原精度/状态门。baseline与投机侧不改，优化仅optimized，远端仅~/cxz。

## Warp职责专门化是交付要求

用户进一步明确：常驻warp有各自清楚的职责，按职责执行其对应工作。交付必须包含真实的warp角色划分与生产者消费者同步；所有warp不分角色地重复运行同一个通用tile循环，不能作为最终实现。一个tile需要多个计算warp协作时允许使用协作组，但各warp的角色、数据交接、就绪与复用条件必须明确。具体角色数、warp数和tile尺寸由设计与实测决定，不将某个尚未验证的分配方案写成用户要求。

该澄清替代M33缺口文档中“先全组协作、warp专职分工只作后续可选优化”的完成标准；旧原型和旧测量保留，不能改称已满足。

最新授权（2026-10-20T00:07:07.268346+08:00）：立即开始整层单kernel源码与CPU编译，与M35计时并行；GPU验证等待对应卡的计时队列关闭和清理，明早汇报实际速度、精度和问题。

用户2026-10-21明确澄清：每个子算子采用其合适的tile大小与逻辑任务数量，不要求每阶段都铺满108个CTA，不为了凑满而改变分块。未来调度依据真实DAG就绪条件，把后续无依赖或依赖已满足的tile提前填补空闲执行资源。逻辑tile任务数与固定persistent worker数量分别说明；两者不能混称。当前优先小算子核对与整层对齐，澄清不要求立即跳过该步骤启动ready调度。
