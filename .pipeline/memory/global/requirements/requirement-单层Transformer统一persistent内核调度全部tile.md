---
kind: requirement
name: 单层Transformer统一persistent内核调度全部tile
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
updated: '2026-10-19T22:25:32.200411+08:00'
supersedes:
  - 将M32仅attention和FFN1阶段内常驻CTA循环视为用户整层persistent目标已完成的解释
sources:
  - 用户明确：整个单层transformer都做成一个算子，然后在内部tile放到persist warp上跑
---
# 整层是统一persistent执行器的工作边界

用户要求一层完整Transformer的GPU统一执行器，内部包含norm/调制、QKV投影、self/cross attention、输出投影、FFN1/FFN2及残差等tile，由常驻warp或必要的协作warp组执行并满足跨阶段依赖。单个Python/C++接口再依次启动多个独立kernel，不构成这一目标的完成。

已有M32只在attention和FFN1采用常驻CTA循环，其余仍独立kernel/host cuBLASLt；它是保留的局部原型，不能称为完整层persistent实现。不能将固定步长CTA循环描述成已实现warp领取ready任务或动态队列。

此要求固定实现目标，不指定必须使用某种原子队列、tile尺寸或warp数量；具体机制需依据真实DAG、SM80资源与数值门设计。只有已就绪任务可被执行；不得令消费者占满常驻资源而阻塞生产者。仍只做一层，不做30层大融合，不降低原精度/状态门。baseline与投机侧不改，优化仅optimized，远端仅~/cxz。
