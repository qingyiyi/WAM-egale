---
kind: decision
name: 依据profile继续单层tile流水优化
level: constraint
scope: cycle:wam-research-20261016;operator_research
status: active
sources:
  - 用户2026-10-20明确要求继续优化tile实现的Transformer算子
supersedes:
  - M38收口后M39仅为建议且尚未启动的执行状态
---
# 继续实现并验证单层tile流水

用户在M38源码讲解与profile完成后明确授权继续优化。按实测慢点先推进GEMM加载/计算流水及完整单层验证，再考虑attention数据通路与更细调度。每次有限实现单独登记真实输入、对照、生命周期、存储和清理，不借旧实验余额，不做无边界参数搜索。

主目标仍为一层、一个persistent kernel、明确warp职责。只在optimized独立入口实现，baseline及其实际依赖、投机源码、原tile/历史HW和保护state/cycle/rules不变。保留当前被另一agent带读的whole_layer.cu供比较；源码教学本身继续只读，本次独立优化授权不修改其任务范围。

匹配已失败原型只表示隔离的算术/状态保持，不表示匹配Torch或生产准入。新速度、原数值和状态门分别报告，旧FAILED与四项质量权限不改。允许已有BAAI资源在无争用时进行必要有界验证，所有远端写入仅~/cxz；不新增训练、机器人TEST、30层或租卡。
