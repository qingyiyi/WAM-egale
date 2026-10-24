---
kind: preference
name: 算子优先寄存器与共享内存并减少local访问
level: guideline
scope: wam-research-20261016/M39;单层persistent算子
status: active
updated: '2026-10-21T23:45:00+08:00'
supersedes: []
sources:
  - 用户明确要求不要过多使用local memory，尽量用shared mem和register
---

# 优先使用寄存器与共享内存，减少不必要的local访问

在当前optimized单层persistent算子中，优先让短生命周期、线程私有的中间量保留在寄存器；需要CTA协作、复用的数据使用共享内存，减少不必要的local-memory读写。优化以实际动态访存、耗时和原数值契约共同验收。

这是用户确认的优化偏好，不是要求无条件增加寄存器/shared用量，也不是禁止必要的全局输出或跨CTA工作区。资源容量、驻留、bank冲突及寄存器溢出仍需权衡；不能为了降低静态local数字放宽精度或掩盖性能退化。具体尚未验证的编译与调度方案不作为长期决定。
