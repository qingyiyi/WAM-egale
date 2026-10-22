---
kind: decision
name: 先实现CrossKV复用并精度通过后计时
level: constraint
scope: wam-research-20261016/M39
status: active
sources:
  - 用户明确同意先做cross KV复用，暂不融合tile和RMS，精度通过后测性能并输出阶段表
supersedes:
  - decision-先核查CrossKV按条件分组复用.md中的仅核查阶段
---
# 先实现 Cross K/V 复用，精度通过后计时

在当前optimized单层实现每层按相同context分组复用Cross K/V，保留独立RMS阶段，不融合投影tile与RMS，不改变self-attention候选私有KV。条件组映射与历史KV行映射必须独立，不能把同历史当成同文本条件。

先核验真实短/中/长输入、CFG和条件换序、输出/中间值/有效KV/RNG；通过原严格数值要求后再测完整层，更新当前阶段表。保留失败及旧版本，不放宽数值门。用户另要求核查LingBot-VA2加速基线，研究结果不能替代本项目实测。

沿M39现有记录和原累计十GPU小时，明确本轮有界作业及清理，不重置旧截止。baseline、投机侧、原tile与保护HW不动。新实现仅是研究入口，不改变四项科学权限false。
