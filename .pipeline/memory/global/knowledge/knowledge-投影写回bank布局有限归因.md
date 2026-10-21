---
kind: knowledge
name: 投影写回bank布局有限归因
level: reference
scope: wam-research-20261016/M39
updated: '2026-10-21T18:00:18.327657+08:00'
sources:
  - .pipeline/experiments/single-layer-async-gemm-20261020/projection-gemm-core-20261021/RESULT.md
---
# 共享布局确实造成部分投影开销

固定真实B14 M7168/NK3072独立投影：shared accumulator stride128使K/V各2,064,384次shared写bank conflict；stride136后0，write wavefront减少75%。两进程三KV同前序padding缩短3.49%–3.66%，四元素搬运再缩短2.75%–2.93%，同期Torch差距仍约4%。SASS是LDS.128和两个32-bit global store，不是64-bit全局写回。

移入当前4warp完整单层，只改context K/V布局，9真实签名输出/23中间量/有效KV/RNG有限逐位通过；B14整层快3.12%–3.77%。独立local0，整层localSize1456→1464仍更快；该字段不能直接当动态spill根因。保留原三CTA调度负结果和8warp整层退化，不把bank padding当新调度创新。

复用测量脚本必须记录各标签的实际二进制路径，防字符串替换让两个arm指同一DSO；本轮无效对照与修复paired-r2保留。当前整层源fdf883e，精确身份与边界看SELECTION和RESULT。下一步仍先剩余原语/资源差距，再ready-DAG，不是完整机器人质量放行。
