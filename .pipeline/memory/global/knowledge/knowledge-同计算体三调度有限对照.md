---
kind: knowledge
name: 同计算体三调度有限对照
level: reference
scope: wam-research-20261016/M39
updated: '2026-10-21T17:04:44.011067+08:00'
sources:
  - .pipeline/experiments/single-layer-async-gemm-20261020/projection-dispatch-control-20261021/RESULT.md
---
# 本次投影慢因不能归于CTA总数

真实B14 Cross K/V，M7168/NK3072，固定256×128/8warp同compiled kernel；普通672CTA、108静态循环、108原子队列同前序两进程三KV的中位数差≤0.17%。36组有限逐位及672tile各一次通过。原子含队列清零增加约0.17%–0.33%；自有独立投影仍慢约11%，未证调度造成主要差距。

这只适用于本次同计算体和输入，不是所有persistent与普通grid等价。独立local0/整层local1456不自动证明根因；插桩增加整层1.32%–1.40%，阶段差不能直接全部归local。原4warp整层保留，新8warp整层退化保留；无新NCU。

测量复用注意：当前NativeBlock完整debug入口不可CUDA Graph捕获；raw Cross K/V字段未在其debug返回。须原版stage独立参考，不能照搬旧失败trace_worker。两脚本失败完整保存，不改变旧证据或质量权限。
