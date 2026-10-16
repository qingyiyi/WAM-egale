# PyTorch2.9 第一版实际验收：一个 full-block 失败保留

2026-10-12T00:50:47.496296+08:00，Asia/Shanghai；纯tile，非投机采样。

## 实际结果

- 按原门槛完成两独立进程，每进程全部六份真实短/长K2160输入、7轮；总576项正确性检查通过。每类36960份event/同步wall原始值全部保留，完整离线审计两份结构均通过，原PyTorch Flash backend ID1。
- 全部48项self/cross eager/core性能门通过；full-block 11/12通过。r1全过，r2的tile0长video Q128/K2160 **1.0040128093（慢0.4013%）**：PyTorch 1.913856ms，CUDA 1.921536ms，差7.67994us。
- **原预声明全gate未通过。** 不用r1、稍快的wall中位、core收益或“差异很小”替代原event验收。审计exit1是真实失败，不是报告损坏。
- 所有raw、包括早期慢轮与长尾、旧实现归档及历史+2.7763%/诊断/计时失败保留；没有同版本重测挑通过。测试期间无NVML采样、profile/interposer；前后197包文件及353保护条目不变，投机/模型/server/state/rules未改。

## 原因界限与下一步

1. Python额外读取shape、幂运算默认scale、转发调用：源码确实比Torch native路径多；值得做主机接口对齐，**尚未证明解释全部7.68us**。
2. 主机runtime/allocator/调度与共享cache同步：此前有长尾证据，当前未定位唯一API内部或间隙机制；不利用未校准的TSC/Nsight结果定因。
3. 单轮非平稳/交错顺序：本次r2早期两臂均慢，所有轮照原median-of-round-medians统计，不删轮、不放宽门槛。

用户已同意仅把默认scale移入C++、Python直接导出native入口；原CUDA核/全部输入检查/vendor/测试/门槛保持。先现有工具单worker编译与CPU核验，再新版本固定两个独立进程、每进程全部六输入、600秒上限；不增加同版本重试。新候选GPU预算0/2，最终清理未执行、goal未完成。

[原r1](../../../review/OMISSIONS.md)；[含失败原r2](../../../review/OMISSIONS.md)；[完整审计](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/release-confirmed-idle-20261012/actual-release-report-audit.json)；[统一证据](../../../review/OMISSIONS.md)；[新候选计划](../../../wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/outputs/tile-pytorch29-baseline-20261011-203651/host-api-align-20261012/plan.json)。
