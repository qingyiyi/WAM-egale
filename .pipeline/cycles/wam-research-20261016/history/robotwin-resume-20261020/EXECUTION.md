# 最近执行 checkpoint

更新时间：2026-10-20T15:01:42+08:00。

## 2026-10-20T15:01:42+08:00：RoboTwin原渲染恢复与MegaMoE差距审查

1. 用户恢复环境排查，不改驱动，并要求合并算子审查；通过Maintain保存新决定。
2. 原A100/GPU1同场景做535配套库→595 compat→535配套库三次独立进程：成功/非法访存/成功。两次成功均完整专家及四路相机观察，非模型质量样本。仅客户端库路径修复，驱动保持535。
3. 新增119.345秒，原账本累计614.666秒；失败/清理/源快照归档，最终无owned残留或compute context。原174局未自动启动。
4. 对照固定MegaMoE与当前源码，完成[差距与下一步](../../../../experiments/single-layer-async-gemm-20261020/MEGAMOE-GAP-REVIEW.md)：优先B14专用化和强baseline/profile，再K片段数据流水。未改算子或运行新性能样本。

## 2026-10-20T14:40:00+08:00：用户暂缓RoboTwin，说明B14边界

1. 用户确认暂无可接续的官方原渲染环境，选择暂缓RoboTwin、优先收口算子结果；通过Maintain保存决定，不改变原失败、预算和权限。
2. 用户明确原始action保留原算子。只读复核B14短/中/长KV单层耗时分别为native的5.022/4.515/3.956倍，非完整verifier或E2E。
3. 核对两次原日志均为svulkan2的OIDN非法访存；首场景前置受阻，0policy/teacher，根因尚未独立锁定为驱动。本次只读核验和文档维护，无GPU启动或远端重试。

## 2026-10-20T14:20:35+08:00：有界验证收口与环境阻塞

1. 用户明确同意两卡详细计划，登记M40独立迁移和原M39有界后继；保护项/旧预算不重置。
2. O1九点数值修复完成；O2计时仍慢native。两个同名扩展初测INVALID，PID命名空间与清理竞态失败保留。
3. O3完整行组ready实现并数值通过，实测更慢；bounded CAS修正空转后仍无净收益。O4依据参数副本/stack证据修正只读描述符，最终两路径有限数值通过，默认仍慢2.68–5.02倍，ready慢1.2%–6.0%。[最终结果](../../../../experiments/single-layer-async-gemm-20261020/FINAL-LAYER-RESULT.md)/[资源](../../../../experiments/single-layer-async-gemm-20261020/FOLLOWUP-STOP-REPORT.md)。
4. RoboTwin固定源码/权重/资产/私有规划依赖准备完毕，真实原渲染连续两版本OIDN非法访存；共495.321秒，0policy/teacher。停止后继并归档，不改系统、物理或视觉配置来绕过。[阻塞](../../../../experiments/robotwin-pilot-20261020/ENVIRONMENT-STOP-RESULT.md)。
5. 最终核对43owned根进程无仍匹配身份、两卡0MiB无compute进程。现无自动队列；等待兼容环境不是占卡运行。

完整中间执行文字保留[本轮归档](history/ordered-followup-20261020/EXECUTION.md)。计算终态、有限精度与科学权限分别记录，不以脚本exit0或单kernel通过覆盖原FAILED/四false。
