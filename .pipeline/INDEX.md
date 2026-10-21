# WAM 当前入口

更新2026-10-21T10:46:42.181704+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- 算子：12小算子独立核对通过，point/RMS整层保留，固定预热较本轮前改善2.2%–3.0%；仍未速度对齐。
- 用户澄清：各算子适配tile/任务数，未来按DAG就绪填空，不凑108；当前未ready。
- [全阶段与CTA等待新证据](experiments/single-layer-async-gemm-20261020/minor-ops-20261021/RESULT.md) / [短状态](PROGRESS.compact.md)。
- RoboTwin parent18/CHILD/CONT/接口CAL齐；新授权已在GPU1启动CURRENT18并自动接ON/CAL。GPU0诊断已完成；四权限false，保护源与失败保留。

- [短/中/长KV及全部两进程统计](experiments/single-layer-async-gemm-20261020/minor-ops-20261021/THREE-KV-SUMMARY.md)：CPU补汇总，源码及实验未再运行。

- [2026-10-21T11:44:03.604996+08:00 GPU1接续状态](experiments/robotwin-pilot-20261020/current18-20261021/RESULT.md)；原24h累计预算不变，DEV未开始。
