# 合法前缀选路、质量协议与标签审计

- [x] P1：独立 PREFIX 选路、反例、穷举与完整 BF16 CPU 语义检查。
- [x] P2：固定质量协议、审计旧标签与最近bonus记录、冻结有限GPU计划并通过启动预检。
- [x] P3：八个原Torch GPU完整chunk精度对照，原T/DEPTH/PREFIX六条完整episode质量检验。
- [x] P4：采用决定、HW记录、保护与封存审核；报告结果及后续MLP训练工作。

本轮仅 selector 变化；不训练、不放宽阈值。先CPU、后GPU；使用 QUALITY-PROTOCOL.md 的有限开发门槛，速度确认后置。create_job.py、preflight.py、supervisor.py、analyze.py 顺序执行。
