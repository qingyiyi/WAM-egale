# WAM 当前入口

更新2026-10-21T13:04:10.565362+08:00。先读[策略](CONTEXT-POLICY.md)、[约束](memory/INDEX.md)，再读[当前Cycle](cycles/wam-research-20261016/PROGRESS.md)及PLAN/EXECUTION/DISCUSSION。固定v15，不全量sync。

- 算子：同源码独立/整层嵌入差小于约1.4%，旧大额函数组织惩罚缓解。专用FFN2小幅改善0.36%–0.65%，完整层仍慢15%–30%，先FFN1再ready。
- [本轮证据](experiments/single-layer-async-gemm-20261020/ffn-resource-control-20261021/RESULT.md) / [短状态](PROGRESS.compact.md)。9签名有限逐位，不外推闭环。
- RoboTwin后台链自然完成：18采集13成功/5失败、ON训练与11195行接口CAL齐。DEV90入口待审查、未排队；原24h累计7.1672h。
- baseline/原tile/投机生产保护、四权限false、旧失败/截止不变。每阶段按适配tile，不凑108；动态spill/HBM/L2仍UNKNOWN。
