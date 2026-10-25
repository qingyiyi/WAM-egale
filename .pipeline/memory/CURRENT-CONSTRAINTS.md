---
kind: derived-constraint-view
level: constraint
scope: project:WAM;read-routing
status: active
---
# 现行约束短视图

本文件汇总既有约束，不创建执行授权；用户最新明确指令、结构化规则和原实验协议优先。具体任务还须读取对应主题原文及实际作业登记。

1. **保护与历史**：不覆盖`.pipeline/state.yaml`、`cycle.yaml`、`rules.yaml`或结构化规则；不修改原tile/冻结源。旧FAILED、负结果、retention、预算起点/截止、PARTIAL及不可重建证据保留；新尝试独立身份，不事后改PASS。
2. **质量与数据**：四项`engineering_quality_screen`、`independent_test_complete`、`noninferiority_established`、`formal_performance_permission`当前均false。DEV/训练暴露、旧T/TF、不同硬件不混成独立TEST；不能挑成功/快样本或删长尾抬高结论。阈值挑战只按已登记CAL方案，不逐task手调、不放宽旧门。
3. **实验契约**：不跑用户不需要的smoke、独立模型SHA或截短任务冒充质量；完整episode沿登记horizon。数值路径变更须重新核验接受/合法路径/scheduler、RNG与全部有效KV。算子replay不能冒充闭环。
4. **指标**：有效推进成本是`ΣT_round/Σg`，target calls/接受率不能代替E2E。SR、全体失败罚分、成功长度与双方成功补充分列；固定chunk与自然episode分开。性能同硬件/缓存/reset/计时边界，正式计时无争用，profile另列。
5. **资源与清理**：每job登记范围/预算/终态/失败/产物/owned清理；等待与失败计生命周期。历史窗口不自动续期，新增研究范围不从旧计划推导；不租新卡、不杀他人进程、不改系统服务/驱动/时钟。EGL并行与单worker例外以原结构化规则和job登记为准。
6. **位置与权限**：BAAI所有写入限`~/cxz`；不提权、不访问已禁止的GPU硬件计数器，软件插桩不能冒充NCU。账户临时目录`/rjs/cxz/tmp`。既有后台任务继续按原范围；HW整理不重启或移动它们。
7. **共同工程条件**：采样/质量/性能双方都用已验证full512 BF16 reset bank，冷启动另列。5090是现有算子性能平台，A100承担已授权质量；本机临时验证以最新明确接续和job为准。持续使用5090不等于每job无预算。
8. **交付**：中文逐段带读/写作；不建论文正文初稿。算子新候选逐项给完整阶段表（tile/工作量/warp/同步/三KV相对Torch），未测与历史标明，失败也保留。公开进度只推已核对的main，不传权重/原tensor/DSO/凭据。

原规则：[实验与证据](global/rules/requirement-实验与证据边界.md)、[资源/EGL](global/rules/requirement-资源与EGL并行规则.md)、[BAAI目录](global/rules/requirement-新BAAI服务器仅在用户主目录cxz内工作.md)、[计数器限制](global/rules/requirement-远端不提权且不访问GPU计数器.md)。其他适用约束由[主题入口](INDEX.md)路由；不从历史计划自动恢复作业。
