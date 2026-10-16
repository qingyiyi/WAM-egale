# 新30完整 Long10 DEV 配对实验

一次性固定顺序为新 T10 → 两阶段 MLP96 10 → 两阶段 TF64 10；均为 LIBERO Long10 task0–9/init24 的真实完整 horizon800 任务。旧40收集完整但终态失败，只用于记录历史失败与原 bank/target 来源；**旧 T10 和旧 MLP10 均不作本实验基线**。两阶段训练已固定各3 seed、2000+500 updates、50/50 targetFIT/历史 same-query FIT；TF seed20260923、MLP seed20260922。新两臂在 setup 时各用完整 20237 条真实 CAL 同 GPU 逐位验收原 predictor 接口。无 synthetic/smoke。

三个 arm 均使用同一原 WAM、target weights、text bank、原 LIBERO client、K2/B2 bonus、阈值 0.007839680526450805、prefix_then_cost/O3；T 关闭投机。每臂一个常驻目标、一个 EGL 客户端，目标 CPU/CUDA RNG 在 draft/CAL 前捕获并每局恢复。每完整局存 command、初末 sim/controller/RNG、两个实际初始 128×128×3 uint8 相机像素、reset 与最终 KV/VAE、全部模型请求/acceptance/counters、原异步 actions/latents/obs 输出。初始相机在原 lean `_init_env` 返回后、首次 policy 前复制到 RAM，`initial_observations.npz` 在原 episode timer 结束后写盘；像素 hash 必须等于 `repro_metadata.json`。终态逐任务比较三臂初始模拟器/控制器、Python/NumPy 和目标 CPU/CUDA RNG、完整 reset 状态及两个相机像素。像素差异以改动像素/通道数和最大/平均幅度写到 `PAIRING-PIXELS.json`，FAIL 时仍保留原像素，不重试或放宽。

资源门：本机 A100-SXM4-40GB，入场可用显存≥22 GiB，运行可用≥0.5 GiB，共享 GPU 只用于算法质量；CPU 2 核。总墙钟90分钟，最后300秒保留给收尾；每局墙钟40分钟。三臂合计 target forward≤40000，单局≤4000，server forward pre-hook 硬拦；输出≤4 GiB、磁盘余量≥5 GiB。全部 160 个实际小源码依赖（WAM py/config、legacy contract/adapter/model/helper、共享 prefix `.so`）均随 plan 绑定；target 大权重仅核 path/bytes/mtime，银行字节单独纳入集成来源。每 arm/最终进程树清理、30完整局、两新臂完整GPU CAL、所有保存文件审计、REPORT 和严格配对证书通过且无失败标记后，最后才写 `RESULT.json`。共享 GPU 墙钟不能作正式加速结论。

预登记结构筛选：TF 成功数同时≥新 T 和 matched MLP；TF/新 T 双方成功环境步数比 P50≤1.10、P90≤1.20、max≤1.30 且双方成功集合非空。并报告每任务失胜、失败800步罚分、接受深度、target/chunk。十对 DEV 不能保证成功率3pp非劣性。真实扩大质量评测和正式速度仍是下一阶段。

根目录执行：

```bash
/rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/target_trace_draft_fresh_dev_v1/run.py --job 投机采样/results/runtime/p0-target-trace-draft-fresh-dev-v1-20261015-r1 --prepare
/rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/target_trace_draft_fresh_dev_v1/run.py --job 投机采样/results/runtime/p0-target-trace-draft-fresh-dev-v1-20261015-r1
```

仅首次命令是 CPU plan 冻结，第二条才会申请 GPU；均一次性，无自动重试。目前新30尚未 prepare/运行 GPU。已用实际 va Python 导入8模块、历史失败/重置来源门与训练入场门通过，AST通过；数值/成功率/速度均待真实30局验证。
