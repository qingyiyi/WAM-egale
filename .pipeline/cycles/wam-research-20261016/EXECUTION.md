# 最近Execution checkpoint

## 2026-10-21T13:45:15.316595+08:00 FFN1有限结果完成，DEV90获准补启

- M39一个尾块候选9签名有限逐位通过，两进程三KV完整层改善0.19%–0.50%；仍慢14%–29%，未ready。当前同源嵌入差−1.23%至+0.36%，动态spill等UNKNOWN。
- GPU53.236828秒，累计1905.463629/36000秒；owned退出，GPU0释放。[全部样本/源/资源](../../experiments/single-layer-async-gemm-20261020/ffn1-tail-control-20261021/RESULT.md)。登记模板旧文字与实际manifest差异单列，未改旧JOB。
- M40首DEV启动加载前导入失败，0teacher/0episode/5.005915秒；用户明确批准新namespace补启，旧失败与截止保留。
- r2真实导入链CPU通过；13:35:21启动guard，模型READY并进入固定完整任务。当前完整3/90；自动接续，后续失败不自动重试。[当前状态](../../experiments/robotwin-pilot-20261020/dev90-20261021-r2/STATUS.json)。
- Maintain已写限定重启决定；原累计24h与10/22 01:26硬截止不变。HW修复Plan/Progress稳定ID格式与引用，仅派生语义入口，保护state/cycle/rules未触及。
- [前执行记录](history/ffn1-tail-dev90-20261021/EXECUTION.md)。
