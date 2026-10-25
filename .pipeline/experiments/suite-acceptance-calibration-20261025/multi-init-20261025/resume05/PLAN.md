# Goal失败修复与固定面板接续

本轮T0 2026-10-25T15:56:53.159109+08:00；硬截止 2026-10-25T21:56:53.159109+08:00，最多6小时含准备、等待、完整任务、失败与清理。单A100 worker，server2/client2 CPU，临时30GiB/结果3GiB，teacher总cap沿既有1500000且旧消耗计入；每局1800秒、600秒清理，开局须预留整局。

- [ ] 核对资产布局、配置和ownership；修复缺失vae/tokenizer链接与自终止收尾。
- [ ] Goal16路/VAE8路分段读取并经native rsync恢复原Goal/common资产；不做SHA、smoke或新小测试。
- [ ] 加载原冻结源码，完整episode接续固定PANEL（旧初态INVALID/完整任务不重跑）。
- [ ] 保留所有attempt，汇总质量/长度/接受深度与附带成本，归档后清理临时权重。

不训练，不改tau/normalizer/初态门/四false，不扩套件或正式性能。原resume03失败及15:45:32截止保持；本轮是用户新要求的接续。
