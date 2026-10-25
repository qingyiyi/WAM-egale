# 5090 RoboTwin接受阈值有限DEV

用户最新要求覆盖本轮Goal优先，改为RoboTwin先执行；旧Goal结果/初态门不修改。沿原21:56:53截止，准备、迁移、等待、失败和清理计入，600秒清理。不是新6小时或无限质量窗口。

- [x] 5090无compute，31.4GiB可用显存，40GiB可用磁盘；本机诊断进程已退出。
- [ ] 复用已跑通的SM120部署：冻结Target/TF_ON/normalizer、三候选tau、full512 BF16 bank、原光追/去噪配置、完整任务及同初态/RNG/双相机检查。
- [ ] 两个独立worker最多并行，每workerserver2/client2 CPU、不同端口/cache/RNG/output；同一case四臂由同worker依固定顺序完成。先worker0实际装载/完整首case测峰值再判断worker1，不能用OOM试探。
- [ ] 六任务valid49×T/OLD/Q75/Q90最多24完整episode；全部暴露DEV，原A100/M41数据不替代新Target。
- [ ] 成功/失败罚分/实际步数/接受深度/有效推进成本；附带墙钟不报正式速度。
- [ ] 结果回传核验并清理全部本轮临时权重/新环境；之后接续共享A输入交接算子实验。

阈值由既有CAL冻结；不追求特定在线接受率，不逐task调参、不搜额外tau。严格配对未通过则保持INVALID；无法完成就PARTIAL，不以像素容差自动放行。不改变六任务原horizon/后处理/渲染，不启动模型训练。
