# 接续未开始的42条Target采集

[登记](REGISTRATION.json)基于最新明确接续；旧首六局及准备deadline保持为历史，当前新namespace明确起止，累计GPU仍原24小时帽。不是重试失败任务。

- [x] 核实首六局监督完整、原固定114初态及剩余valid01..07角色。
- [x] 独立wrapper泛化case标识，加载已核验完整512-token BF16 bank；保持原teacher/scheduler/RNG/物理和动作映射。
- [ ] GPU1单worker按固定valid index优先、六task轮转，完整42episode；每局1800秒/8000teacher，开局预留整局与900秒清理。失败/超时停止不重试；自然算法失败有效保留。
- [x] guard自动接续固定下一局，逐局保存RNG/初态/命令/监督/耗时；累计原账本。
- [ ] 完成后审计Target48的FIT36/CAL12与teacher标签，再接后续固定训练链；本guard不自动启动训练。

原数据未包含完整视觉/KV，不能声称可在新latent重query teacher。reset bank本轮复用同原Target缓存，完整episode cached/uncached等价仍未实测。资源墙钟预计约3–5小时，6小时执行帽；无法完成则PARTIAL，不扩帽。
