# RoboTwin固定1800局新一期

用户明确授权：160GPU小时、最长10天，现有远端两卡，不租新卡。旧24小时账本/截止/失败不改。冻结六任务、ON3000与原tau，T/C/P三臂，全部完整终态；各task100个官方expert有效初态，非独立TEST，四权限false。

- [x] 资源授权与原复现/工程等价证据核验。
- [ ] GPU1先补齐官方完整expert筛选；GPU0长期保留给算子，不自动借用。
- [ ] 冻结GPU1六任务×100有效初态，600组×3=1800局；顺序TCP/CPT/PTC，按task与seed标记旧FIT/CAL/DEV暴露。
- [ ] 同全512 BF16 text bank与COMBINED工程选项；先CPU依赖核验，再自动完整三臂队列。
- [ ] 每30局shard汇总，任何实施/初态/首两S失败停止本worker，不自动重试；自然算法失败继续。

GPU1单worker累计最多160GPU小时；准备最多4小时，包含在160小时内。每局≤1800秒，开局保留完整上限和900秒清理。新增持久证据合计≤12GiB。保留used cameras/原光追与OIDN、物理、成功判据。此处只用一个远端worker，不在本机抢占GPU。

存储：完成且C/P前两段审计通过的原始初态与prefix张量才进行gzip-1无损归档，逐字节解压核验后移除冗余未压缩副本；失败终态同样保存。LOSSLESS-ARCHIVE.jsonl记录原路径/归档路径；使用torch.load(gzip.open(path,"rb"))恢复。旧实验不改。单CPU归档器不使用GPU；本轮代表旧文件压缩比约0.18–0.21，预计符合12GiB上限，仍由guard实际监控。
