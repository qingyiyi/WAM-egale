# RoboTwin执行进度

更新：2026-10-20T16:24:50+08:00。固定六任务各19初态齐，114/114。首个Target case完整成功，8policy/240实际命令、184.197秒；GPU1第二个stack_blocks_two正在运行。初始物理与双侧原观察逐位核对通过。

- [x] 原授权、固定源码/权重/环境/资产/设备绑定。
- [x] 配套535客户端库修复原rt/OIDN；两次同场景成功，旧失败保留。
- [x] 六任务×19固定有效初态，按原FIT/CAL/DEV角色。
- [ ] 原Target首6局：first6-2首case视频导出实施错误，0teacher/0动作PARTIAL；用户明确批准补跑，first6-3首局完整成功并已自动接第二局。
- [ ] draft/verifier/reset完整接口准入：108条完整512 bank张量拷贝通过；完整任务缓存等价未实测。
- [ ] 84局采集、固定训练链、90局三臂DEV：后继未完整实现，未自动排队。

专家None抓取异常按官方规则从下一seed接续，旧FAILED不改，[依据](EXPERT-SELECTION-CONTINUATION.md)。视频入口修正仅移除可选导出目录，[补跑证据](FIRST-CASE-RETRY.md)。已闭合GPU累计2266.544秒，当前运行另计；准备原18:20:28北京、24h/原存储帽不重置。首六局原17:46:42北京deadline，开局须预留1800秒与900秒清理。

[当前首批计划](RESUME-FIRST6-PLAN.md)、[现场](resume-evidence/resource-snapshot.json)。现有自动链只覆盖原Target首六局，不声称TF已经运行。四项科学权限仍false，不是标准50task评测或独立TEST。所有写入~/cxz，驱动/原物理渲染不动；AutoDL未连接。
