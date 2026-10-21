# CURRENT18 → ON3000 → 部署接口CAL：完成

2026-10-21 13:02北京核验。用户要求RoboTwin继续后台，既定链未经人为暂停，已自然完成；当前没有机器人GPU worker。DEV90入口尚未完成审查、没有排队，不宣称整个迁移完成。

- CURRENT18：18/18完整，13成功5自然失败，FIT12/CAL6固定。完整失败为press_stapler/valid11、open_microwave/valid11、open_microwave/valid12、press_stapler/valid13、open_microwave/valid13。原始终态保留；这是CONT3000访问数据，不是ON3000的DEV成功率。
- [监督审计](AUDIT/RESULT.json)：5779 FIT行、3502 CAL行；无缺失/排除，按实际到达query保存。完整teacher条件未保存，不能声称可在新latent任意重新query teacher。
- [ON3000](ON3000/RESULT.json)：从CHILD2500固定再500updates，Target FIT/CONT生成FIT各50%，normalizer/seed/loss冻结，不用CAL选checkpoint，新增teacher为0。
- [部署CAL](CAL/RESULT.json)：Target_CAL12 7693行＋CURRENT_CAL6 3502行，共11195行；GPU adapter/decode有限逐位、decision flips=0。固定tau=0.007879490032792091，未重校准。不是B2/B14 teacher一致或闭环质量证书。
- reset：108条完整512-token BF16 bank沿用；RoboTwin完整episode bank等价没有新增验证。

CURRENT采集GPU生命周期5028.801秒、teacher13921次；后继GPU28.417秒、CPU审计4.325秒。原24h账本累计25801.965秒（7.1672h），余60598.035秒（16.8328h）。[终态与PID出生身份](TERMINAL-SNAPSHOT.json)确认guards/worker退出，GPU0/1均无compute context。

watch的`complete_collection_episodes=17`是等待时最后快照、未在终态刷新，不能用它覆盖CURRENT最终18局及AUDIT固定18；旧字段原样保留。本报告直接使用完整终态和审计。

下一步：完成既定DEV90的三臂轮换、canonical T/SST路由、初态/RNG/首两S与成本审计入口，才启动固定valid14–18面板。不是新增采集/训练或新测试集。不得仅为占GPU启动不完整审计的作业；四项科学/正式性能权限false不变。
