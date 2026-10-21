# 官方复现核查与每任务100次准备

用户最新明确要求每任务100次，替代此前50；六任务三臂目标1800完整局。不是RoboTwin全部50任务。原DEV90和四项false保留。

- [x] 固定LingBot/RoboTwin共11个关键源码文件，与对应commit归档逐字节一致。
- [x] 论文单任务结果、官方seed=0/demo_clean/test_num=100及LIBERO三seed口径核对。
- [x] 注册两条已曝光microwave原Target完整诊断，原_infer不安装draft Runtime，增加joint遥测。
- [ ] 完整终态、全命令/前两policy/RNG与历史Target比对；保留失败。
- [ ] 确认每任务100次面板与资源后接续扩大；不按旧余额启动无界1800局。

诊断运行边界见原PLAN.json（注册原件保持）：GPU1最多3600秒、2完整局、16000teacher尝试、512MiB；原M40累计50006.379538/86400秒及10/22 01:26:04.706686硬截止不改。旧PLAN.json中沿用的pair_order字符串只用于溯源，实际此诊断只运行INPUT-MANIFEST指定valid14/16两条T，不存在三臂或配对。

扩大评估具体范围/成本见EVAL100-PLAN.md；尚未启动。新增资源需单列，不能重置旧24h账本。
