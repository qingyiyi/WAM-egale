# 最近执行

## 2026-10-21T22:14:16.314145+08:00 入口精简、原Target复现与成本诊断

- M39直接改optimized：去NativeBlock依赖、两死函数、正常路径的debug统计；数学/tile/sync不变。
- 九签名与普通/debug有限逐位，完整层三KV两进程变化−0.132%～+0.103%。GPU55.564秒、CPU133.950秒，owned退出，当前两机源/DSO一致。
- M40两条原Target完整轨迹及保存前两policy/RNG与旧T相同；valid14仍失败、valid16成功。不是新独立质量数据。
- 用户要求先判断100次方案为何慢，未授权160GPU小时。旧90局reset仅0.0047%，client RPC外60.02%；补一条完整profile后观察规划/渲染/初始化成本。
- profile完整成功、全部提交命令相同；GPU285.934秒，原累计51483.683/86400秒；原deadline保留，owned清理。
- 新增计数语义：原actual_commands是调用数，真实推进另列。旧FAILED/权限/罚分不改。

[算子结果](../../experiments/single-layer-async-gemm-20261020/readability-20261021/RESULT.md)、[RoboTwin慢因](../../experiments/robotwin-pilot-20261020/runtime-profile-20261021/RESULT.md)、[前执行](history/readability-and-runtime-before-20261021/EXECUTION.md)。

## 2026-10-21T22:31:52.701622+08:00 接续RoboTwin两项工程优化

用户已授权，先源码核对，随后隔离变体的完整任务验证；计划最大4局、90分钟GPU生命周期、1GiB新增存储，保留原累计/硬截止。仍未启动1800局。
