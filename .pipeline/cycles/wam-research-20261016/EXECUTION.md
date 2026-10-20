# 最近执行

## 2026-10-20T09:33:50.273752+08:00：M37数值候选与成本归因有界完成

本轮远端T0=2026-10-20T01:26:56.101368Z，数值终态01:30:01.799432Z，原截止不重置。首次metadata入口在第一个block前因冻结DSO本机依赖失败，0block/0heuristic；原return0/status FAILED保留。仅本轮独立脚本改用已部署sequential调试参考，原Torch Linear仍直接做金标准；baseline/依赖不改。

4公共宏配置门通过，唯一CUTLASS设备候选四个Q/FFN2原逐位均FAIL；4参考block/12原语/8heuristic、0full30/episode。输入/RNG/重复保持，但不集成异步整层、不测性能。总预留生命周期185.698秒含失败/修正等待/CPU编译，不等于CUDA active。owned活动0、两卡context为空；本机88、远端86已部署保护项不变，原远端state/rules未部署的88项false保留。[结果](../../experiments/device-contract-20261020/RESULT.md)。

Long/Object已有330局[成本归因](../../experiments/device-contract-20261020/TASK-COST-DIAGNOSIS.md)完成，用户关于只报成功局仍是问句：建议共同成功同case总时间比与SR/全attempt/含失败成本并列，没有把建议提升为用户永久决定。旧执行见[本阶段开始归档](history/M37-start-20261020/EXECUTION.md)。
