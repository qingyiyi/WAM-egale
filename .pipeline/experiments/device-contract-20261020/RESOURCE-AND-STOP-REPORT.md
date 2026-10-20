# M37资源与止损

- 原T0：2026-10-20T01:26:56.101368+00:00；工作截止2026-10-20T02:01:56.101368+00:00，清理截止2026-10-20T02:11:56.101368+00:00，没有重置。
- 数值终态：2026-10-20T01:30:01.799432+00:00，完整预留生命周期185.698秒；45分钟上限内收口。剩余额度不作为后续自动授权。
- 全child生命周期和91.284秒；GPU worker生命周期和33.639秒（含模型读入/CPU工作，非CUDA active）。CPU编译57.644秒，2核。不得把有重叠边界相加成计费小时。
- 原始实施失败1次，后续修正1次；0 timeout，无完整episode自动重试。4参考block、12 primitive API调用、8 heuristic。
- 最终GPU0/1无compute context；全部登记PID+出生时间观测无活动owned。child实际return/reap见原receipt；父进程退出码未知，只记已观测不活动，不补exit0。
- 新namespace存储1844777字节，另有workspace唯一DSO，见最终观察；新增总量远小于512MiB。模型与fixture未复制。
- 仅新optimized组件；本机88项保护核对及远端86已部署项核对，结果分别见FROZEN-LOCAL.json/FROZEN-REMOTE.json。原远端缺state/rules的88项检查false不改。
- 四原语全部未通过原逐位门；不做异步整层集成/性能扩测/参数网格。attention契约仍未完成。没有新训练、完整机器人任务、TEST、第二环境、30层或租卡。
