# 已批准的DEV90补启

用户明确回复：“允许修复后重启，保留原记录”。新namespace为dev90-20261021-r2；仅覆盖前一次模型加载前导入错误，不授权后续自动重试。

原attempt终态 `FAILED_OR_PARTIAL`，0teacher、0episode，GPU生命周期5.005915秒，原返回码1，remaining_owned=[]。原记录[STATUS](../dev90-20261021/STATUS.json)和server.log保留。

真实导入顺序已由setup绑定mesh_helper后再导入Runtime；CPU-IMPORT-REVIEW说明CUDA未初始化、未构造模型。补启CPU-READY为PASS；不将这些CPU核查作为机器人质量门。

原24小时累计不增，prior=25806.971332秒；绝对截止仍为2026-10-22T01:26:04.706686+08:00，每局1800秒并预留900秒清理。冻结30初态×T/C/P=90局，6任务valid14–18；原ON权重、tau、分区、108完整512-token BF16 bank不变。仅一个GPU1 worker，server/client各2CPU，按固定顺序自动接续。

四项质量/正式性能权限仍false。程序运行时不改冻结入口/清单；后续基础设施或首两S隔离门失败停止新局、保留失败，不自动重跑。
