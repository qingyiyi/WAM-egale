# M41已完成：5090投机速度与临时资产清理

终态：COMPLETE_WITH_LIMITATIONS_AND_RECORDED_FAILURES。不是质量通过。

- [五环境完整结果、时间账本和长度](PERFORMANCE-RESULT.md)
- [原始统计JSON](RESULTS.json)、[逐局CSV](EPISODES.csv)、[有限记录审计](AUDIT-RESULT.json)
- [主张边界](CLAIM-EVIDENCE-MATRIX.md)、[下次部署注意事项](DEPLOYMENT-NOTES.md)
- SOURCE-AND-RESULT-ENTRY.json绑定实际源码/输入manifest；RESOURCE-AND-CLEANUP-REPORT.md记录全部消耗、失败与清理。
- 原始数据raw/，不是公开仓库完整tensor镜像；不含模型权重。

5090本轮临时目录已核验回传后删除；174个出生身份核查，无自有活跃工作/GPU compute context。原算子工作区/环境保留，实例未关闭。最终1664文件88104231B经native rsync、尺寸清单与本地fsync核验，不运行独立模型SHA。

44组完整三臂用于主表，共132局；另有2条当前引擎完整非配齐Long局与1条旧FSDP完整Target保留在外，不混入主比较；两条已teacher执行但0环境动作的Long PARTIAL保留、不重跑。其余纯准备失败单列资源，不能算独立任务。

TF/PF2的真实action单位成本均下降，但质量、成功长度和环境/初始化成本决定任务收益。Goal有一次C失败/P成功；Spatial/Object有PF2长尾，Robo新增失败且分叉前隔离失败。四科学权限仍全false。原A100质量队列独立接续，未停止或扩大预算。
