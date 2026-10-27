# FastWAM 夜间推进清单

登记 2026-10-27T00:43:27.787745+08:00，先于任何新闭环结果选型。用户授权见[长期决定](../../memory/global/requirements/decision-FastWAM夜间三卡推进并在筛查通过后自动全量三臂评估.md)；[固定扩展登记](OVERNIGHT-EXTENSION-v1.json)保留原预算与失败。

- [x] 三台机器固定权重、统计量和组件齐备，见 THREE-COPIES-COMPLETE-20261027.json。
- [x] 本机第一次模型加载 SIGKILL 留证，349.313秒计费；第二次采用 T5 meta+assign 与 checkpoint mmap，完整键检查通过，官方 forward 未改。
- [x] 本机4个完整 canonical、预处理/动作/compile一致性与可视审阅，4/4成功；见PILOT-REVIEW-attempt02-r2。
- [x] 三张A100分片100 FIT/10 CAL；最终3000步小TF，新CAL Q80 tau=0.01541235298冻结。
- [x] 同机20组/60局完整：T17、TF19、PF18成功；PF2固定S,S,T。
- [x] 60局G=10/提交前缀及三机真实B7核验通过；像素差逐组报告，全部失败保留。
- [ ] 达到登记工程条件后自动全量：先Long500组三臂，再其他suite同配方采集/校准/筛查及全量。每suite10任务×50初态，不称独立TEST。
- [ ] 5090与算子会话交接独占窗口，测真实trained TF与PF2完整成本；不乘算子倍数。
- [ ] RoboTwin单独核查专用权重/统计量/环境，缺失不写0%或伪造复现。
- [ ] 汇总逐任务SR/区间、配对胜负、共同成功trace、tau/接受、5090单chunk与等长及自然episode成本，更新HW。

算子会话00:35登记新persistent窗口至04:35；投机暂不占5090，待真实draft就绪后交接。A100耗时仅资源记录，非独占性能结论。四科学权限保持false。
