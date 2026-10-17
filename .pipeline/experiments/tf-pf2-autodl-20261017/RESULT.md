# TF_ON 连续/PF2：六对完整 DEV 已结束

2026-10-17 13:09北京时间GPU作业清理，随后本地证据回传及CPU审计完成。**有限实验完成，投入结论NO_GO，研究仍active。** 科学父版本4f04b35不动，实际独立执行版本9acc37cee9ae9b62b8311f1f8dbb01bffa37a86a；未读取main替代固定快照。

|task/init|连续C|PF2 P|
|---|---|---|
|1/29 风险|失败796|成功264|
|1/36 对照|成功260|成功405|
|3/26 风险|失败796|成功241|
|3/36 对照|成功228|成功225|
|4/26 风险|失败796|失败796|
|4/36 对照|成功234|成功235|

十二局自然完整。C3/6、P5/6，本轮rescued2、lost0；不把暴露过的风险富集DEV外推总体LIBERO成功率。保存初始物理/controller/RNG/双相机六对匹配，前两S现场有效KV/RNG比较通过；reset/S0_before保留六个原native会覆盖的lazy scheduler字段例外，不声称所有隐藏状态或离线全部KV重验。

风险罚分2400→1305；对照722→865；全体3122→2170。实际命令总数3110→2166，失败实际796与罚分800分开。共同成功三对长度比P50/P90/max=1.0043/1.4470/1.5577；P风险全成功、对照罚分、P90/max门失败，救回不能抵消。两项成本投入门为UNKNOWN，不偷改PASS。

探索性ΣT_round/Σg为0.04068→0.05251秒/合法更新（P/C1.2909）；共同成功总policy成本165.90→226.99秒（1.3682），完整policy/chunk为3.612→4.193秒。资源PID容器身份未完全映射、连续可比性UNKNOWN；不是正式计时、原Target对照或E2E加速。

所有实验默认使用已验证完整512-token BF16 reset bank，本轮两臂十二次实际15–107ms；冷启动另列。runner实际20.89分钟；从原授权到终态累计1.962小时，六个server累计0.728764 GPU-worker生命周期小时，非CUDA active。r1导入失败与r2两个零步client失败/保守扣账完整保留。1267原证据文件约52.5MiB回传；临时249MiB资产包及20.1MB证据包双端清理，原件保留。

自有进程组EMPTY、guard CLEAN；后续85个登记身份无存活，两GPU各1MiB/0%、无计算进程，SSH已退出。**实例尚未关机，用户可以关闭。** 不为用完12h而另开实验。

- [完整质量与成本](../../../投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r3/analysis/closure-v1/PF2-QUALITY-AND-COST.md)
- [逐项登记投入门](../../../投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r3/analysis/result-v1/INVESTMENT-RESULT.json)
- [路由、状态与计数](../../../投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r3/analysis/closure-v1/PF2-ROUTE-AND-COUNTER-AUDIT.md)
- [资源、FAILED与止损](../../../投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r3/analysis/closure-v1/RESOURCE-AND-STOP-REPORT.md)
- [原始终态与完整报告](../../../投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r3/report-v1/RESULT.json)
- [早期两S接受代理分析](../../../投机采样/results/runtime/p0-tf-pf2-autodl-v1-20261017-r3/analysis/pre-treatment-score-result-v1/pre-treatment-score-analysis.md)

下一步建议：原权重连续/PF2均不作全量验收配方；TF64结构/API继续研究。首两S既有score未明显区分风险/对照，不能据六点拟合阈值。另行登记有界teacher路径诊断，区分B2/B14 teacher差异与接受draft的闭环作用；当前未启动，不自动追加训练、PF3/4、120、TEST或正式性能。

原engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission全false；旧TF_ON NO_GO、FAILED/retention/22旧清理缺证、全局A_off、保护状态与原tile均不改。
