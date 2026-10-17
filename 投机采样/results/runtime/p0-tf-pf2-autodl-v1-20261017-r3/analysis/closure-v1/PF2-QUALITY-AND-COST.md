# 冻结 TF_ON：连续与 PF2 的有限 DEV 结论

OBSERVED：固定科学快照4f04b35，实际独立执行源码9acc37cee9ae9b62b8311f1f8dbb01bffa37a86a。六对十二局全部完整，初态物理/controller/RNG/双相机保存组件6/6严格匹配；原始阈值、权重、normalizer与树未改。此为暴露过的风险富集DEV，不是独立TEST。

|pair|task/init|分层|连续 C：终态/实际命令|PF2 P：终态/实际命令|
|---|---|---|---|---|
|1|1/29|risk|失败/796|成功/264|
|2|1/36|control|成功/260|成功/405|
|3|3/26|risk|失败/796|成功/241|
|4|3/36|control|成功/228|成功/225|
|5|4/26|risk|失败/796|失败/796|
|6|4/36|control|成功/234|成功/235|

C成功3/6，P成功5/6；风险0/3→2/3，对照3/3→3/3。本轮真实rescued=2、lost_success=0、双方成功3、双方失败1。没有用历史C替代本次C，也没有原Target/MLP新实验臂。实际命令总数C3110/P2166；失败800罚分分别3122/2170。风险罚分2400→1305，对照722→865；各失败实际796命令，罚分800，未补造四命令。

共同成功3对的P/C长度比：P50=1.0042735，P90=1.4470085，max=1.5576923；分位数为线性index=(n−1)q。task1/init36从260变405，是明确的长度反例。风险救回不能抵消控制组罚分、P90和max失败；task4/init26的PF2仍失败，P风险全成功门也失败。

**登记投入结论：NO_GO。** 详见../result-v1/INVESTMENT-RESULT.json的逐门重算；两项成本门因可比性未知保持UNKNOWN，不能改成PASS。

OBSERVED：探索性ΣT_round/Σg为C=0.040679905、P=0.052514468秒/合法更新；P/C约1.2909。这里包含实际draft/anchor/verify/pack/接受/bonus/末尾action工作。内部g不是环境步。全体完整policy/chunk为3.612273→4.192598秒；共同成功的policy总成本165.899302→226.989656秒（P/C=1.368238）。全体policy总成本711.617718→582.771179秒，包含失败，PF2少了两条跑满失败。B2/B14 forward分别保留typed账本，不将8643→7411调用数称为加速。

UNKNOWN：稀疏资源采样包含无法映射到容器出生身份的GPU PID，成本连续可比性未建立；上述是带审计AutoDL观察，正式E2E speedup、相对原Target速度、总体LIBERO成功率、非劣性、完整GPU活跃时间均UNKNOWN。

HYPOTHESIS：PF2可能缓解部分持续投机风险；它同时改变draft提交频率和B2/B14 teacher路径比例，当前实验不能单独证明误差积累或某一数值算子是根因。

RECOMMENDATION：本轮结束，不把连续或PF2设为已验收的最终TF方案，不上全量训练/120/TEST/正式性能，不搜PF3/4、不改tau。TF64结构/API继续作为研究接口。下一项应是另行登记的教师路径因果诊断，而非无依据追加训练。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧TF_ON NO_GO、旧FAILED/retention及22收据缺口全部保留。
