# TF可用性：本轮有限验证

- [x] 读取最新证据/原协议/两机资源，确认当前TF质量未通过。
- [x] 独立训练入口：同child23起点、同旧配方再500 updates，固定结构/normalization/接受器。
- [x] 新源码CPU独立复核通过并已执行；不改旧schema或元数据。
- [x] 两机各一个新DEV初态，T/当前TFchild/继续训练TF/当前MLPchild各10局，共80完整episode，原horizon800。
- [x] 严格配对与全部失败已保全；冻结原聚合实际PARTIAL，独立80描述完成，TEST40–49未读。
- [x] 条件判定：新FIT/CAL与TF-on/MLP-on未触发，继续HOLD，不追加采集。
- [x] 终态/预算/自有进程清理、报告及HW短入口更新完成，全部本地链接核验通过。

绝对停止12:43:35、清理12:48:35；每机最多3 GPU小时、两机合计6 GPU小时、每机新增3GiB。优先结论，必要简单baseline留后续主实验，不同时展开系统优化。

启动前修订：原PLAN.json保留，active为PLAN-v2.json；固定加入MLP child seed23满足原matched-MLP工程门。未查看新DEV，未启动GPU。

TF_CONT实际CPU终态：125.154秒，主seed23固定500更新完成，完整20237 CAL新loader/原decode同CPU逐位、normalization保持；仅新DEV候选，不是机器人质量PASS。

09:53恢复登记：本机r3监督进程异常消失，工具会话返回143；具体退出原因未核实。原13条完整与中断任务全部保留PARTIAL，独立clock已清理所观察的自有后代。r4实际CPU准备通过：同162源码、3权重及全部40初态/任务；拟独立会话完整重跑，不混用旧13或挑最好轮。累计本机3小时与3GiB仍从r3计，最多此次恢复，不无限重试。参见LOCAL-INFRASTRUCTURE-RECOVERY.json；新采集D01仍HOLD。

最终有限工作已收口：80条完整；T19、child19、CONT20、MLP20。数值描述六门通过，像素56/60、主CONT19/20；原远端FAILED/原aggregate PARTIAL保留，四质量权限false。固定13恢复复核仅10条完整commands一致、12条success一致，旧13不混80。保留TF64与本次权重用于前瞻扩大验证，先核重复性；无新GPU/SSH，AutoDL可由用户关机。详见TF-USABILITY-RESULT-20261016.md。
