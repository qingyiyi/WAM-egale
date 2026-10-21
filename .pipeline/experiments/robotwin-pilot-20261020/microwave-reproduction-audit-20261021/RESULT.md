# 两条原Target完整复现核查

11个关键源码/配置与固定官方版本逐字节一致；原VA_Server._infer没有安装投机Runtime。两条已曝光DEV完整运行，结果与旧Target相同。

|case|终态|提交命令次数|环境take_action_cnt|初态/首次两policy/RNG|所有命令|最大开门角rad|
|---|---|---|---|---|---|---|
|open_microwave-valid14-T|COMPLETE_ALGORITHMIC_FAILURE|1520|1500|逐项相同|全部相同|0.048515|
|open_microwave-valid16-T|COMPLETE_SUCCESS|368|363|逐项相同|全部相同|0.942481|

valid14失败时门仅达到约0.048515rad，远低于0.942478rad判据；valid16自然达到门限并成功。有限证据支持这两例旧Target的成败并非由投机Runtime插入而产生；不能排除其他初态、模型种子或论文运行差异，也不能解释TF失败的全部原因。未运行uncached完整bank对照。

计数补充：旧actual_commands字段来自wrapper对take_action的调用计数，本轮明确为submitted_commands；env.take_action在成功/原horizon后直接返回，所以提交1520次不等于物理执行1520步。旧字段/原数据不改，追加environment_take_action_cnt口径，后续报告必须分列。

资源：本轮GPU生命周期1191.369558095932秒，原M40累计51197.74909591675/86400秒；原截止不改。返回码[-15, 0, 0]，remaining_owned=[]。之后的完整成本profile单独计入同一旧预算。

SOURCE-AUDIT.json、COMPARISON.json、INPUT-MANIFEST.json、STATUS.json与每case commands/joints是原始依据。未追加质量N，四科学权限false。扩大100次仍待耗时诊断与新增资源确认。
