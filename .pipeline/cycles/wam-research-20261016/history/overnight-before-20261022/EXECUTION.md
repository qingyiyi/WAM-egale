# 最近执行

## 2026-10-21T23:47:01.562039+08:00 接续跨任务完整核验

- 遵用户当前版本继续运行，远端GPU1 guard PID797230已启动；首次READY与完整任务推进见[状态](../../experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021/STATUS.json)。
- 固定2组×2工程臂，旧初态/Target/reset bank、全程观察/状态/RNG/命令/RPC审计；原结果保留，不构成新TEST。
- 旧M40累计52520.882秒＋当前活动作业，原预算/截止未变；本机当前不派发。
- Maintain保存register/shared优先及本机≤1/4回传远端两条长期要求。

## 2026-10-21T23:02:48.466096+08:00 两项工程优化收口

- 原版/相机/预热三个完整成功case分别243.634/233.767/237.787秒，有限全观察/状态/RNG/命令/RPC审计通过。
- 原组合闭包引用错误0teacher/0动作PARTIAL保留；用户明确授权新namespace单局补跑，修复后225.782秒、全部同样审计通过。相对原版本轮减少7.33%，不是稳定总体或投机E2E结果。
- 原作业781.254秒＋补跑255.945秒；M40累计52520.882/86400，原硬截止不变。两作业owned为空，原固定source未改。
- 两项/组合为可选工程候选；其他任务、长失败及重复计时待做，1800局/160小时未启动/未批准。

[结果](../../experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/RESULT.md)、[资源](../../experiments/robotwin-pilot-20261020/runtime-opt-combined-fix-20261021/RESOURCE-AND-STOP-REPORT.json)、[前执行](history/runtime-opt-before-20261021/EXECUTION.md)。

## 2026-10-22T00:53:00+08:00 夜间对齐与M40终态

- M40跨任务4局完整终态/清理，见[RESULT](../../experiments/robotwin-pilot-20261020/runtime-opt-cross-task-20261021/RESULT.md)，失败保留。
- M39 Self-Q通用路由9签名逐位通过，但两进程整层−0.14%～+0.23%无稳定收益，未选用。
- point静态分配9签名逐位通过，强基线计时进行中。原动态Torch Graph与shared扩展路径两次实施计时失败均独立保留，固定静态布局及绑定现有扩展后重测；未改数学或质量权限。

## 2026-10-22T01:51:26.059486+08:00 夜间阶段检查点

- 28阶段×3KV×两进程控制完成；隔离资源仍固定整层100096B，不能混称最优独立原语。两进程输出/有效KV/输入/RNG通过。
- 原版小投影grid24×2×2/128线程；FFN1 grid56×2/256线程。NCU在本机hlj新任务进入后启动拒绝，0GPU生命周期。远端无NCU权限，不改驱动。
- M39失败/负结果、08尾部微差、修复与候选选择均在night namespace保存；M40已清理且原01:26截止已过，无新增队列。
