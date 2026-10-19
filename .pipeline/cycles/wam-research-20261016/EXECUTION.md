# 最近执行检查点

## 2026-10-19T10:53:24.163097+08:00：短KV full30通过、资源中断与原预算接续

原版/baseline/optimized短KV block、write1/2与真实30层共22项输出/全部有效KV/输入/RNG逐位通过。24 full30、30 block、37.297秒后发现其他GPU进程，原PARTIAL/计时未获放行保留。新continuation-1使用同一起点/13:14截止和余量；baseline两独立进程三轮，optimized首进程计时、两进程数值，总full30最多150。[登记](../../experiments/operator-consolidation-20261019/continuation-1/PLAN.md)。

## 2026-10-19T10:47:48.521957+08:00：算子两份实现迁移与清理

用户要求简化代码并先完成30层与性能，随后明确投机侧代码禁止改动。已通过Maintain保存。独立baseline/optimized完成，block forward AST一致，参数/cache原位安装与还原、新CUDA库CPU导入过；16旧算子目录、26444694字节已删除，重复build另5044712字节；308源先存Git a0ed2d8，1875投机源清理前后逐字未改。原tile被shared-prefix构建脚本引用而保留。GPU有其他任务，supervisor CPU等待原13:14:26截止；没有新GPU结果。准备阶段三处脚本错误均发生GPU前、已修正保留，未占forward预算。[实际清理](../../experiments/operator-consolidation-20261019/CLEANUP-RESULT.json)。

此前所有检查点逐字保存在[本次history](history/M25-before-operator-consolidation-20261019/EXECUTION.md)，包括M24有限GPU、旧AA失败、Object新14修复与训练/全500接续。原FAILED和资源记录不覆盖。
