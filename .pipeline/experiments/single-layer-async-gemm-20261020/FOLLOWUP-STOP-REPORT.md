# 本次有界后继的资源与止损

本次新授权承接O1–O4，旧M39登记240/900冲突及原FAILED不变。新工作沿M39归档，没有新增研究ID。

|阶段|实际完整block / 帽|终态|
|---|---:|---|
|O1数值|261 / 300|有限9点数值/状态通过|
|O2原语效率|390 / 400|数值通过，速度NO-GO|
|O3行组ready|432 / 500|交接机制成立，净速度NO-GO|
|O4最终引擎核验|297 / 300|两路线有限数值通过，速度NO-GO；完整矩阵未执行|
|合计|1380 / 1500|不继续扩大搜索，不集成机器人|

[去重原件](RESOURCE-JOBS-DEDUPLICATED.json)以PID+birth合并43条owned根进程记录，GPU worker生命周期合计324.748秒，CPU编译1669.927秒，CPU生成另列；**CUDA active时间UNKNOWN**。这些不是整个任务墙钟：同时包含源码分析、网络与环境准备，保守从12:20:28授权到14:12:01最终资源快照约1小时52分，仍低于8h总帽。各原始stage/job起止保留，不能把GPU几分钟解释为整个研究几分钟完成。

O3 timing receipt意外重复嵌入dependency build/gate记录，原件保留，汇总按身份去重，未双算。旧norm缺字段、不同bias/归约假设失败、两轮同名DSO初测INVALID、两次PID命名空间预检失败、一次gate终态/进程清理竞态均保留。脚本捕获异常后exit0不代表科学通过。

最初missing-parent launcher未产生GPU worker，精确CPU耗时UNKNOWN，不补造收据。上述接续未重置旧起点或改旧FAILED。

[最终现场](FOLLOWUP-FINAL-RESOURCE.json)：43条已登记根进程均无仍匹配身份；未见本命名空间活动guard/worker，两卡均0MiB、0%且无compute进程。此状态仅是该时间的观察，不保证未来他人不启动任务。未清理他人进程、未改驱动/时钟/系统服务。

各新证据目录实际9–55MB，均低于各1GiB帽。大型原始fixture仍用原共享引用，没有复制30层KV；DSO、工具链与私有供应商SASS不公开。

停止扩大ready队列与tile搜索：当前计算效率差距仍大，且实际交接增加成本。O4仅完成最终引擎1/7状态对应真实输入核验；B6不准入、强cache/Graph矩阵缺失，不把未完成项写PASS。四质量权限false，不自动启动机器人、TEST、受阻45chunk或新训练。
