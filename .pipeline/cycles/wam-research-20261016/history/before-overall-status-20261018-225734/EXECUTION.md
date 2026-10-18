# 最近执行检查点

最新用户取消进一步开发迁移，自行结束遗留模拟器。只读核对root的Aurora gem5/QEMU，主PID3658238，PGID3658199，共40gem5+2QEMU；未发送信号。临时身份见M23 LOCAL-SIMULATOR-IDENTIFICATION.json，不改原评测。

## 2026-10-18 22:39：M23新机Codex安装完成、登录待用户

两机只读资源核验：本机16逻辑CPU近满，PSI some约94.9%；新机45核聚合配额、180GiB，CPU使用约4.74核、PSI约1.2%，双A100-SXM4-40GB与本机单卡同型号/400W但驱动不同。支持CPU工作迁移建议，未实测工作负载倍数。新机两卡原评测继续。

在~/cxz内复用本机完整原生包安装Codex0.162.1，入口~/cxz/bin/codex；配置/缓存/临时范围固定。版本/帮助通过，login status未登录，未迁凭据/会话或最新算子。约431.51MiB，传输安装123秒，GPU0；[报告](../../experiments/remote-codex-setup-20261018/RESULT.md)。不将询问迁移写成全部工作已迁移，旧门和预算不变。

## 2026-10-18晚：M22数值与新性能已齐、GPU释放

去除fast_math且原kernel数学/vendor不改的新v2，108attention/54block、六完整30层与三短中长完整canonical policy配对逐位及声明有效状态/RNG通过；CPU27保存输出独立复核一致。旧M21 standalone完整六query非逐位、三容差失败保留，不能由新通过追溯更正。原tile/保护源/默认/四false不变。

用户再次确认GPU空闲后，新54点/684单元/18360Event及等量wall采齐；self/cross Graph-core新/Torch中位1.00144/1.00193，block .98465，9/54慢>5%、最大1.34367。真实安装60op、六full30三轮交错各一forward，中位.99393、范围.73927–1.06654，两例慢>5%。没有完整policy或机器人E2E计时，不宣称普遍稳定无退化/新Mega收益。

CPU15同源检查（九保存输入、六完整原始输出）全等，无CUDA；r5 warmup原始/新输出逐位及全部有效KV通过。r1/r2正确性22.950/592.107秒；r3性能319.481秒；r4准入整数下界冲突，0GPU/0forward失败保留并计.140秒；修正supervisor后独立r5完整160.103秒。M22总1094.781/1200秒、新增约406.87MiB≤512MiB、1002 full target forwards，CUDA active未知，自有PID+出生核实退出。不再用剩余105秒扩作业。

用户确认窗口，边界无他人context，但0MiB下NVML24–25%异常未解；单进程少重复，资源/统计限制与全部慢轮保留。后续优先定位block慢点/host/clock/allocator等假设，未证明根因，不盲改内核。

[新性能](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md)；[数值](../../experiments/operator-torch-alignment-20261018/RESULT.md)；[自有清理](../../experiments/operator-torch-alignment-20261018/OWNED-CLEANUP-PERFORMANCE.json)。

## M16/M17和迁移

Long首轮三臂各500完整，原配对FAILED/50init曝光/旧NO_GO与四false保留。Object本机16/30后自然释放，余14/ON/final500迁远端等自然槽位；原匹配Target200迁移增量独立记账。Goal/Spatial原授权全量继续，22:14核验Goal208共同组/690局、Spatial202/683，四model仍满载；Object余14等槽。已启动每5分钟CPU只读巡检，详见M17快照；旧19:08与17:25数不当实时。BAAI只限~/cxz，AutoDL不再SSH。原M16/M17起点/绝对截止在PLAN，不重置。

## 历史

压缩前完整执行记录见history/EXECUTION-through-M22-performance-20261018.md；更早四文见history/*-before-operator-final-20261018-204358.md。M21各失败/准入/r5用户收尾PARTIAL与原1649.274/1800秒均保留。完整原始job与来源见各Experiment，当前仅语义入口。
