# 最近执行检查点

最新整体状态见[STATUS](STATUS-20261018.md)与远端派生快照；旧22:14数字仅为当时记录。

最新用户取消进一步开发迁移，自行结束遗留模拟器。只读核对root的Aurora gem5/QEMU，主PID3658238，PGID3658199，共40gem5+2QEMU；未发送信号。临时身份见M23 LOCAL-SIMULATOR-IDENTIFICATION.json，不改原评测。

## 2026-10-18 22:39：M23新机Codex安装完成、登录待用户

两机只读资源核验：本机16逻辑CPU近满，PSI some约94.9%；新机45核聚合配额、180GiB，CPU使用约4.74核、PSI约1.2%，双A100-SXM4-40GB与本机单卡同型号/400W但驱动不同。支持CPU工作迁移建议，未实测工作负载倍数。新机两卡原评测继续。

在~/cxz内复用本机完整原生包安装Codex0.162.1，入口~/cxz/bin/codex；配置/缓存/临时范围固定。版本/帮助通过，login status未登录，未迁凭据/会话或最新算子。约431.51MiB，传输安装123秒，GPU0；[报告](../../experiments/remote-codex-setup-20261018/RESULT.md)。不将询问迁移写成全部工作已迁移，旧门和预算不变。

## 2026-10-18晚：M22数值与性能齐、GPU释放

新v2只去fast_math，108attention/54block/六full30/三完整policy有限逐位与有效状态通过；CPU27输出/15同源核对一致，原M21失败保留。新54点/684单元/18360 Event及等量wall：self/cross中位1.00144/1.00193，block .98465但9/54慢>5%；六full30中位.99393，2/6慢>5%。没有机器人E2E或所有输入证书。空context非零利用率及少样本保留。

r1/r2/r3/r4/r5全部原件保留，r4准入下界错误在GPU前失败、计.140秒。总1094.781/1200秒、1002 full target forwards、新增406.87MiB，自有PID已退出。完整数值/计时/清理见[新性能](../../experiments/operator-torch-alignment-20261018/performance/RESULT.md)；不借105秒扩任务。CPU模拟器后来退出不追溯解释本次慢点。

## M16/M17和迁移

Long首轮三臂各500完整，原配对FAILED/50init曝光/旧NO_GO与四false保留。Object本机16/30后自然释放，余14/ON/final500迁远端等自然槽位；原匹配Target200迁移增量独立记账。Goal/Spatial原授权全量继续，22:14核验Goal208共同组/690局、Spatial202/683，四model仍满载；Object余14等槽。已启动每5分钟CPU只读巡检，详见M17快照；旧19:08与17:25数不当实时。BAAI只限~/cxz，AutoDL不再SSH。原M16/M17起点/绝对截止在PLAN，不重置。

## 历史

压缩前完整执行记录见history/EXECUTION-through-M22-performance-20261018.md；更早四文见history/*-before-operator-final-20261018-204358.md。M21各失败/准入/r5用户收尾PARTIAL与原1649.274/1800秒均保留。完整原始job与来源见各Experiment，当前仅语义入口。
