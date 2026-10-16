# 投机采样：中断计数持久化与真实 worker 终止 CPU 结果

2026-10-12T11:24:32+08:00（Asia/Shanghai）

## 结论

**本步仅CPU修复及验证完成，首次全部通过；没有新GPU、排队或训练。**旧GPU的丢失次数仍未知，不回填或改写旧FAIL。

## 已实现

1. 独立、默认关闭的任务修订。调用前写`BEFORE`并fsync，返回后写`RETURN`并fsync；记录阶段预算、失败不退款，原895/900 caps和RAM Ledger数学保持不变。
2. 记录有序号/前项哈希/checksum、CPU job/plan SHA与PID/start_ticks身份，单writer独占创建不允许重开续跑。写失败阻止进入公共数学。恢复校验已持久化完整前缀；缺失/无有效header为UNKNOWN，完整记录损坏拒绝，残缺尾部显式标注，不套父旧0、不修改raw。
3. 实际Runtime/counted/dispatch/serve接日志；TERM handler只设停止标志，不重入I/O。当前已预算调用返回后确认，在下一安全边界写CLOSE/EXIT；若卡住升级KILL，仍能只读恢复账本和未确认项。
4. CPU CLI只加载真实未mock生产loader的随机小checkpoint；非CPU入口与Runtime路径无条件拒绝。没有新GPU监督器/队列/授权，GPU模式未改。

## CPU 验证

|验证|结果|
|---|---|
|新增日志/终止回归|**44/44 PASS**：34日志、边界、budget、损坏/尾部、身份、I/O故障测试 + 10实际worker/数值测试|
|原真实生产loader正负例|**37/37 PASS**，原测试文件逐字不变|
|原接口/全层C-HC回归|**81/81 PASS**，原测试文件逐字不变|
|新实际worker FP32 probe|全30层、29变化query，R/C/HC逐bit一致，history/metadata/RNG不变；日志87意图/87返回|
|新持久化BF16 probe|同真实生产loader、全30层29变化query R/C/HC逐bit一致，history/RNG不变；日志87/87|
|边界|0 CUDA初始化/禁止初始化调用、0新GPU/排队/episode/EGL/完整policy chunk/训练|

CPU-001 首次通过，**162项 + 实际同worker FP32 probe**，总 **292.089秒（约4分52秒，含子进程加载）**。CPU随机夹具不是trained target GPU验收。

## 真实 worker 终止结果

确实启动本修订worker子进程，运行实际`Runtime.serve`与生产`modules.model`的30层原batch14/attention。只在本CPU plan的显式test fault中通过实例hook/边界屏障暂停；公共数学未替换。信号仅发给验证了PID/start/UID/PGID/argv的自建CPU进程，全部退出，不信号他人。

|场景|预算意图|已确认返回|未确认|整段ack|EXIT|
|---|---:|---:|---:|---|---|
|normal|29|29|0|有|有|
|term-idle|0|0|0|无|有|
|term-mid|2|2|0|有|有|
|kill-mid|2|1|1|无|无|
|kill-begin|0|0|0|无|无|
|kill-before|2|1|1|无|无|
|kill-return-gap|2|1|1|无|无|
|kill-pre-ack|29|29|0|无|无|
|term-kill|2|1|1|无|无|

- `kill-mid`：第二次真实前向内部被KILL，恢复2/1/1，而父尚未收到命令确认仍是0。
- `kill-before`：意图已持久化但第二次尚未进入数学，仍2/1/1；不能把预算意图误当已经真实调用。
- `kill-return-gap`：第二次CPU forward已返回但RETURN未写，仍2/1/1；不能猜已确认返回=2。
- `kill-pre-ack`：29个返回记录已写但整段ack未写，恢复29/29/0，**只说明计数，不据此标整段验证PASS**。
- 有效OPEN/BEGIN的idle/尚未调用场景可证明真正0，区别于旧GPU缺失journal的UNKNOWN。

BEFORE是预算消费，RETURN是**host forward返回确认，非CUDA设备完成**。强制终止时最后一个意图可能已调用/已返回，也可能没有，最多一个未确认；预算数和持久化确认可恢复，不声称最后一次数学状态总能精确知道。旧GPU的精确次数仍不可恢复。

## 不变、成本及未证明

- 371份前置源码/证据与146旧HW raw record SHA保持不变；15份已测私有实现有CPU-001源码快照。8份接口/数学/旧测试/ownership文件逐字复制，14个旧Runtime方法AST不变，原Ledger/caps AST不变。
- 公共模型/server、batch14、原attention/SDPA、大小模型权重、selector/阈值/bonus不变。受保护state/rules不改、cycle仍不存在；9份trained target文件stat、A_off和原fixture不变。
- 两次fsync/forward都位于原完整序列计时内，R/C/HC一致收费。没有把日志成本藏在OFF外；以后必须用接入日志的新R同负载对照，不能与旧无日志R直接混算。CPU时间不证明GPU日志代价可忽略，未给加速百分比。
- 0新GPU/排队/训练/mega；真实target GPU等价性、失效、显存、完整verifier成本、policy/成功率/环境步数/E2E全部未新增证明。自审非独立审计。仅投机采样，不混入TILE结果。

## 能否独占GPU

本用户PATH下未发现Slurm CLI；只读`nvidia-smi`核查A100当前是**Default共享模式**。本机帮助提供`3/EXCLUSIVE_PROCESS`选项：可另请管理员评估计算进程排他配置或正式调度器预约，但涉及共享卡全局访问，需另行授权并协调其他用户。账号在sudo组不等于本轮获得系统配置授权，本步未探测/使用sudo、未改模式、未装调度器、未填满显存抢卡。

仅靠每3分钟检查空闲和脚本文件锁，不能强制排除未遵守同一约定的其他用户，也不能保证45分钟窗口。将来即使有排他配置，仍应保留资源监控/失败停规则，并明确恢复共享配置的责任。

## 下一步与目标

下一步先单独确认独占方式/窗口和新GPU预算。建议仍1 worker、最多900次target前向、45分钟含加载清理、0episode/EGL/完整chunk/训练；确认后才将日志与中断恢复接入新默认关闭GPU入口/监督器，过新job CPU预检，再恢复/捕获→数值/失效/隔离→六块OFF→独立ON，失败停不补跑。当前不自动排队/GPU/训练/mega，也不复用旧FAIL。

目标不重构：可靠保存真实实验预算/中断证据，然后在可靠独占窗口验证长历史单worker全层C/HC的数值、失效、显存及**含构建/staging/检查/释放/持久化的完整verifier净成本**；有净收益才另讨论policy/冻结target闭环。

[新增44项](../../../review/OMISSIONS.md)；[CPU结果](../../../review/OMISSIONS.md)；[不变核查](../../../review/OMISSIONS.md)；[日志实现](../../../review/OMISSIONS.md)；[实际worker](../../../review/OMISSIONS.md)；[旧GPU中断复盘](SINGLE-WORKER-CHC-GPU-V2-RESULT-20261012.md)。
