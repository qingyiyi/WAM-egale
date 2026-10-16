# 投机采样：单worker全层 C/HC v4 CPU修复通过，GPU资源冲突停止

2026-10-12T15:16:38+08:00（Asia/Shanghai）

## 本轮做了什么

- 用户允许后台修包装bug，并明确开启只读subagent讲解；未改变公共模型、batch14、原SDPA、权重、阈值、bonus，不训练或重构。
- 新建私有v4及独立新job，保留v3 FAIL。修复旧负例只清理V2 flags导致漏新版本授权的环境隔离；CPU旧job路径断言改为精确新fixture拒绝分类。新增四种真实CLI在Torch/公共server导入前拒绝证明。
- 补默认关闭mode auto交接：CPU全通过后后台自动检查资源/启动一次GPU；CPU失败不交接，GPU入口仍重新核对新job CPU gate。独立session、SIGHUP ignored、stdin DEVNULL、日志落盘，本机熄屏不导致中断。
- [输入形状、CFG两份历史与当前十四份KV、prompt共享边界及后续接口](SINGLE-WORKER-VERIFY-SHAPES-20261012.md)，由只读subagent取证、主agent轻量复核并归档。不是GPU实验结论。

## 已通过

|关卡|结果|
|---|---|
|监督器/入口/自动交接回归|51/51|
|持久化计数与真实worker终止|44/44|
|真实生产loader正负回归|39/39|
|既有真实接口/数值回归|81/81|
|合计|215/215，无CUDA初始化|
|同实际worker FP32 probe|生产modules.model全30层R/C/HC，87次CPU host返回，PASS|
|真实训练target GPU生产接口|全30层原forward/原SDPA，PASS；不是C/HC输出等价性通过|

CPU预检从15:00:08至15:05:08约300秒。加载的是CPU小随机checkpoint作接口验证，不将它冒充真实target GPU正确性。11个数学/API/计数实现文件逐字不变，resource_guard仅版本注释不同/AST相同；全部Runtime方法AST不变。

## GPU为什么停止

- 15:05进入资源等待；15:08:24在15个稳定空闲样本后启动唯一GPU worker PID1664549。
- 15:08:54完成真实target全30层接口准入。15:08:56 NVML出现非本任务PID1665051，独占资源关失效；监督器只终止本任务，未信号对方。
- 从GPU模型spawn到清理共35.994813秒，无限时超时。不是SSH断开，也不是外部用户kill我们的worker。
- 后续只读工具观察到另一用户wangcui/UID1006运行连续smoke/cuda-graph/throughput任务，说明有外部活跃GPU工作负载。但旧PID1665051已退出，其UID/argv未保存，不能把之后观察到的身份强行回填到旧PID。

## 计数与性能边界

持久化journal只有OPEN和BEGIN capture两条；校验writer PID/UID/start、hash-chain/plan身份后，预算意图/已确认host返回/未确认分别为0/0/0。此处0有日志证据，不是父账本未ack的旧0。capture预留52不是执行52。未写CLOSE标记，状态为INTERRUPTED_VERIFIED_PREFIX，不是正常整关成功。

恢复/捕获未通过，C/HC GPU数值、十二类失效、13行隔离、额外显存、warmup、六块OFF、ON及全verifier净收益全部NOT_RUN。不能报优化后的速度。0完整episode/EGL/完整policy chunk/环境步/训练。

## 当前思路和下一步

现有C/HC缓存的是14行展开历史，并非真正两份prefix共享kernel。后续可讨论prefix [2,2144,24,128]、current [14,16,24,128]、rowmap [14]，但必须维持prefix+current同一softmax、候选隔离和有效slot/失效语义；prompt/text KV是独立优化方向。

私有bug已经修复并通过CPU；当前需要协调连续独占GPU窗口，不继续修改公共模型/attention或反复在短暂空档抢跑。用户已授权持续修复，不再为每个CPU小修确认；后续新job不得复用本FAIL或弱化资源监控，并须把本轮已耗35.995秒计入累计≤900次/45分钟GPU预算。窗口明确后先核对剩余预算/新job预检，再恢复→数值/失效/隔离→六块OFF→独立ON。未获速度结果前不宣称加速，不训练/mega或自动共享prefix改造。

当前所有本任务CPU/GPU worker/controller均已退出，所属端口已释放，没有后台自动重试或队列。已异步询问用户连续独占窗口预计何时开始；不改GPU Default模式，不占满显存抢卡，不干预其他用户。

## 证据

[任务简报](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/BRIEF.md)；[CPU gate](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/preflight/CPU-GATE.json)；[原始FAIL](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/FAILED.json)；[只读审计](../../../review/OMISSIONS.md)；[恢复计数](../../../投机采样/results/runtime/p0-single-worker-chc-v4-20261012-145820/RECOVERED-COUNTER.json)；[形状说明](SINGLE-WORKER-VERIFY-SHAPES-20261012.md)。

829冻结文件、21已执行私有源码、旧v3 FAIL原样保留；当前v4 FAIL的全部原始文件已另存只读SHA清单。自审，不是独立GPU审计。
