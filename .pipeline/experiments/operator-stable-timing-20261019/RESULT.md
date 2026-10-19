# 新授权算子计时接续结果

两独立进程均自然exit0，完整覆盖原版/baseline/optimized、B2/B14与短中长KV的block和实际30层。每臂额外2次预热、9轮循环交错，全部原始样本保留。计时输入在相同进程先检查数值/有效状态；本轮共108项检查逐位通过、max_abs=0，含write1/2、有效KV、输入、当前RNG与安装还原。B14 full30有保存历史gold，B2 full30仍明确为真实verifier root CFG0/7派生诊断，不扩称原始顺序轨迹、FSDP或完整policy。

本轮432 full30、444 block，GPU worker生命周期131.338秒；两进程各51次资源采样，未检出外来GPU PID。全部自有GPUworker退出。原M25的150/168、137.306秒和FAILED/PARTIAL保持原身份，未重置账本。

|输入|进程1 baseline/原版|进程2 baseline/原版|进程1 optimized/原版|进程2 optimized/原版|
|---|---:|---:|---:|---:|
|short-B2|0.9796|0.9539|0.9535|0.9892|
|short-B14|0.9701|0.9852|0.9984|0.9735|
|middle-B2|0.9630|0.9883|0.9763|1.0218|
|middle-B14|0.9993|0.9907|0.9668|0.9930|
|long-B2|0.9690|0.9915|0.9932|1.0065|
|long-B14|0.9826|1.0058|1.0081|0.9918|

完整30层baseline相对原版中位耗时为0.9539–1.0058，optimized为0.9535–1.0218；即本轮完整模型最慢中位回归分别0.58%和2.18%。这是本轮描述性结果；不能只按最慢回归忽略更快点及波动。block baseline为0.9710–1.0272、optimized为0.9750–1.0162。严格±2.5%筛查并非全点通过（包含更快的差异），不把“接近”写成旧门PASS。

启动前/退出后均零显存、无可见compute或graphics PID，但NVML持续24–25%利用率异常原因UNKNOWN。因此未获得正式无争用证书；本轮不作为机器人E2E或正式投机性能。两进程较上一轮接近并不能证明所有输入、所有环境稳定。

投机1875源/配置复核无修改，保护state/cycle/rules与原tile不改。本次仅共用工具增加登记控制的预热/轮次和资源采样；两份数学实现未变。下一步可冻结当前可替换基线，先profile真实完整block，按占比选一个子图，另登记开发/验收；不直接大融合。

- [全部原始计时](TIMING-ALL-SAMPLES.csv)、[每进程/每点中位表](TIMING-SUMMARY.csv)
- [进程1](process-1/RESULT.json)、[进程2](process-2/RESULT.json)、[资源与清理](RESOURCE-AND-STOP-REPORT.json)
- [新授权与预算](REGISTRATION.json)、[输入身份](INPUT-MANIFEST.json)、[源码](SOURCE-MANIFEST.json)、[投机保护](SPECULATIVE-SOURCE-GUARD-RESULT.json)

engineering_quality_screen=false、independent_test_complete=false、noninferiority_established=false、formal_performance_permission=false。
