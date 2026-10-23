# 当前研究入口：局部warp有界实验完成（2026-10-23 16:34）

请优先读[REVIEW_RETURN](../.pipeline/experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)、[完整阶段表](../.pipeline/experiments/local-warp-pipeline-20261023/STAGE-TABLE.md)、[最终同轮对照](../.pipeline/experiments/local-warp-pipeline-20261023/FINAL-COMPARISON.md)。源码固定提交1ef5b83af38d523b835f2511018334ba99a4b694；算子研究/optimized在此公开快照显示L13，但两机原运行入口未部署该版。

FFN2局部准备主线INCONCLUSIVE，不采用；LN1/LN3调制融合L13小幅改善0.4%–0.7%，中长KV仍慢强guarded约3.4%–3.6%。有限225调用/候选通过，不证明全部输入或机器人质量。四科学权限及production_admitted仍false。全部本轮GPU0 owned已退出，GPU1原RoboTwin继续；16:33配齐125组三臂，微波炉20/100组，等100组完整后才查因。原失败、旧预算起点/截止和retention不改。

这是新完成的有限算子实验，替代此前“等待网页版设计”的当前状态；下方TF暂停72是历史科学快照，不改其身份/结果。此前入口保留于[history](history/current-before-local-warp-20261023/START_HERE.md)。

---

# 最终同轮完整层比较

|KV|A ms|L1 ms|L13 ms|D ms|G ms|A/L13|D/L13|G/L13|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|288|0.872525|0.868147|0.866534|1.128986|0.880717|1.0069|1.3029|1.0164|
|1152|1.042637|1.039027|1.036672|1.920973|1.000627|1.0058|1.8530|0.9652|
|2160|1.053517|1.050189|1.049011|3.146394|1.014605|1.0043|2.9994|0.9672|

T4-final1/2：两独立进程，各档各臂150次预热、24轮交错样本×10次Graph replay。表为两进程中位数均值；每进程与IQR/配对变化见T4-final-SUMMARY.json，逐样本见原TIMING JSON。独占GPU0；GPU1仍有RoboTwin，不把整台机器称独占。GPU事件边界为warm完整block CUDA Graph，分配/首次缓存构建不在warm延迟，原始TIMING另存cold构建和展开成本。双方同Cross最终K/RMS/V预计算、相同真实B14/Q16、相同有效历史与私有suffix。

A固定319ac6a独立persistent；D原Torch静态dense-history；G现有可用guarded执行路径，两参考均经输出/有效状态核验。L13只采用LN1/LN3的调制融合，无BC主线组合。未测跨硬件、30层、完整chunk或E2E。

相对A三档延迟降0.687%/0.572%/0.428%，与前轮方向一致，幅度小，不声称统计显著。G/L13中长小于1：L13仍慢G3.602%/3.391%。只报D的1.30–3.00倍会掩盖强基线差距。

本轮所有臂（含自有算子）均为外层CUDA Graph；普通双方eager对照未测。见[计时澄清](../.pipeline/experiments/local-warp-pipeline-20261023/MEASUREMENT-MODES.md)。
