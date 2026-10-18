# 新v2性能复测

状态：COMPLETE_PERFORMANCE_CAPTURE_VALIDATION_PENDING。54固定点，684单元，18360 Event和等量同步wall样本。

此处standalone字段指已去fast_math的新v2。与M21旧二进制时间分开，不借旧结果。

精度依据M22先前同代码的108attention/54block/六full30/三完整policy逐位验证；本计时过程不在计时区间重复误差统计。新捕获输入的CPU同源核对另列。

启动时无可见他人context/0MiB，用户确认GPU空闲；空载NVML25%异常仍未解释，单进程3轮，保留全部慢点，只作限定资源条件下的延迟证据。

覆盖B2 video/action、B14独立七候选CFG2，各三KV×first/last×层0/8/29。kernel eager/Graph-core、完整block eager。不是完整机器人E2E或新MegaTransformer收益。

[全部性能](PERFORMANCE-ONLY.csv) · [逐轮](PERFORMANCE-ONLY-ROUNDS.csv) · [输入](PERFORMANCE-ONLY-INPUTS.csv)

聚合问题：[]

原始：/rjs/cxz/论文探索/投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r3
