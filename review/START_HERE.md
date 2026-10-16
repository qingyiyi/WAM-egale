# 给外部研究评审者的入口

评审对象：冻结LingBot-VA/WAM上的连续动作投机，以及多候选verifier的Transformer调度。目标是CVPR，内部中文稿节点2026-11-01 18:00（北京时间）。

## 先确定你真的读到了快照

先报告仓库提交或`SOURCE_MANIFEST.json`中的snapshot_id，以及实际打开的文件；不能读取GitHub时直说，不得从标题编造已读源码。用户可改用`REVIEW_PACK.md`或下载当前commit的ZIP。

事实优先级：最新原RESULT/事前协议→对应独立审查→最新研究报告/短状态→旧报告与主题记忆。较新记录不能追溯更改旧FAILED；旧时间点的“正在运行/扩大候选”不代表当前状态。

## 当前事实

- 本机50完整horizon800已经结束、资源清理完成。fresh三臂各10/10，TF长度三项通过；风险task4/init26出现Target成功而TF失败，原retention=false。
- 28个质量配对保存物理/controller/实际初始Torch RNG一致，初始相机25/28。否决task4这一对本身双相机也exact。未保存模拟器内部与后续完整随机状态UNKNOWN。
- 当前无GPU作业、无CPU waiter；AutoDL已归还，不再连接，实例关机未知。明天120局入口SOURCE_READY但实际前置拒绝，未注册、未运行。
- TF结构可暂冻用于算子研究，权重/训练配方未accepted。MLP只在本轮fresh有公平对照，未在本轮8个风险组取得匹配结果。没有最终架构优越性结论。
- 全局旧A_off不变；engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission均false，新TF正式45chunk受阻。

## 三条不能混淆的证据

1. 本轮新TF质量、晨间80、旧child50是不同权重/范围；不得混池替代独立TEST。
2. 同7个冻结query的逐个验证/B14批量对照证明批处理经济性，不是原始顺序机器人闭环加速。
3. 新attention原型在强B14上的有限改善，不等于cached完整chunk、TF闭环或E2E获益；普通Graph/shared-prefix/共同缓存不是新增贡献。

## 待解决的最小缺口

当前task4已有20个初始数组及首保存观测相同，首环境command索引0即有差；失败与另一个成功初态的接受深度/σ计数相近。缺已有teacher root/候选velocity到提交动作、环境命令的无歧义映射，不能据此决定补训练、累计预算或PF。先前瞻登记2–4对完整诊断；详细日志应有明确20chunk/256query等上限，整个episode仍完整完成。

## 优先阅读

- [当前数字](RESULTS.json)
- [最新50完整报告](../.pipeline/research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)
- [原统计协议](../.pipeline/research/speculative-sampling/STATISTICAL-QUALITY-PROTOCOL-20261015.md)
- [系统与历史负结果](../.pipeline/research/speculative-sampling/OVERNIGHT-RESULT-20261016.md)
- [本轮原RESULT](../投机采样/results/runtime/p0-draft-freeze-quality-v1-local-20261016-r1/RESULT.json)
- [本轮独立raw审查](../投机采样/results/runtime/p0-draft-freeze-full50-independent-raw-audit-v1-20261016-r1/SHORTREPORT.md)
- [实际task4诊断](../投机采样/results/runtime/p0-draft-freeze-q02-t4i26-diagnostic-v1-20261016-r1/REPORT.md)

其后按问题读取README中的源码，不要求一次读取全部历史文件。未发布的原始大张量仍在服务器，不能声称已在GitHub上重算。
