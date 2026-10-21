# 当前讨论与授权

用户最新明确“先做1和2”：先Cross KV设备函数/编译资源控制，然后FFN；本轮已完成并保留两项。固定tile/数学/同步，真实三KV与9签名有限门；不先做跨阶段ready。
完整单层下降9.46%–10.87%，仍慢15%–31%。函数组织控制对Cross有效，FFN只再改善约0.6%。静态SASS及插桩支持机制方向，不把动态spill/traffic当已测事实。
下一步先剩余FFN主循环/归约/epilogue和资源成本，再最小DAG补位。每子算子采用合适tile和自然任务数，108只是worker池，不要求凑满。旧失败/负结果/截止保留。
GPU1优先RoboTwin，已启动CURRENT18并自动接ON/CAL；DEV90尚未排队，不宣称迁移完成。原24h累计预算不重置。
仅optimized；保护baseline/原tile/LIBERO生产、state/cycle/rules、四权限false、paused72与retention。BAAI写入限~/cxz，不改驱动/时钟/服务、不杀他人。GitHub仅main小证据。
本轮M39累计1770.668/36000秒；GPU0本轮作业已退出。历史授权与讨论见[前检查点](history/gemm-function-control-20261021/DISCUSSION-SUMMARY.md)。

最新确认：继续FFN剩余原语/资源核对再调度，用户要求解释整层退化原因；RoboTwin持续后台。已通过Maintain合并到两条现有constraint，不把诊断猜测写成长期事实。
