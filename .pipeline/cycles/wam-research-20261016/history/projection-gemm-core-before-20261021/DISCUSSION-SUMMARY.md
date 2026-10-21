# 当前讨论与授权

用户明确同意同计算体三调度对照，已完成普通672CTA、108静态循环、108原子队列。直接在optimized添加诊断入口，原4warp整层保持；不新增工作Cycle，不修改baseline/投机生产/保护文件。沿原M39十小时累计，旧失败/预算与截止不重置。

结果不支持把原版的优势归于CTA总数或原子领任务。三调度同前序中位数差≤0.17%，自有计算体仍慢约11%。现在应继续原语主循环/epilogue和整层资源诊断，再真实ready tile。当前仍未实现异步DAG，不以这个负结果否定用户的单层单kernel目标。

增加MMA warp的两个旧候选完整层退化保留；当前4warp继续。独立local0与整层local1456是编译资源事实，不能据此称local是全部根因；插桩成本已单列。两计时脚本失败明确保留，不伪造全部一次通过。

RoboTwin GPU1固定DEV90自动接续，17:02完整45/90；原24h累计/10月22日01:26硬截止不变，不自动重试失败。BAAI所有工作限~/cxz。四科学权限false；LIBERO和当前DEV都不是独立TEST。

[前讨论](history/projection-dispatch-before-20261021/DISCUSSION-SUMMARY.md)。

2026-10-21 17:20：用户再次同意继续：固定本次输入和tile，核对主循环/共享搬运/epilogue，依据profile再改；沿既有预算和optimized，不预判根因。
