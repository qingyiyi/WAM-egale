# 当前讨论与授权

用户同意先对齐再调度，先固定真实输入、tile与计算体区分主循环/共享搬运/epilogue，不再无依据扩CTA/warp搜索。当前两候选完成：padding136去掉共享写bank冲突，四元素搬运减少循环；局部与4warp整层均有收益，有限9签名严格门通过后保留。

这只是原语对齐，不是MegaTransformer调度创新。单层单kernel目标保留，尚未实现ready-DAG；先说明FFN/小投影与整层资源剩余开销，再按真实依赖提前tile。独立投影仍慢约4%，不能称全面对齐。两次实现/测量问题与修复保留，不能以PASS覆盖无效对照或旧FAILED。

直接修改optimized、基线及投机生产不动；本轮不新建Cycle。沿10小时累计，所有历史截止/清理收据不重写；两个计算体候选已用完，已清理GPU0。RoboTwin GPU1固定DEV90按原队列自然终态，24小时累计与10/22 01:26硬截止不变，无自动重试。

四科学权限false；LIBERO与当前DEV不是独立TEST。BAAI全部文件在~/cxz。

[前讨论](history/projection-gemm-core-before-20261021/DISCUSSION-SUMMARY.md)。
