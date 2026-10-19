---
kind: discussion-summary
cycle: wam-research-20261016
updated: '2026-10-19T21:08:07.097619+08:00'
---
# 当前讨论与授权

用户最新要求只实现一层完整Transformer block，在这一层内做tile和persistent；取消本轮30层迁移/验证，不以完整30层作为完成门。实现与有限验证放BAAI ~/cxz，未来最终E2E在本机。baseline连同共享依赖冻结，优化只在独立optimized。

M32已实现阶段内部常驻CTA循环并通过有限精度/状态门；探索计时反而慢，B14相对同开FFN1静态路径慢4.42%–6.26%。没有整层单kernel、跨阶段ready/warp专职分工或MegaTransformer加速结论。后续坚持真实self/cross/FFN依赖，只讨论有复用价值的单层tile调度；M33新GPU作业未启动。

用户问CUDA13.x是否可用，不等于批准驱动升级。本机595.71.05，远端535.161.08；常规CUDA13要求driver≥580，本轮保持独立CUDA12.4工具链/cu126 wheel。

Object 1500/1500，完成数T500/C500/P500；来源2026-10-19T13:06:38Z。全部500共同case齐；成功数T497/C498/P499（99.4%/99.6%/99.8%）。原controller已退出，既定评估计算完成；审计和资源终态仍按原证据，不把完成覆盖写成质量放行。 GPU0本轮算子已退出。

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。

只main公开核对源码/HW/小结果，无权重/fixture/binary/凭据。历史授权见[归档](history/M32-before-summary-20261019/DISCUSSION-SUMMARY.md)。
