# WAM 当前状态

更新2026-10-20T23:19:23+08:00；[Cycle](cycles/wam-research-20261016/PROGRESS.md)。

- **简易版已选定并删除复杂活动入口**：[报告](experiments/single-layer-async-gemm-20261020/stage-alignment/RESULT.md)。9真实签名/23中间量/18有效KV转移与RNG等有限逐位通过；两进程交错计时无稳定退化，简易较复杂中位数低0.4%–0.7%。历史失败/源码保留。
- **当前完整单层尚未对齐原版**：简易/native短1.847–1.928、中1.8625–1.8628、长1.7126–1.7142；GPU0无争用，uncached/eager、准备排除，非强cache/Graph或E2E。
- **[逐阶段矩阵](experiments/single-layer-async-gemm-20261020/stage-alignment/STAGE-RESULT.md)**：cross K/V独立GEMM慢9%–10%，FFN1约1.84倍；cross-K RMS约3.4倍、GELU约6倍、table约5倍、layout约21–36倍。C插桩含队列/同步/缓存差异，不能当纯吞吐。NCU动态counter UNKNOWN。
- **下一步**：只改简易版，先pointwise与B14只读layout，再cross-K RMS、FFN1/cross-FA2，最后ready调度。当前仍108CTA/8warp、阶段grid.sync，没有全局异步执行器。
- **算子阶段已清理**：662/720block、4524/6000stage，GPU worker生命周期106.745秒，18owned回收、GPU0空；累计10h帽，旧用量计入。无效slot比较FAILED与其余构建/路径失败保留，原675/240资源FAILED不改。
- **RoboTwin GPU1继续**：截至2026-10-20T23:14:45+08:00：新增19/42完整终态、15成功、4自然失败，当前`stack_blocks_two-valid04`；原首六5成功1失败保留。原42Target采集固定队列自动接续，目标合并Target48 FIT36/CAL12，非TF评测；复用完整reset bank，完整episode等价仍未知。截止10/21 03:37:02、原24h总帽不变；不自动训练。
- **LIBERO原完整6000局**：T/C/P各500，Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。首两SFAILED/训练DEV曝光/长尾保留，非独立TEST。

四项engineering_quality_screen/independent_test_complete/noninferiority_established/formal_performance_permission均false。旧FAILED/retention/paused72/MLP0/22缺收据/default A_off不改。baseline/投机/原tile/保护state-cycle-rules不动，BAAI仅~/cxz，AutoDL归还，不动驱动/时钟/他人进程；只main镜像。

10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。
