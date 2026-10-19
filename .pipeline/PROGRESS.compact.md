# 当前短入口

更新：2026-10-19T21:08:07.097619+08:00。当前Cycle：[wam-research-20261016](cycles/wam-research-20261016/PROGRESS.md)。先约束记忆，再读当前四文；旧大索引/原trace按需。

## 当前：M32只实现一层完整Transformer block

[实现/有限验证](experiments/persistent-tile-remote-20261019/RESULT.md)、[两进程九点性能](experiments/persistent-tile-remote-20261019/PERFORMANCE.md)、[源码与使用](../算子研究/wam_ops/optimized/SUBGRAPHS.md)。用户明确不要30层大算子；本轮full30为0，取消迁移/验证，不作为交付门。最终模型E2E后续本机。

远端独立optimized已编译并执行阶段内部persistent CTA：attention/FFN1按固定步长多tile循环；self/cross/FFN与norm/RoPE/调制/残差完整保留。QKV/Wo/FFN2仍cuBLASLt，无跨阶段ready或warp专职分工。baseline及依赖88项不变；未加载baseline/旧DSO。新源码本机DSO仍待重编，新ABI会拒绝旧产物。

9真实block、90连续有效KV、48原语逐tile覆盖/输出/RNG/参数/API门逐位通过；tile/persistent的24中间量（action融合23）与独立顺序实现对照。两进程378计时原始样本：persistent+FFN1对静态+FFN1，B14慢4.42%–6.26%，B2 action慢2.56%–2.80%。没有新增加速；A/A最大10.07%保留，GPU1/host共享及三样本限制明确。不是最强B14/Graph、完整chunk或E2E。

五GPU作业77.927秒、731block/48原语/0episode，owned全部退出。首计时0block因宿主/容器PID误判失败，修正守卫后新attempt2/3完成；传输超时/错误全保留。远端约636.5MiB隔离材料；时间主要花在传输，CUDA active为UNKNOWN。M32预算收口不挪用。

## 历史与机制边界

[M31](experiments/transformer-tile-subgraph-20261019/RESULT.md)独立静态路径9block/72状态及独立进程54状态；[M28](experiments/block-native-baseline-20261019/RESULT.md)六actual30及两进程动态block是旧版本结果，不自动套给新M32。M21/M25严格2.5%/NVML、M26 CUTLASS FFN2/NCU、M27资源与旧误判均保留。

[M30真实DAG](experiments/transformer-tile-handoff-20261019/ATTENTION-SUBGRAPH-DESIGN.md)：D3072/H24/d128/FFN14336，action Q16/Linear M32或224，每attention实例仍Q16；action self-K288/1152/2160、video本批272/1136/2144、cross512。候选private KV不共享。MegaMoE只借思想，SM100原语不照搬SM80。

[M29文献与主张门](experiments/speculative-denoising-novelty-20261019/RESULT.md)：ICML2025投机扩散、ICLR2026 FlowCast、TS-DP/FLASH、Speculative Forcing/WAMJET有直接近邻。复现/普通batch/缓存/现有融合不能单列新贡献。

## 远端质量与资源

Long T483/C484/P473、Goal463/464/467、Spatial472/466/464，各臂500完整；[成功率表](experiments/libero-four-suite-morning-20261019/COMPLETED-SUITES-SUCCESS-20261019.md)。Object 1500/1500，完成数T500/C500/P500；来源2026-10-19T13:06:38Z。全部500共同case齐；成功数T497/C498/P499（99.4%/99.6%/99.8%）。原controller已退出，既定评估计算完成；审计和资源终态仍按原证据，不把完成覆盖写成质量放行。 GPU0本轮算子已退出。

原CPU约5分钟巡检继续；动态数看原watch的LATEST，SSH失败不冒充实时。首两S严格配对FAILED、训练/DEV曝光、组件审计与长尾仍有效；描述性覆盖不是独立TEST或非劣。

TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost及BF16/FP32冻结。PF2=SST完整policy，T50更新/51action调用含末尾；reset/cache不计周期，不是bonus或状态恢复。所有臂同用512-token BF16 reset bank，cold另列；actual796/罚分800分列。

## 下一步

M33仅单层内部收益定位：先profile常驻循环的寄存器/驻留/尾部成本，再选一个可复用公共历史或中间量的tile分组/相邻子图；静态路径作当前对照。没有启动M33 GPU，也没有30层融合或新机器人评估。主要系统主张最终仍需强B14及sum(T_round)/sum(g)，不能由调用下降替代。

CUDA13.x：本机driver595.71.05，远端535.161.08；远端不满足普通CUDA13最低580条件。未升级驱动或系统，本轮独立CUDA12.4/cu126，未来另核验。

## 保护

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

M17原10/21 16:56:31停新、17:11:31清理，18723000尝试/15GiB、每卡最多2worker；matched Target前200组同GPU1，不为填卡重复Target或破坏配对。BAAI仅~/cxz；AutoDL不再SSH、关机UNKNOWN。本机计时自然空闲、精度可共享；不杀他人/改服务/时钟。

投机源码/配置/运行依赖、原tile与state/rules保护，不建缺失cycle.yaml、不升级/旧sync覆盖HW。仅baseline/optimized两份主实现；baseline及依赖冻结，优化只在optimized。无自动新训练/120/独立TEST/新TF正式45chunk/第二环境/大融合。中文正文会话交飞书。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包。

约束入口：[memory/INDEX.md](memory/INDEX.md)。旧方案以Cycle归档追溯，不重复全量展开。
