# WAM 当前进度

更新：2026-10-20T21:59:48+08:00。先按[INDEX](INDEX.md)读取当前Cycle和约束；不展开全部旧证据。

- **算子当前分析已完成**：[context128 profile](experiments/single-layer-async-gemm-20261020/context128-profile/RESULT.md)。24完整block、150原语；输出/中间量/KV有限逐位。GEMM约49–57%，cross-K RMS0.606–0.654ms，长KV self+cross attention约23%；NCU权限拒绝，动态spill/occupancy/HBM/L2未知。源码/二进制未改。
- **速度仍用前轮普通计时**：B14短/中/长对同轮native比1.866–1.868 / 1.836–1.875 / 1.732–1.737；uncached/eager完整单层，非等速、强cache/Graph或E2E。108CTA/8warp，context128×128/K64、action64×128；仍阶段grid.sync，ready未启用。
- **RoboTwin GPU1固定42队列运行**：现场10/20 22:07:29北京：新增6/42完整终态、5成功1完整算法失败（open_microwave），当前`place_object_basket-valid02`；原首六5成功1失败保留。21:37:02启动，次日03:37:02硬截止，6小时含清理、4GiB新增、累计原24GPU小时帽。Target48目标FIT36/CAL12；108完整BF16 reset bank复用，完整任务bank等价仍未知。当前采集非TF评测；训练/DEV待审计接续，本guard不自动训练。
- **算子GPU0已清理**：本轮约23.2/900秒GPU生命周期，证据约25.3MiB；旧1497/1500、372/420、FAILED及旧截止不改。
- **四套LIBERO原结果**：T/C/P各500，Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499。首两S审计FAILED、训练/DEV曝光与长尾保留，非独立TEST。

下一步：先同输入/同边界逐阶段三路径对齐，再改主要差距，最后context K/V ready填尾波。归约、BF16、合法KV门不放宽；普通缓存不是新贡献，强baseline矩阵缺项保留。

四项engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission全部false。旧FAILED/retention/paused72/MLP0/22缺收据/default A_off保持。baseline/投机/原tile/state/cycle/rules不改；原action沿原算子；BAAI只写~/cxz，AutoDL归还，不改驱动/时钟/服务或杀他人。

[Cycle完整进度](cycles/wam-research-20261016/PROGRESS.md) · [算子审计](experiments/single-layer-async-gemm-20261020/context128-profile/FINAL-AUDIT.json) · [Robo状态](experiments/robotwin-pilot-20261020/target-remaining42/LIVE-SNAPSHOT.json)。

10/22 go-no-go、10/23冻结、10/28主表、11/01 18:00证据包。main镜像只发布源码/HW/小证据。

最新[逐阶段对齐计划](experiments/single-layer-async-gemm-20261020/context128-profile/STAGE-ALIGNMENT-PLAN.md)尚未启动新GPU；当前profile预算已终态。
