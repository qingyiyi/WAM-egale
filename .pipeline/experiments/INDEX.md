# 当前研究实验

- [M34整层单kernel](single-layer-warp-specialized-20261019/RESULT.md)：9签名各1计算kernel，原数值FAIL，耗时为native的5.12–11.28倍；[唯一布局诊断](single-layer-warp-specialized-20261019/DIAGNOSTIC-RESULT.md)后600调用收口，未准入。
- [M35四套工程计时](libero-four-suite-performance-20261019/ISOLATED-RESULT.md)：360原上限，Object首轮CPU白名单失败、0局不重试，最多330；新重复不作独立质量N。[6000完整质量与原门](libero-four-suite-performance-20261019/QUALITY-RESULT.md)。
- [M36 RoboTwin只读调研](robotwin-migration-survey-20261019/RESULT.md)：官方权重/50任务/seed与迁移接口；0新环境实验。

- [M33单层退化归因](single-block-persistent-profile-20261019/RESULT.md)：profile与216 CTA反事实完成、无新增正收益；NCU计数UNKNOWN。该版本没有整层单kernel，后续M34另列。

- [M32单层persistent](persistent-tile-remote-20261019/RESULT.md)：有限精度/状态通过，两进程新增性能收益未成立；本轮30层取消。

- [M31隔离optimized与tile子图](transformer-tile-subgraph-20261019/PLAN.md)：明确授权，baseline/共享依赖冻结；实现与必要验证，不含persistent队列。

仅按当前任务读取；旧 runtime 与失败记录仍是原始证据。以下均为历史入口，描述原登记阶段，不表示当前仍在运行或可以自动重启。

- [算子短中长真实输入矩阵](operator-kv-matrix-20261018/PLAN.md)：用户新准本机补B2/B14性能精度，当前CPU等待独占，无GPU已测新结果。
- [原算子完整既有证据](operator-parity-evidence-20261018/REPORT.md)：576检查/60性能单元/420逐轮完整CPU复核，旧有限通过不外推整模型。

- [四套研究与三套同配方TF](libero-four-suite-20261017/PLAN.md)：M17已批准，控制器已登记等待资产；Long原job继续，旧四false不动。
- [Long全量原job](libero-fullscale-20261017/ACTUAL-RUN-STATUS.md)：M16双卡四worker运行，旧69局只读接续、48h预算不重置。

- [TF效果收口1→2→3](tf-effect-closure-20261016/EXPERIMENT.md)：已有限收尾77/96，TF风险NO_GO、旧失败和审计缺口保留。原四权限false、120/TEST/正式性能不启动。

- [task4 方向到环境命令诊断](task4-direction-20261016/EXPERIMENT.md)：固定 b0dfac3；原定2对、完成1对，PARTIAL保留；不放行质量或120局。
- [draft真实teacher路径](draft-teacher-path-20261016/EXPERIMENT.md)：两记录chunk/首query与时间反事实完成并清理；TF64固定主模型，接下来只优先TF效果与训练收口，四权限false。
- [TF递归监督与训练覆盖](tf-training-support-20261016/EXPERIMENT.md)：2712真实query CPU审计完成；只推进TF，不将单失败病例的描述性分布差异写成已证实根因。

- [M37设备数值契约与成本归因](device-contract-20261020/PLAN.md)：最新继续授权；旧M34/M35已关闭。
