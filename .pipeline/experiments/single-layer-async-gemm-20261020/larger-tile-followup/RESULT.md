# 较大 tile 与协作 epilogue：有限单层结果

2026-10-20 北京。终态：COMPLETE。选择 `context128`，本轮 B14 三个 KV 的两次独立进程中位耗时比均≤2；这是本轮工程筛选通过，production_admitted=false，四项科学/正式性能权限仍false。

## 实际改动

仅 `算子研究/wam_ops/optimized`：

- CUTLASS 设备 MMA 后由全部8个warp协作执行成对FP32读取、原顺序bias/partition加法和BF16写回，替代单warp串行epilogue。计算仍由warp3–6完成；未声称专职加载/计算已全面重叠。
- pointwise与cache copy任务从128增到2048元素，pointwise直接由计算warp写出；保留每元素算术/舍入边界。
- 两个预登记形状：`cooperative64`全大M输出tile64×128；`context128`仅M≥1024改为128×128，K64/3-stage不变，action M224仍64×128。默认构建选128；显式`WAM_CONTEXT_TILE_M=64`复现对照。
- CUTLASS helper内联；实测寄存器仍255/thread。没有证据表明spill已消除。

B14的context M=14×512=7168时，单个K/V投影任务数2688→1344；action M224的3072列投影维持96个任务。pointwise任务理论数量减至原1/16。这些是源码任务数，不能替代GPU流量或周期计数。

## 精度与状态

每种实现各72次完整block门检。B2 action、B14 action、B2 video ×短中长KV，共9真实输入签名；每格输出和23中间量均逐位匹配原Torch/native，18个合法KV更新转移通过；重复输出与有效KV、输入不变、CPU/CUDA RNG、tile exactly-once通过。B14是14个Q16实例，不改变candidate私有KV可见性。

门与计时读取同一原始fixture路径；计时warm输出再次逐位比较。未保存新的全模型/PT/KV副本。只能称这批有限输入通过，不能证明所有输入等价或继承机器人质量。

原始：[64门](cooperative64/CUTLASS-GATE-RESULT.json)、[128门](context128/CUTLASS-GATE-RESULT.json)、[源清单](SOURCE-MANIFEST.json)、[终态源核对](FINAL-AUDIT.json)。门后仅把构建默认值设为已测显式128并修正两处注释；CUDA数学源码不变。

## 完整单层速度

远端A100-SXM4-40G，GPU0启动前空卡，每格核验只有本worker计算context；两独立进程、各8组交错native/new普通样本。CUDA event计时，uncached/eager完整层，准备阶段排除；profile单列不混入样本。12条profile各1个compute kernel。没有完整verifier/chunk/E2E或最强cached/Graph结果。

所选128版本逐进程中位（ms）：

|KV|Torch/native进程1 / 2|新整层进程1 / 2|同进程耗时比1 / 2|
|---|---:|---:|---:|
|短288|2.6312 / 2.6250|4.9142 / 4.8978|1.8677 / 1.8658|
|中1152|2.8145 / 2.4940|5.1681 / 4.6768|1.8363 / 1.8752|
|长2160|3.0284 / 2.9325|5.2467 / 5.0951|1.7325 / 1.7375|

同轮64控制的耗时比为短1.9953–2.0004、中1.9357–1.9394、长1.8160–1.8166。短KV某次略超2，因此没有把64控制写成严格筛选PASS。128在三格的同进程归一化比均更低，选择128。

中KV绝对延迟存在明显进程间变化，所以主表逐进程报告，不用混合样本的两个独立中位相除掩盖差异。两种variant不是同时/交错部署；只支持当前有限配置选择，不宣称普遍大tile优势。旧7.94/8.14/8.64ms不是同轮旧/新AB，不据此签发严格百分比增益。

原始：[汇总](SUMMARY.json)、[128进程1](context128/cutlass-timing-1-RESULT.json)、[128进程2](context128/cutlass-timing-2-RESULT.json)。

## 尚未解决

128资源：108CTA、每CTA256线程、255register/thread、98848B shared、1608B localSizeBytes；64是82464B/1536B。更大的tile没有降低整体资源占用，实际occupancy、stall、spill流量、HBM/L2变化UNKNOWN。

默认仍有阶段grid.sync；此前FFN ready路径本轮未测，保持关闭。cross-K/V尚未提前与其他子图交叠。本轮不是完整异步DAG，也不是新算法系统协同证明。

下一步以128为单层研究基线：先测当前阶段和spill/调度开销，再做独立context K/V及其norm的ready任务填尾波；消费者必须等相应数据就绪。保留FA2和GEMM数值边界，重新通过状态门后再比较同一强baseline。不要直接扩大到30层。

## RoboTwin

[现场终态](ROBOTWIN-LIVE-CHECK.json)：六个FIT身份Target完整任务5成功、open_microwave失败；83policy/6557teacher。失败实际1520命令与罚分1500分列。无owned残留、无后继自动队列；TF collector/训练/DEV未完成。当前只关注状态，未按过期首六局deadline重启。完整任务reset bank等价仍UNKNOWN，不能把六FIT例外推为标准成功率。
