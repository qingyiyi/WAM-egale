# Long＋三个小套件：同配方TF与三臂完整评估

当前Cycle：wam-research-20261016，M17。用户已明确批准[三个套件的独立训练与评估](../../memory/global/requirements/decision-四套LIBERO同配方TF与三臂评估.md)。这是新增执行范围，原Long M16和4f04 paused72/旧FAILED/四false保持，原tile与保护文件不动。

## 可见执行清单

- [x] 公开发布身份核查：Wjjjh三个固定revision，transformer各10,177,831,668B，配套norm_stats；非官方复现版本。
- [x] 实际资源：10/17 16:01 AutoDL两卡各98%、四Long worker继续；数据盘空17.97GB、OS盘空32.15GB；本机A100空闲，本项目盘空约104GiB。
- [ ] 有界下载/原始配置与归一化归档，runtime使用独立模型目录及共享base组件，不改旧registry。
- [ ] 复用原数学构建三套完整512-token BF16 prompt bank。
- [ ] 独立通用server/client/collector/训练/监督入口完成并冻结来源；不执行smoke或独立测试。
- [ ] 利用本机空GPU先进行完整Target采集和质量；AutoDL原Long自然完成后接续可用资源，不截断现有episode。
- [ ] 三套各一次固定训练链、最终checkpoint冻结，完整T/C/P各500局；全成败、长度、800罚分与探索性成本汇总。
- [ ] 质量满足登记条件后申请/使用无争用窗口，测完整chunk和完整任务E2E；不以并行墙钟或调用数伪装正式加速。

## 固定方法与新数据

三套统一TF64（53070参数）、seed20260923，原raw-velocity双头loss、AdamW、task→episode→query均衡采样、初始2000updates/lr1e-4/batch512/wd.01；后续各500updates/lr3e-5/batch512，target与对应观察FIT各256。normalization仅由各套件Target FIT计算并冻结，CAL不选seed/最佳checkpoint。接受器固定原sigma-weighted L2、tau .007839680526450805、prefix_then_cost、K2/B2 bonus/O1/O2/O3、BF16方向/FP32打分，PF2按policy chunk SST。

Target初始200完整局：init0–17 FIT180、18–19 CAL20，各任务均衡；不仿旧TRAIN194因缺记录丢掉6局。父TF2000采30完整局：init20/21 FIT20、22 CAL10；target/该FIT50:50训CHILD2500，再同数据500更新训CONT3000作采集器。CONT采新30：init23/24 FIT20、25 CAL10；ON从同CHILD2500起点用target/新FIT50:50再500，最终ON3000。CONT只是固定数据生成中间模型，不新增结构或seed搜索。仅保存并标注实际reached监督，反事实节点另列，不额外teacher调用。每个阶段仅最终checkpoint。

每套Target500局中的前200可作为训练源且只读复用其完整质量记录，计时受采集影响的部分明确分层；C/P为冻结ON后的新完整500局。全部官方50init含训练/校准暴露，故主表是完整描述覆盖而非独立TEST。最终C/P同case同物理GPU与Target匹配，不能跨硬件混作严格配对或正式计时。

## 资源与边界

新增三套各1500评估＋60训练采集＝1560完整episode，合计4680；每局原horizon800、1200秒、最多4000 target尝试，无自动重试。每套总6,241,000尝试（含非episode1000），三套18,723,000。每个实际phase开局前预留整局及清理余量，阶段/系列账本均不重置。

新执行窗口建议上限96小时，在实际job登记时落绝对起止，清理预留15分钟；10/22质量go/no-go之前收口，超限PARTIAL。该上限不是精确预计耗时。原Long仍遵守自己的10/19 13:55:56清理deadline，不借新授权续旧job。

源资产本地新增上限40GiB（3个target约28.44GiB＋下载暂存/metadata），研究原始新增上限15GiB（每套compact teacher≤1GiB，代表fixture总≤3GiB，不逐query存历史）。AutoDL利用现有OS盘先缓存至多两套新transformer，数据盘保留≥5GiB；第三套按空间和已结束缓存依赖核验顺序切换。归档可回传本项目，旧失败与不可重建tensor不删除，不自动扩容。

本机先最多2独立模型worker/同A100，server/client各2CPU，总8；bank CPU≤7、训练各CPU2且错开同资源密集阶段。AutoDL现Long4worker/16CPU不加GPUworker；自然收尾后新phase同样最多每GPU2独立worker，总CPU不越20quota。模型加载峰值/其他用户负载以实际准入为准；不杀他人进程。

训练及新套件首批完整任务后给出基于实测的工期；当前仅粗估新增总体约48–72小时，受新任务长度与失败尾部影响，96小时为止损帽。正式性能需质量与独占条件，不能保证按该时间已放行。

每phase登记goal/source commit/input manifest/CPU-GPU/绝对deadline/forward/storage cap/终态/清理身份/证据索引。代码与原数学审查并行，资产/bank只做真实产物生产，无smoke/独立试跑/哈希。证据不足保持UNKNOWN及四false。
