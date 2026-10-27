---
kind: plan
cycle: wam-research-20261016
status: active
updated: '2026-10-27T10:11:52.929047+08:00'
---

# WAM当前计划

当前主验证平台为固定Base FastWAM。用户已授权三张A100继续投机实验，小筛查无问题后自动进入全量target/TF/TF+PF；不再等待LingBot十步或VA2完整资产。四科学权限仍false。

|工作|当前安排与入口|
|---|---|
|投机小闭环|[夜间清单](../../experiments/fastwam-base-small-closedloop-20261026/OVERNIGHT-PLAN.md)：100FIT/10CAL和20组三臂完成；T17/TF19/PF18，新tau=.01541235，原登记扩量门通过|
|三卡并行|[PLACEMENT-v3](../../experiments/fastwam-base-small-closedloop-20261026/PLACEMENT-v3.json)：本机30FIT/3CAL+4组三臂；远端70FIT/7CAL+16组三臂；非测速尽量每卡两个隔离环境客户端共享一模型|
|条件全量|[四suite共6000局全部审计完成](../../experiments/fastwam-full-triarm-20261027/FOUR-SUITES-RESULT-v1.md)：每suite各500组三臂，同配方各100FIT/10CAL/20DEV先行；合计T1950/TF1945/PF1944（每臂2000）。完整50初态含训练/选择暴露，不称独立TEST。|
|真实速度|[共同prompt缓存四套已完成](../../experiments/fastwam-full-triarm-20261027/CURRENT-RESULT-v5.md)：Long/Spatial/Object/Goal action1.242/1.205/1.229/1.150×，chunk1.080/1.068/1.071/1.057×；源组N16/19/20/20。原未缓存计时保留。跨进程路径差异未定因，缓存新版本不冒充原版全量质量。|
|RoboTwin|[首四canonical审阅通过](../../experiments/fastwam-robotwin-canonical-20261027/CANONICAL-REVIEW-v1.md)：4/4完整成功，14维/24执行/十步协议与图像检查通过；含失败/诊断累计502.120秒。下一步登记自身FIT/CAL/DEV并核验真实14维采集和B7，尚未训练或扩量。|
|算子另会话|[chunk内Cross KV缓存完成](../../experiments/fastwam-crosskv-cache-20261027/REVIEW_RETURN.md)：算子chunk内CrossKV缓存已完成：6组层、4观测×T/S完整30层/10步动作/路径及条件RNG逐位一致；B1/B7分别建cache，不跨chunk复用。warm层相对同缓存eager为1.407/.925×，相对同缓存compile为.605/.459×；完整policy缓存使compile Target/投机约1.061/1.015×，自有仍慢同缓存compile约19–20%。compile新图动作relL2约.25–.37%，不能继承eager证据。显式研究入口保留、生产未改；5090已释放。 下一步优先B7视觉KV布局/复制与输出投影、FFN2并行度；前轮FP32不采用。|
|共同文本缓存|三臂共同优化已完成，225输入同进程条件/动作/路径exact；主计时1125对、每臂75次同步profile。文本约0.06ms。VAE在GPU且未改；[分布和成本](../../experiments/fastwam-full-triarm-20261027/TEXT-CACHE-v1/REVIEW-4suites-v1/REPORT.md)。|
|论文证据|只维护[主张证据](../../experiments/fastwam-base-small-closedloop-20261026/CLAIM-EVIDENCE.md)，不生成正文；冻结target+小模块、batch/cache/compile本身不称创新|
|后续跨模型|FastWAM之后，有时间和资源时补StarVLA-PI、StarVLA-GR00T；研究范围为flow-based robotic action models，尚未启动。[决定](../../memory/global/requirements/decision-研究范围扩展至基于流的动作模型并候补StarVLA跨模型验证.md)|

原小规模截止10/27 21:32保留；[夜间扩展](../../experiments/fastwam-base-small-closedloop-20261026/OVERNIGHT-EXTENSION-v1.json)另登记36h wall/72 aggregate A100h上限并含已有消耗，5090投机2GPUh。旧实验全部暂停，不因剩余额度重启。保护state/cycle/rules和原tile不动。

[整理前完整计划](history/before-fastwam-overnight-20261027-010743/PLAN.md)保留所有历史M39/M40/M41/M42节点与证据入口；旧M01–M04完成不代表当前研究完成。
