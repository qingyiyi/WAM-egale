---
kind: progress
cycle: wam-research-20261016
status: active
updated: '2026-10-23T19:38:39.747829+08:00'
current: M39
next: 本轮5090GPU作业已闭合；建议下轮先对齐当前Torch投影累加与舍入，再评估FFN，不自动扩展DAG或恢复新GPU作业。
---
# 当前研究入口

5090完整单层已实际迁移，终态CLOSED_DIAGNOSTIC_NOT_QUALIFIED。B14真实短/中/长KV288/1152/2160，两个独立进程，正常二进制双方eager原Torch比值1.2201/1.7155/2.1406×，双方Graph1.1062/1.6024/2.0361×。输出相对L2为0.9387%/0.7519%/0.6759%，最大绝对差2.5/2/2，首次分歧在Self QKV；不能称精度对齐加速。LN1/调制1、有效KV、输入/RNG门通过，仅三fixture，不继承A100证书。初轮compact-input缓存不等价；最终改为原B14输出去重，Torch缓存baseline恢复逐位一致，自有仍失败。原r1/4次构建失败保留。

当前SM120研究副本：L13＋FFN1尾块两槽；170CTA×8warp，shared100096B=97.75KiB、255寄存器、静态local784B、1CTA/SM。没有新tile/warp搜索或DAG。原A100实际采用入口A仍保留；冻结L13与旧A证据不改。

5090阶段诊断：Self QKV约1.40×；Self Attention约3.12/5.49/5.64×（含历史读取/拼接边界）；FFN联合约0.736–0.740×，是主要慢项。Cross Q RMS约0.64–0.68×；三个剩余投影约0.92–0.96×。自有阶段区间含同步，Torch为独立Graph；不可相加/相乘或将差异全归因于算术。插桩扰动约+0.23/+0.27/+1.09%，硬件counter未取得。

所有本轮源码、两DSO、生成头、原始结果已回本机，源文件逐字节核对。19:35:16 owned PID+出生身份全退出，GPU显存0/无可见compute；实例未关机。本轮31.929分钟，GPU worker生命周期62.395秒、CUDA active UNKNOWN；旧累计21237.865秒不重置，新累计23153.579/36000秒，余3.568小时，不自动消费。新任务目录约1.04GiB。四项科学权限及production_admitted均false。

[5090回传](experiments/rtx5090-block-port-20261023/REVIEW_RETURN.md)、[完整27阶段](experiments/rtx5090-block-port-20261023/STAGE-TABLE.md)、[精确数值](experiments/rtx5090-block-port-20261023/MEASUREMENT-SUMMARY.json)、[资源与失败](experiments/rtx5090-block-port-20261023/RESOURCE-AND-STOP-REPORT.md)、[源码清单](experiments/rtx5090-block-port-20261023/SOURCE-MANIFEST.json)。

## 已保留的边界与历史结果

5090隔离环境Torch2.9.0+cu130、nvcc13.0.88、driver580.105.08，原版32GB/170SM。counter需宿主管理员，本轮不提权/不改驱动/不重启。利用率此前100%与空context矛盾，结束0%，原因UNKNOWN；只见自有CUDA进程不等于宿主级独占证明。[环境](maintenance/rtx5090-environment-20261023/SETUP-RESULT.md)。

A100局部warpT0–T4：FFN2局部准备重叠无可信正常完整层净收益，主线不采用；副线LN1/LN3融合L13有限225调用输出/23中间量/有效KV/输入/RNG及合法B2状态通过，实际使用入口A未被替换。L13原Torch D Graph比值1.3029/1.8530/2.9994×，相对A仅改善0.4%–0.7%；中长仍慢强guarded3.4%–3.6%。不是SM120结果，也不是完整机器人收益。旧失败与B控制不纯记录保留。[原回传](experiments/local-warp-pipeline-20261023/REVIEW_RETURN.md)。

两槽补充：FFN1尾块输入槽3→2，shared109.75→97.75KiB；原CPU首次Config失败保留。本轮首次GPU测量发生在5090且数值失败，不可称两槽在A100已通过。[CPU结果](experiments/single-layer-async-gemm-20261020/ffn1-two-stage-20261023/RESULT.md)。

M40 RoboTwin最后核实为10/23 16:33：125完整三臂组/375局，严格同组T102/C98/P101；微波炉16:32为20/100组T4/C1/P2。GPU1既定队列继续；GPU0原借用窗口已闭合，本轮未复查远端，不能当19:38实时状态。等待微波炉100组三臂完成后再查失败原因；原160GPU小时/10天、11/01 22:03截止及失败保留，不自动扩预算。[原计划](experiments/robotwin-pilot-20261020/eval100-20261022/microwave-completion-20261023/PLAN.md)。

M17四套LIBERO6000局历史数据齐；训练/DEV暴露、严格配对与长尾缺口仍在，四项科学权限false，不恢复旧队列。TF结构与权重/质量证书分开；不以target调用减少代替E2E收益。

10/23冻结身份/claim，10/28主表，11/01 18:00中文证据包；不足则收缩claim。当前原模型冻结，baseline/投机侧/原tile不属于本轮改动。protected state/cycle/rules不覆盖，固定v15入口，不全量sync。

下一步：本轮5090GPU作业已闭合；建议下轮先对齐当前Torch投影累加与舍入，再评估FFN，不自动扩展DAG或恢复新GPU作业。

[Cycle计划](cycles/wam-research-20261016/PLAN.md)、[完整进度](cycles/wam-research-20261016/PROGRESS.md)、[执行日志](cycles/wam-research-20261016/EXECUTION.md)、[优化账本](experiments/single-layer-async-gemm-20261020/OPTIMIZATION-LEDGER.md)。
