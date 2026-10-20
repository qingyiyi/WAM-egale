# 当前Execution checkpoint

## 2026-10-20：单卡RoboTwin迁移与算子下一步规划

完成只读资源/源码/官方资料核查及[唯一详细计划](ROBOTWIN-AND-OPERATOR-PLAN-20261020.md)。12:03北京两卡空闲；现有va环境无SAPIEN/mplib发行包，系统Vulkan兼容性未知。计划区分84采集、90三臂DEV及算子数值→原语→就绪流水；预算与时间为建议。已保存用户规划范围记忆，未执行GPU/安装/下载/训练，不重置旧账本或更改四false。

## 2026-10-20T11:30:55.620554+08:00：自有整层内部FA2交付与误差定位

[内部FA2结果](../../experiments/single-layer-async-gemm-20261020/FA2-RESULT.md)：直接修改optimized，在整层单kernel内接入固定FA2 self/cross、split-KV/combine。9真实签名、18同输入attention全部逐位PASS；9重复/输入/RNG/参数、18相对旧async的有效KV转移和18单kernel trace通过。完整hidden仍9/9逐位FAIL，相对L2 0.133%–0.563%，最大0.5–1.5；同输入Q/FFN2独立误差和video短/中Norm2的12/8元素差异已定位，长video首次差异仅FFN2。

两个进程216样本：相对旧async，短action慢8%–9%，长action耗时降18%–21%，长video降43%；仍比native慢3.20–7.60倍。108 CTA、255 registers、82464B shared、2168B local资源，非实测spill/occupancy。387block＋18SDPA＋18Linear；GPU worker生命周期69.979秒，阶段到清理836.9秒，约6.79MiB。两次CPU编译失败保留，7组父子14个PID/birth退出，两卡空；本地75/远端71保护项不变。无新机器人/训练/30层，四false保持。

旧[M39双缓冲](../../experiments/single-layer-async-gemm-20261020/RESULT.md)675block、264.838秒、432样本及240/900登记不一致FAILED保留；旧预算未重置。本阶段沿既有记录承接用户FA2授权，没有新建实验登记。下一优先FFN2/Q及小量Norm2数值，再处理短Q资源压力与跨阶段就绪调度。

源码直接改optimized；原baseline/依赖、投机源、原tile及state/cycle/rules保护。阶段前检查点见[history](history/M39-before-fa2-20261020/EXECUTION.md)，原双缓冲预算失败并未被本阶段成功清理覆盖。

## 2026-10-20T13:42:42+08:00：O1终态与M40实际阻塞

数值门及资源见[O1结果](../../experiments/single-layer-async-gemm-20261020/NUMERICAL-ALIGNMENT-RESULT.md)，后继O2总帽400block/2h。RoboTwin完整policy仍0局，OIDN真实场景失败/清理写入原GPU账本，私有环境兼容修复中；不得把capability查询当场景通过。

## 2026-10-20T13:56:55+08:00：有界负结果与后继

O2实测收口NO-GO，完整输出与状态门保持有限通过；O3只先一条FFN依赖交接，预算内验证。M40两次原始渲染非法访存，停止后继；无需等待空GPU再自动启动。新增报告和原件在对应namespace，既定四false/失败/保护项不变。
