# 当前讨论（2026-10-23T02:12:38.420042+08:00）

用户同意其余三个投影复用8warp，并要求进一步优化Self QKV、最终同步完整阶段表。六投影8warp正向；固定库默认K128不可直接实例化，0GPU；合法较大M128通过精度但慢9%–11%。恢复M64并联合QKV独立tile队列，288任务、三份权重/输出保留，grid27→23，完整层增量仅0.08%–0.44%，不包装成显著系统突破。最终四臂同轮对上版延迟降约1.8%–2.2%，原Torch完整单层1.2967/1.8469/2.9910×。

新全阶段重测，QKV联合行约0.86×，不能拆成三行虚构计时或用旧热缓存比值推断提升。剩余投影/小阶段仍未全部追平。下一步建议核查原K1536分段部分和能否保持BF16边界并留在CTA私有workspace，尚未实现；当前算子已清理，禁止继续占GPU0扩候选。

用户明确临时双卡RoboTwin到上午：GPU0借用至10/23约10点，按完整任务预计时长停派、自然终态，GPU1继续。已Maintain替代本时段固定保留GPU0。原160GPU小时/10天、配对同卡、固定1800局及失败保留不变。原27局在完整pair后暂停，新共享队列从未运行组接续。两卡实际运行，合计37/1800；GPU0首完整pair3/3成功、初态/首两chunk审计通过；完整reset bank直接引用，无重复大bank。

[新版完整表](../../experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/STAGE-TABLE.md)、[资源](../../experiments/single-layer-async-gemm-20261020/small-projection-followup-20261023/RESOURCE-AND-STOP-REPORT.json)、[临时双卡](../../experiments/robotwin-pilot-20261020/eval100-20261022/dual-until10-20261023/PLAN.md)。四科学权限及production_admitted=false，旧FAILED、保护文件、baseline/投机生产/原tile不动。
