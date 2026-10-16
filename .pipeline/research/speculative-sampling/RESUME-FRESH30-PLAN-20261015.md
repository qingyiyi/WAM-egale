# 恢复后的执行清单

更新时间：2026-10-15T22:04:00+08:00，北京时间。用户明确恢复本机GPU，原tile/融合撤回与SHA/smoke限制继续。使用Hypo-Workflow知识记录规范保存证据，受保护state/cycle/rules不改。

- [x] 原40完整采集/清理和配对FAILED确认；不复用旧T10。
- [x] 两阶段TF/MLP六组CPU训练与selected各20237真实CAL CPU逐位一致、退出通过；TF seed23 / MLP seed22。
- [x] 新30独立协议/真实依赖来源/初始相机原像素保存与逐帧覆盖审核。首次reset记录层修复已独立复核。
- [x] 新T10、两阶段MLP10、TF10同批完整LIBERO任务全部30完成，两draft各20237真实CAL GPU接口逐位通过；90分钟预算内结束，非正式速度。
- [x] 原相机严格门FAILED保留，独立终态诊断证明物理/环境与target RNG全配对；仅task8腕相机不同。T/MLP/TF为10/9/9成功，两个draft都task7失败，未过工程成功数门。全部自有PID清理。
- [x] 同seed TF23真实parent2000完整10局消融终态/清理，对照child2500及同批T；parent和child各9成功、失败task1/task7，父子全10初始像素/物理/RNG逐位。800罚分333.2/311.2，研究规则选child，但质量门仍false。
- [x] 独立research quality远端版本实际CPU准备/最终字节及直接CLI增量审阅：70.3MB最小包、原metadata不改、真实20237 CAL逐列拷贝。dispatcher-r2已22:01启动只读WAIT，旧100为78/100，尚无新远端GPU。
- [ ] 通过后且本机独占时，正式45完整chunk真实输入计时；未通过则保留准备。
- [x] AutoDL已恢复SSH，只读确认原100继续；新的纯归档watcher r2已启动。旧watcher认证失败不等于远端实验失败。
- [ ] 原100先终态归档汇总，新GPU任务须依赖实际新质量门与剩余租期。
- [x] 用户追加：已盘点runtime30和项目临时17个失败作业，删4份无用大输出615922780字节（587.39MiB），保留少量失败原因及当前引用数据，tile不动；见[清理报告](FAILED-ARTIFACT-CLEANUP-20261015.md)。
- [x] 汇总报告、HW知识/PROGRESS、CVPR倒排与下一步；本机本轮收尾，旧/新远端终态仍单独待做。知识ledger228条。

GPU质量可共享，正式速度需无外来compute/graphics。原40仅task8 T腕相机初始obs hash不同，物理初态和其余RNG/obs字段相同；无初始像素不能猜差异大小。原FAILED、旧20草稿、旧r2等待证据全部保留。新30须实际保存相机原像素，禁止结果导向重试或放宽门。

恢复时AutoDL archive-only watcher因SSH控制会话失效退出（认证错误），不等于远端GPU任务失败；先重连只读核实，禁止重启正在跑的job。

20:04更新：fresh30 r1在首次reset前因包装层过早读取原server尚未创建的frame_st_id退出，0完整局/0策略调用，自有进程已清理。原FAILED与修复前8源码快照保留。最小修复仅允许reset前记录None，reset后/策略路径仍严格校验；独立CPU复核通过。fresh30 r2已在全新目录prepare并正式运行，不复用r1结果，也不改接受规则、训练权重或完整任务分组。AutoDL原100最新60/100，纯归档watcher r2正常。
