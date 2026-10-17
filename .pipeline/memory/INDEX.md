# 长期记忆索引

这是从旧228条记录提炼的当前主题入口；原记录和原分类索引保留，按具体问题追溯。临时进度与 AutoDL 完成数看 Cycle，不作为长期记忆复制。

## 当前资源偏好

- [本轮AutoDL暂不扩容](global/requirements/decision-本轮AutoDL不扩容并先清理废弃资源.md)（constraint，M17）：先核依赖清理废弃项，必要时分阶段轮换可重建权重；原始结果及在用组件保留。
- [本机减负与容量告知](global/requirements/preference-本机压力时优先远端计算与容量告知.md)（guideline）：优先现有AutoDL；容量不足先明确告知，项目清理仍核依赖。

## 必须读取：约束级

- [四套LIBERO同配方TF与三臂评估](global/requirements/decision-四套LIBERO同配方TF与三臂评估.md)：用户已新批准三个小套件各自训练与T/TF/PF2完整评估，独立预算；不再只是扩展建议。

- [全量三臂与90独立draft](global/requirements/decision-Long与LIBERO90全量三臂评估及独立draft训练.md)、[AutoDL用至完成](global/requirements/decision-AutoDL全量研究使用至完成.md)：新全量研究已批准双卡；旧23:11仅M15历史，新job单独有界登记。
- [质量并行与独占计时分离](global/requirements/decision-质量并行评估与独占重放计时分离.md)：保存有限真实fixture；replay阶段成本与完整任务E2E分开。

- [实验默认优化reset](global/requirements/decision-实验默认启用等价reset优化.md)（采样/质量/性能所有对照臂同等启用已验证512-token BF16 bank，冷启动另列）

- [PF2执行与12h窗口](global/requirements/decision-PF2源码执行与AutoDL十二小时窗口.md)（仅已完成M15的历史起止；新全量授权见上，不重写旧账本）

- [AutoDL直接源码运行](global/requirements/requirement-AutoDL直接源码运行且不做smoke或SHA.md)（不做smoke、测试脚本或SHA；必要审计随获准的完整任务记录）
- [PF2优先只读与新双卡核查](global/requirements/decision-PF2优先只读方案与新双卡核查.md)（历史只读阶段；执行暂停现由本轮12h批准替代，PF2优先顺序保留）

- [实验、数值与证据边界](global/rules/requirement-实验与证据边界.md)
- [资源授权与 EGL 并行规则](global/rules/requirement-资源与EGL并行规则.md)
- [本轮夜间截止](global/requirements/decision-本轮TF验证截止与晨报时间.md)（仅TF收口实验：10/17 10:00，替代原09:00用户上限）
- [恢复固定验证与项目内清理](global/requirements/decision-恢复固定验证并限定项目内清理.md)（用户已明确继续；新Attempt补缺、仍按10:00收尾，只删项目内核实无用项）

## 按主题读取

| 主题 | 入口 |
|---|---|
| HW版本与读取 | [本轮工具与加载决定](global/requirements/decision-HW版本与读取策略.md) |
| 用户写作与汇报偏好 | [中文论文与汇报](global/requirements/preference-中文论文与汇报.md) |
| 当前因果主线与算子保护 | [投机重组与协同调度](global/requirements/decision-投机重组与多候选协同调度.md)；[原双路线边界](global/requirements/decision-双路线与算子保护.md)保留追溯 |
| draft 与成功率状态 | [投机候选和质量](global/knowledge/decision-投机候选与闭环质量.md) |
| 当前阶段执行优先级 | [优先完成draft收口](global/requirements/decision-当前阶段优先完成draft收口.md)（当前Cycle约束） |
| 暂停历史与网页复核 | [前次暂停](global/requirements/decision-暂停TF实验并先外部文献复核.md)（执行暂停已被新恢复决定替代，历史证据保留） |
| 主draft结构与投入 | [固定TF64，集中训练与效果](global/requirements/decision-TF64固定为主模型与训练收口.md)（当前Cycle约束；MLP仅对照） |
| 训练数据、σ、阈值、PF | [训练与接受标准](global/knowledge/decision-训练数据与接受标准.md)（含PF周期定义、旧pilot和FLASH适用边界；非启用决定） |
| 速度口径与 reset | [性能证据和 reset](global/knowledge/decision-性能证据与reset使用.md) |
| 论文评测范围 | [LIBERO范围与Long主实验](global/knowledge/decision-LIBERO评测范围依据与Long证据边界.md)（含三小套件社区权重；文献事实与未采纳建议，未把问句记为取消90或启动补测） |
| tile 与 B14 verifier 接口 | [算子基线边界](global/knowledge/decision-算子基线与接口限制.md) |
| 截稿与规划 | [投稿时间](global/knowledge/decision-投稿时间与研究规划.md) |

历史映射见 [原记录清单](../maintenance/hw-upgrade-20261016/LEGACY-RECORD-INVENTORY.json)，仅在查旧问题时打开；不把旧 GPU 暂停、旧“未完成”或早期计划当成当前指令。
