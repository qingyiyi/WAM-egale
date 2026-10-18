# 长期记忆索引

这是从旧228条记录提炼的当前主题入口；原记录和原分类索引保留，按具体问题追溯。临时进度与 AutoDL 完成数看 Cycle，不作为长期记忆复制。

## 当前资源偏好

- [当前优先整模型Torch算子对齐并同步GitHub](global/requirements/decision-当前优先整模型算子Torch对齐并同步GitHub.md)（constraint，当前Cycle/M22）：独立实现修复、允许复用固定上游源码；整模型输出/状态与性能验证；同步源码/HW和网页审查prompt。旧基线和质量门保留。

- [有限窗口先性能、精度后验](global/requirements/decision-有限独占窗口优先性能采集后验精度.md)（constraint，本轮M21）：剩余独占不足20分钟，独立缩短性能采集协议并预留清理；旧协议PARTIAL与待验状态保留。

- [评测迁新机、本机供算子](global/requirements/decision-评测迁往新机本机用于算子对照.md)（constraint，当前Cycle）：评测迁出本机；补原始/B14短中长算子数据，持续低频等他人自然结束、空闲即接续，不重复询问；替代自动接本机400局的安排。

- [算子对照完整数据](global/requirements/requirement-算子对照须提供完整性能精度数据.md)（constraint，operator-validation）：提供全部已测输入/进程/轮次的性能与误差、原始证据及缺口，不以局部通过外推整模型。

- [接近预计结束时每三分钟检查](global/requirements/preference-接近阶段预计结束时每三分钟检查.md)（guideline，当前Cycle/M17）：替代全程四分钟汇报；平时休息，近阶段ETA再查，原GPU任务/guard不暂停。

- [已授权任务及时接续](global/requirements/feedback-已授权任务依赖满足后及时接续GPU.md)（guideline，当前Cycle）：核资源/隔离后提高原任务吞吐，阶段间及时接续，不为占卡增加研究臂。

- [先完成既定四套LIBERO再讨论调整](global/requirements/decision-先完成既定测试并在Long收口后讨论.md)（constraint，M16/M17）：保持当前方案，四套结果齐后再讨论调整；已授权链继续，原门和预算不变。

- [先核实整体加速](global/requirements/preference-先核实整体加速并暂缓长尾深入诊断.md)（guideline，当前Cycle）：暂缓长尾新增深入实验，优先计时边界/真实重放准备；原成功长度与全体SR/罚分保留。

- [10/18晨间先交Long首轮](global/requirements/decision-晨间优先交付Long首轮证据.md)（constraint，当前Cycle）：09:30前真实完整覆盖/缺项汇总，不抢跑后两轮。
- [三轮均值与波动偏好](global/requirements/preference-Long重复评估汇报三轮均值与波动.md)（guideline）：建议轮间固定不同eval seed，轮内三臂配对；后两轮预算尚未登记。

- [指定失效任务补齐](global/requirements/decision-仅补齐明确授权的基础设施失效任务.md)（constraint，M16/M17）：Long八case×三臂、Object case0/1新namespace补齐，原预算/身份不重置。
- [基础设施错误产物清理](global/requirements/decision-清理基础设施错误产物但保留有效任务失败.md)（constraint）：先核活跃依赖；自然完整算法失败是有效数据，不能删除来抬高成功率。

- [迁入BAAI后再释放AutoDL](global/requirements/decision-将已授权研究迁往BAAI并核验后释放AutoDL.md)（constraint，当前Cycle）：匹配环境、全部原件与未开始工作去重迁移，确认新机接续及旧机清理后才告知关机；原预算不重置。

- [新机接入前继续推进](global/requirements/decision-新服务器接入前继续推进已授权研究.md)（constraint，当前Cycle）：先备份/分析，可新namespace小批次接续未开始完整组；原件、原预算和可暂停边界保留。
- [旧AutoDL阶段收尾后暂停换机](global/requirements/decision-旧AutoDL阶段收尾暂停并准备换机.md)（constraint，当前Cycle）：收尾已进入完整配对，阻止后续自动开工，备份与清理核验后再释放旧机；研究范围和原账本不变。
- [空闲本机可分担](global/requirements/decision-本机GPU空闲时分担完整研究任务.md)（constraint，当前Cycle）：先核资源，尚未执行完整分片可放本机，同case对照同卡，原预算和计时边界不变。
- [本轮AutoDL暂不扩容](global/requirements/decision-本轮AutoDL不扩容并先清理废弃资源.md)（constraint，M17）：先核依赖清理废弃项，必要时分阶段轮换可重建权重；原始结果及在用组件保留。
- [本机减负与容量告知](global/requirements/preference-本机压力时优先远端计算与容量告知.md)（guideline，当前Cycle）：现有新机主力、本机补充；依赖满足及时接续、逐资源汇报任务/ETA，容量不足先告知。

## 必须读取：约束级

- [少步数Target强对照与投机理由](global/requirements/requirement-投机须与少步数Target比较质量成本.md)：论文加入直接减少action步数的质量—成本对照；推荐25/17对原50，不预设投机获胜，新增执行另登记。

- [新BAAI服务器仅在~/cxz内工作](global/rules/requirement-新BAAI服务器仅在用户主目录cxz内工作.md)：该主机所有项目文件、环境、缓存与临时文件都限此目录；迁移与原授权接续已批准，仍不准越界写入。

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
| 训练数据、σ、阈值、PF | [训练与接受标准](global/knowledge/decision-训练数据与接受标准.md)（含全10任务FIT174、新CONT生成FIT20/CAL10及PF边界；非新增训练或启用授权） |
| 速度口径与 reset | [性能证据和 reset](global/knowledge/decision-性能证据与reset使用.md) |
| 训练/验证与标准评测 | [数据划分与独立性](global/knowledge/decision-训练验证与LIBERO标准评测的区别.md)（reference）：同套任务训练/闭环是常见做法，不把曝光面板签作独立TEST；新建议不自动改协议 |
| 论文评测范围 | [LIBERO范围与Long主实验](global/knowledge/decision-LIBERO评测范围依据与Long证据边界.md)（含三小套件社区权重；文献事实与未采纳建议，未把问句记为取消90或启动补测） |
| tile 与 B14 verifier 接口 | [原算子基线边界](global/knowledge/decision-算子基线与接口限制.md)；[10/18真实B2/B14完整验证边界](global/knowledge/decision-真实B2与B14算子完整验证边界.md)（reference：standalone完整30层失败、shared/guarded有限六query通过，性能仍探索性） ；[去fast_math后新版本有限对齐](global/knowledge/knowledge-去fast_math后的算子有限逐位对齐.md)（reference：108attention/54block/六full30/三完整policy逐位/有效状态通过，新二进制性能待测，旧失败保留） |
| 截稿与规划 | [投稿时间](global/knowledge/decision-投稿时间与研究规划.md) |

历史映射见 [原记录清单](../maintenance/hw-upgrade-20261016/LEGACY-RECORD-INVENTORY.json)，仅在查旧问题时打开；不把旧 GPU 暂停、旧“未完成”或早期计划当成当前指令。
