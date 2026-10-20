# 长期记忆索引

- [先逐阶段对齐再异步](global/requirements/feedback-先逐阶段对齐原版性能再做异步调度.md)（guideline，M39）：先同输入/同边界对照原版、自有阶段独立、自有阶段嵌入整层，分清原语与整层资源/调度成本，再优化与ready调度。


这是从旧228条记录提炼的当前主题入口；原记录和原分类索引保留，按具体问题追溯。临时进度与 AutoDL 完成数看 Cycle，不作为长期记忆复制。

## 当前资源偏好

- [接续RoboTwin并先分析当前算子开销](global/requirements/decision-接续RoboTwin并先分析当前算子开销.md)（constraint，M39/M40）：恢复未开始的既定迁移阶段；当前算子先实际profile和源码讲解。

- [继续实现并评估更大tile](global/requirements/decision-继续实现并评估更大tile.md)（constraint，M39）：用户明确接续实现/验证，允许更大tile；RoboTwin继续关注，不自动重启过期链。

- [先将B14整层耗时压到Torch两倍以内](global/requirements/decision-先将B14整层耗时压到Torch两倍以内.md)（constraint，当前算子）：短/中/长KV同轮完整层均为目标；精度与公平基线不放宽，尚未达成。

- [继续算子优化并先核对当前瓶颈](global/requirements/decision-继续算子优化并先核对当前瓶颈.md)（constraint，M39）：用户同意接续，区分当前实测与旧profile；明确复用成熟设备端GEMM源码并保留整层单kernel。

- [允许补跑RoboTwin首局视频入口失败](global/requirements/decision-允许补跑RoboTwin首局视频入口失败.md)（constraint，仅首case）：用户明确允许修复后补跑0teacher/0动作的place_object_basket/valid00，原PARTIAL、预算与deadline保留。

- [修复后接续RoboTwin有限迁移](global/requirements/decision-修复后接续RoboTwin有限迁移.md)（constraint，当前Cycle）：用户恢复原六任务有界迁移链，先固定初态/原Target；原预算、截止与全部失败保留。

- [优先修复RoboTwin环境且保持系统驱动](global/requirements/decision-优先修复RoboTwin环境且保持系统驱动.md)（constraint，当前Cycle）：用户恢复环境排查，核查A100/RTX适用性；不改驱动、不自动采购，和MegaMoE算子审查合并汇报。

- [暂缓RoboTwin，优先收口算子结果](global/requirements/decision-暂缓RoboTwin并优先收口算子结果.md)（constraint，当前Cycle）：用户暂无兼容环境，原迁移链暂缓，不自动重启；旧授权/预算/失败保留。
- [原始action保留原算子](global/requirements/decision-原始action路径保留原算子.md)（constraint，当前集成边界）：原始单状态action路径继续用原算子，当前先单独讨论B14成本；不等于新GPU或生产集成授权。

- [执行RoboTwin迁移与算子有限收口](global/requirements/decision-执行RoboTwin迁移与算子有限收口.md)（constraint，当前Cycle）：用户已批准两卡计划；GPU1迁移≤24h、GPU0算子≤8h，先原Target/数值，再后继，旧门保护；替代此前仅规划。

- [先在自有整层算子内接入FA2](global/requirements/decision-先在自有整层算子内接入FA2.md)（constraint，当前算子）：先self/cross设备核心与误差核验，再研究跨阶段同步；RoboTwin仅调研。

- [直接在optimized迭代、不另建登记](global/requirements/decision-直接在optimized迭代且不另建登记.md)（constraint，当前算子）：解除为另一agent带读冻结源码/另建版本的安排，沿现有记录保留验证证据。

- [依据profile继续单层tile流水优化](global/requirements/decision-依据profile继续单层tile流水优化.md)（constraint，本轮M39）：用户已授权继续实现；其中独立带读版本/另行登记安排已被上方最新决定替代，原准入门不变。

- [当前优先解释并profile整层算子退化](global/requirements/decision-当前优先解释并profile整层算子退化.md)（constraint，当前算子）：先测慢点，不等待数值修复；准入门不变。

- [继续数值契约与异步tile](global/requirements/decision-继续数值契约修复后验证单层异步tile流水.md)（constraint，当前算子）：新有界阶段承接M34负结果；不借旧预算、不放宽质量门。

- [允许远端统一测性能](global/requirements/decision-允许远端统一测量LIBERO与算子性能.md)（constraint，当前性能）：解除最终E2E必须回本机的地点限制；四套三臂完整质量/长度/成本，正式计时无争用。

- [整层统一persistent执行器](global/requirements/requirement-单层Transformer统一persistent内核调度全部tile.md)（constraint，当前算子）：全部单层tile在一个kernel中，由职责明确的常驻warp分工执行；M32局部CTA循环尚未完成该目标。

- [单层persistent有限验证与负结果](global/knowledge/knowledge-单层persistent有限验证与性能负结果.md)（reference，M32）：9block/90状态/48原语通过，B14比静态慢4.42%–6.26%；后续只优化单层。

- [本轮仅单层完整block](global/requirements/decision-当前算子仅实现单层完整Transformer-block.md)（constraint，M32）：tile/persistent以一层为边界，停止本轮30层迁移与重放；其中最终E2E地点已被最新远端许可替代。

- [算子远端实现、本机最终E2E旧条款](global/requirements/decision-算子远端实现验证与本机最终E2E.md)（constraint，本轮算子）：远端隔离与baseline冻结仍有效；最终E2E必须本机的地点限制已被上方最新远端许可替代。

- [optimized独立执行与静态tile子图](global/knowledge/knowledge-optimized独立执行与静态tile子图有限验证.md)（reference，M31）：9block/72状态有限逐位与独立进程隔离；full30/性能待验，persistent未实现。

- [冻结baseline，仅在optimized推进tile优化](global/requirements/decision-冻结baseline并仅在optimized实现tile优化.md)（constraint，算子代码）：优化从等价路径开始；连同共享依赖冻结，先隔离会变化的源码/构建产物。

- [Transformer子图应覆盖self/cross-attention](global/requirements/feedback-Transformer子图设计应覆盖self与cross-attention.md)（guideline，本轮算子设计）：依据完整真实DAG组织tile；FFN局部实验不代表整体，具体实现仍需验证。

- [tile与persistent warp为当前算子主线](global/requirements/decision-算子tile化与persistent-warp调度为主线.md)（constraint，本轮算子）：设备tile、常驻worker与依赖调度，替代旧仅路线B阶段优先级；不预设收益或扩大质量权限。
- [MegaMoE作为思想参考](global/requirements/preference-MegaMoE作为tile调度思想参考.md)（guideline，本轮设计）：结合HW与SM80借鉴任务/warp/就绪机制，不要求复刻官方实现；具体策略仍需验证。

- [完整native block有限对齐](global/knowledge/knowledge-完整native-block有限对齐.md)（reference，M28）：完整顺序原生链、六actual30有限逐位与两进程动态block齐；最强B14/Graph及闭环尚未验。
- [投机去噪近邻与创新边界](global/knowledge/knowledge-投机去噪近邻与创新边界.md)（reference，2026-10-19）：直接先例已核验；潜在新增调度仍需强baseline与有效提交成本证据。
- [完整block基线与优化版均为原生算子](global/requirements/decision-完整block基线与优化版均改为原生算子.md)（constraint，当前算子）：最新范围扩展；归档旧baseline后可改造，先完整顺序原生链，再对齐精度/性能；投机与原tile不改。

- [优先定位短KV与B14算子性能回归](global/requirements/feedback-优先定位短KV与B14算子性能回归.md)（constraint，本轮算子）：逐点对照固定Torch调用链定位慢点，保留原始计时；不预设根因。

- [算子先完成强基线profile与设备原语](global/requirements/decision-算子先完成强基线profile与设备原语验证.md)（constraint，当前1/2）：用户已同意执行并要求结果和源码讲解；后续scheduler另阶段。

- [远端并行评估自动接续、本机聚焦算子](global/requirements/preference-远端并行评估自动接续并聚焦本机算子.md)（guideline，当前剩余评估）：核资源与状态隔离后最多每卡两个worker，确认实际自动派发；原预算与科学边界不变。

- [算子等待GPU自然空闲](global/requirements/preference-算子测试等待本机GPU自然空闲.md)（constraint，本轮算子）：用户再次选择自然空闲，不反复协调窗口；原预算/截止不重置。

- [算子仅baseline/optimized，投机侧不改](global/requirements/decision-算子代码只维护baseline与optimized且不改投机侧.md)（constraint，算子代码）：独立目录收拢、删除无依赖旧实现；先完成30层与性能基线，再讲解。

- [算子交付明确源码与验证范围](global/requirements/requirement-算子交付明确源码入口与验证范围.md)（constraint，当前算子交付与带读）：提供完整路径与准确行号；中文每次一节，以具体tile解释线程、数据流、同步和限制，先答追问再推进；本次教学只读，不触发新实验。

- [先直接复制Torch调用链完成复现](global/requirements/decision-算子先直接复制Torch调用链完成复现.md)（constraint，当前算子基线）：先可替换的精度/性能基线，允许抄原实现；自有device基元/调度另议，计时噪声与实现正确性分列。

- [Object固定14局补采并接续](global/requirements/decision-Object只补采输出冲突的14局并接续原任务.md)（constraint，本次M17修复）：旧16/失败保留、新输出隔离；14局/56000次上限，原截止不重置。

- [本机空闲GPU接续完整block验证](global/requirements/decision-本机GPU空闲后接续完整block验证.md)（constraint，本次M24 GPU阶段）：最新授权空闲即接续，单A100≤120分钟、CPU≤2；旧CPU账本保留，汇总两台机器实际状态。

- [完整block顺序基线优先并同步HW](global/requirements/decision-算子先建立完整block顺序基线并同步HW.md)（constraint，本轮算子）：固定ca5f37e，原CPU阶段历史；今晨GPU接续已有新授权见上，实际结果/PENDING分列。

- [GitHub只保留main进度镜像](global/requirements/decision-GitHub仅保留main作为当前进度公开镜像.md)（constraint，公开发布）：本地核对后的当前内容直接推main；旧commit保留，不再自动创建review/experiment发布分支。

- [暂不迁移开发、模拟器由用户终止](global/requirements/decision-暂不迁移开发工作且本机模拟器由用户终止.md)（constraint，本次M23）：取消因CPU拥塞进一步迁移；只报PID，不代kill；已运行远端评测继续。

- [BAAI的Codex启动入口](global/knowledge/decision-BAAI的Codex安装与启动入口.md)（reference，限定新机）：`~/cxz/bin/codex`，配置/缓存均限~/cxz；版本和登录状态动态核对，不代表会话或最新源码已迁移。

- [任务未结束时定期核验服务器](global/requirements/requirement-任务完成前定期核验服务器运行与进度.md)（constraint，active-server-jobs）：低开销巡检进程身份、日志推进、终态和原截止；当前约每5分钟，替代只近ETA才检查的安排，不自动重试任务。

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

- [单卡规划RoboTwin与算子并行](global/requirements/decision-单卡规划RoboTwin迁移并行推进算子.md)（constraint，当前Cycle规划）：用户要求远端一张GPU迁移方案及算子下一步；具体GPU编号/预算是建议，本轮不等于启动第二环境。

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
