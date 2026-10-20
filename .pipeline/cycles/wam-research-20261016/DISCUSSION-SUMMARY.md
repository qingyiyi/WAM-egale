# 当前讨论与授权

最新用户要求现在直接实现：整个单层Transformer在一个persistent kernel内，把全部算子切tile交给职责不同的warp，借鉴MegaMoE；明早汇报实测速度、精度与问题。只一层，不做30层大融合。baseline及共享依赖冻结，仅修改optimized；投机代码/配置与原tile不改。

用户允许四套LIBERO速度统一远端测，替代旧“最终E2E必须本机”的地点条款；每卡计时单worker且无观察到争用，M34 GPU验证等对应计时卡清理，CPU实现可并行。本机可用自然空闲窗口先核对精度；不杀他人。AutoDL已归还不再连接，BAAI所有写入限~/cxz。

当前M34结构实测成立但精度/性能NO-GO，已按600调用有界诊断收口；没有授权放宽原门。M35原360面板因Object1实施失败最多330，原失败保留、不自动重跑或延长截止；原未启动2/3可以按既定预算接续。RoboTwin仅调研，没有新增实验/安装/权重下载权限。

四质量权限保持false，旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off保留。不能用成功子集、调用减少、match-v1、单kernel结构或普通batch/cache宣称质量与协同加速。新训练/120/独立TEST/受阻45chunk不自动启动。

只main公开核对源码/HW/小证据，模型/PT/KV/图像/DSO/工具链/连接配置/凭据不公开。中文正文逐段交用户飞书，不建本地初稿。10/22 go/no-go、10/23冻结、10/28表、11/01 18:00老师中文证据包；官方投稿日期见日期报告，作者profile仍UNKNOWN。

旧讨论原文保留[归档](history/M34-complete-20261020/DISCUSSION-SUMMARY.md)；保护state/cycle/rules、不升级或旧sync。当前进度以PROGRESS与实际终态为准。

## 当前优先级

用户先要求profile整层慢点，再明确继续实现；M38/M39已收口，GEMM双缓冲带来相对旧原型有限提速，仍未匹配Torch数值/速度。资源登记字段不一致作为失败保留，不能以脚本exit0替代全部审计通过。

最新指令：不考虑另一agent带读版本，直接改optimized，不再另起实验登记；这是对原“独立入口保护带读”的替代。Maintain记录在[直接在optimized迭代](../../memory/global/requirements/decision-直接在optimized迭代且不另建登记.md)。baseline/投机与既有准入门继续保护，已有本轮记录继续保留最小必要证据。

用户询问更大tile，当前GEMM64×64×32、attention Q16/KV64。建议优先K32→64，再按形状评估N128；B2 M32不能盲目扩大M，B14不能变成跨候选互相注意。尚未运行这些候选，不把提议记为实测收益或已启动工作。
