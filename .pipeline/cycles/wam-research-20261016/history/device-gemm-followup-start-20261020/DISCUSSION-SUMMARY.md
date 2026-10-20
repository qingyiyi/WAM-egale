# 当前讨论与授权

用户最新明确[修复后接续原RoboTwin迁移](../../memory/global/requirements/decision-修复后接续RoboTwin有限迁移.md)，替代暂缓/仅修复的优先级。114个固定专家初态已齐，GPU1原Target首6局有界接续；后续训练/DEV未完整实现。原预算、失败、驱动保护保持。用户另[明确批准首case视频入口失败后补跑](../../memory/global/requirements/decision-允许补跑RoboTwin首局视频入口失败.md)，只覆盖该case，原PARTIAL及绝对deadline不变。


M39有限数值通过、速度NO-GO。M40专家None抓取异常按固定官方筛选规则跳过，不是模型失败；旧FAILED保留。未启动的模型/训练不写运行中，动态事实看PROGRESS及现场。

用户要求整个**单层**Transformer在一个persistent kernel内，tile由职责明确的warp协作，借鉴MegaMoE；不是30层融合。baseline及依赖冻结，直接在optimized迭代，不为带读另建代码版本/研究ID。当前有完整单kernel和FFN行组就绪交接，其他阶段仍有barrier，FA2计算warp仍自行load/MMA/softmax/store。当前ready更慢，默认False；不能把结构实现当作性能创新通过。

用户提出用后续无前置依赖任务填补小于108CTA尾波：合理待验证设计，非既成性能结论。空闲CTA逐tile取ready工作、主链优先，cross K/V可提前，cached时不能重算制造工作；warp分工须形成真实流水并移除不兼容grid barrier。本次未新增算子GPU/改代码。

BAAI所有写入仅~/cxz，允许性能统一远端；不改系统驱动/服务/时钟，不杀他人。AutoDL归还不再连接。原GPU0≤8h、GPU1≤24h、CPU准备10/20 18:20北京及stage/call/存储帽保留。客户端535配套库/原rt保持。

TF64/各套权重与tau/selector等旧身份不改；PF2=SST完整policy请求，不是bonus或物理状态恢复。所有可运行的LIBERO臂同等使用完整512-token BF16 reset bank，冷启动另列；RoboTwin已有108条完整bank及张量拷贝逐位检查，完整任务cached/uncached等价仍未实测。原始FAILED/retention/NO_GO/paused72、MLP0、22缺收据/default A_off与四质量false保留。完整6000局和计时重复不混成独立TEST。新算子有限精度不能继承为机器人质量或正式性能许可。

只main公开核对源码、HW和小证据，不公开模型/PT/KV/相机/DSO/工具链/私有SASS/连接配置/凭据。论文中文逐段交用户飞书，不建本地初稿。10/22 go-no-go、10/23冻结、10/28表、11/01 18:00老师证据包；官方日期/profile UNKNOWN项见日期报告。

旧讨论与授权演变完整保留[本轮历史](history/ordered-followup-20261020/DISCUSSION-SUMMARY.md)。不要从旧“未授权/下一步”段落自动推导当前权限，也不修改保护state/cycle/rules或升级/全量sync。
