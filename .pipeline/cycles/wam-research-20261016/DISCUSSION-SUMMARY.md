# 当前讨论与授权摘要

用户最新明确将主任务转为实际Torch算子对齐：允许独立修改/复用固定Torch源码，目标是替换进模型后精度一致、性能接近，并授权同步HW与GitHub及网页审查prompt。M22独立登记，不重置M21预算；旧tile原件保留。共享GPU精度继续，正式计时等待无争用。此前“先讨论，不做新修正”的限制在M22被此新授权替代。

用户本轮要求尽可能完备的算子与Torch实际性能/精度数据，并明确允许本机补原始输入与verifier规模的短/中/长KV。质量评测尽可能迁新机，本机供Transformer研究；本机Object已在完整episode后自然释放，不沿用自动接续本机400局。

等待他人GPU任务自然结束，持续低频监控、空闲即接续，既有授权不重复询问。用户确认窗口后开始；剩余独占不足20分钟时又明确性能优先、输入输出后验，因此独立r6短协议按时结束，原r5保持PARTIAL。随后用户明确授权共享GPU继续精度及完整I/O并统一汇报；这一授权不追溯延长旧独占，不把共享用时混入计时。上述决定已通过Maintain限定当前Cycle/M21保存。

M22后续受控修正：只移除fast_math，六完整query和三完整policy冻结配对均逐位/状态通过，108attention与54block也逐位；旧M21失败仍是旧二进制真实结果。新二进制性能未复测，当前他人context不能当正式独占；没有机器人成功率新证书。

本轮M21结果明确分范围：单attention/block容差通过≠逐位；standalone完整30层六query全非逐位、三例容差失败；shared/guarded有限六query逐位及输入/KV/RNG通过≠任意输入/闭环证书。性能block波动与NVML计数异常保留，不宣称稳定对齐。原tile代码/HW与撤回融合不动。

四套评测保持原配方，结果齐后再讨论调整；不因提问自动新训练Long、120/TEST或扩大环境。TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost不变。PF2为SST完整policy chunk，canonical T50更新/51 action调用含终端，不是bonus或状态恢复。全部评测默认512-token BF16优化reset，冷启动另列。

Long首轮三臂各500、1500局已完整交付；成功来自真实LIBERO终态，不由acceptance判断。失败计SR与800罚分，实际796分列，双方成功长度仅补充。50init存在训练/DEV暴露，C36/P66严格配对FAILED不改；三轮均值是后续偏好，追加两轮seed/预算未冻结。原错误冗余大产物可核依赖清理，但不删自然失败来抬SR。

用户已批准迁移前失效Long八case×三臂、Object case0/1独立补齐，原attempt/预算保留、不算新独立样本。BAAI所有文件/环境/缓存/临时项限~/cxz；旧AutoDL已核清理、不再SSH，是否关机UNKNOWN。三小套件各自训练相应TF，不拿Long draft冒充；90无可信同代权重仍BLOCKED。

原FAILED/retention/22清理缺证、4f04 paused72/96及MLP0、本机live77、默认A_off与四false不变。计时与质量分开，未做完整scheduler/chunk/动作或机器人E2E新验收。原M16/M17起点与绝对截止不因迁移、补缺或当前报告重置。

用户要T25/T17对T50强对照，video20不变、完整sigma网格；新增执行尚未登记。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。中文正文会话逐段交飞书，不建本地论文初稿。

前完整讨论和过程逐字保留history/*-before-operator-final-20261018-204358.md，当前摘要只作轻量恢复入口。
