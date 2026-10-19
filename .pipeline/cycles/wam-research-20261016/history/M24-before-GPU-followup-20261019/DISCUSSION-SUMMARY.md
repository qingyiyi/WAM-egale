# 当前讨论与授权摘要

最新用户上传“完整block基线优先”的网页方案，要求今晚先做、早上9点汇报，并明确同步HW算子进展。固定ca5f37e为科学base；MODE=CPU_AUDIT_AND_BASELINE_PREPARE，没有新GPU、CUDA context、完整模型加载、模拟器/训练/新episode/迁移/production patch授权。独立namespace已实现B0与CPU检查、预算/新授权约束的GPU验证计时入口；本轮仅CPU验证，不先开发persistent scheduler。CPU≤2/8h、新增≤1GiB；原tile与state/cycle/rules不动，不借M22余额。GPU120min只是待申请建议。

本机负责算子、BAAI既定质量链继续；用户已取消进一步开发迁移并自行kill本机遗留模拟器，已确认退出。新机Codex0.162.1装于~/cxz，登录待用户，不代表会话/最新算子已迁移。远端代码、依赖、缓存、临时、凭据写入永远限~/cxz；不拷SSH私钥、不杀他人/改系统。AutoDL不再SSH，关机UNKNOWN。

M22仅去fast_math后有限attention/block/30层/冻结完整policy严格数值通过；实际替换仅attention核心，不把验证范围写成全自有block。新性能有慢点和NVML空载异常，不签全形状稳定对齐。新B0是显式顺序混合执行，可用Torch GEMM/norm，但不能把host调用当persistent device基元。

既定评测TF64/TF_ON、normalizer、tau、K2B2/bonus/prefix_then_cost不变。PF2=SST完整policy，T50更新/51 action调用含末尾，reset清零、cache不计，不是bonus/物理恢复。所有臂用512-token BF16优化reset，cold另列；horizon800，actual796和罚分800分列。Goal/Spatial各自训练对应TF；Object余14/ON/匹配T及全500自动排队；90无可靠同代权重仍BLOCKED。

Long首轮各500已齐；三轮是偏好，后两轮seed/预算未冻结，不自动运行。训练/DEV曝光和首两S严格配对FAILED不改；不能签独立TEST或只报共同成功隐藏失败。迁移失效Long8×3、Object case0/1的显式补齐授权保留；原attempt与成本不删，不计独立样本。基础设施冗余大产物需核依赖再清理，自然失败保留。

四false、旧NO_GO/retention/22缺收据、4f04 paused72/96与MLP0、本机live77、全局A_off不变。不自动新训练、120/独立TEST、新TF正式45chunk、架构搜索、第二环境。T25/T17对T50须另登记，video20同网格。原M16/M17预算起点及绝对截止不因迁移/补缺重置，巡检不自动重试。

用户明确GitHub只是网页审查中转，仅维护main进度镜像，不再开发布分支。旧review/RESULTS.json保持历史身份；单次审查使用固定commit。中文论文逐段会话交飞书，不新建本地论文初稿。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文证据包。完整旧讨论逐字保留本次history。
