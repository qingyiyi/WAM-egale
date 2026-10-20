# 当前短入口

更新：2026-10-20T11:30:55.620554+08:00。当前Cycle：[wam-research-20261016](cycles/wam-research-20261016/PROGRESS.md)。先[约束记忆](memory/INDEX.md)，再读当前四文；原长记录/trace按需，不展开全部旧索引。

## 已观察结果

四套原6000完整episode齐：每套10task×50init×T/C/P，成功数Long483/484/473、Goal463/464/467、Spatial472/466/464、Object497/498/499，各臂500。[完整质量/门](experiments/libero-four-suite-performance-20261019/QUALITY-RESULT.md)、[原逐局与成本](experiments/libero-four-suite-performance-20261019/RESULT.md)。八组合P50/P90都在旧长度界内，但max都>1.30；失败/800罚分、首两S严格配对FAILED、组件失败和训练/DEV曝光保留，不是独立TEST/非劣。无匹配500局MLP。

当前[M39内部FA2](experiments/single-layer-async-gemm-20261020/FA2-RESULT.md)：直接在自有整层单kernel内接入FA2 self/cross、split/combine。9真实签名×两attention的同输入18项全部逐位PASS；18相对旧async的有效KV转移/重复/RNG/输入及18单kernel trace通过。整层9输出仍FAIL，相对L2 0.133%–0.563%、最大0.5–1.5；Q/FFN2独立差异，video短中Norm2各12/8元素差异；长video首次分歧为FFN2。不是完整模型质量准入。

两个独立进程216样本：短action较旧async慢8%–9%，长action耗时降18%–21%、长video降43%；仍比native慢3.20–7.60倍。108CTA、255registers、82464B shared、2168B localSizeBytes；实测occupancy/流量UNKNOWN。387block＋18SDPA＋18Linear，GPU worker生命周期69.979秒、阶段836.9秒、6.79MiB，两次CPU编译失败保留，14owned PID/birth退出，两卡空。75本地/71远端保护项不变，四false保持。

旧[M39双缓冲](experiments/single-layer-async-gemm-20261020/RESULT.md)432样本/675block及PLAN900/registration240资源审计FAILED原样保留；新FA2阶段未重置旧预算。旧[M34](experiments/single-layer-warp-specialized-20261019/RESULT.md)9逐位FAIL、600调用及旧数值诊断仍有效，不能借attention局部通过改旧证书。

## 资源终态与硬截止

M35新同卡工程计时已收口，原面板PARTIAL，当前330/360完整局，固定10task/init49×三臂×3独立server进程重复。Goal/Spatial各90，Object60；Object1误入旧Goal/Spatial白名单，0局0teacher，保留不重试，仅接续原未启动2/3，最大330。Long三phase已全部终态并清理；四套本轮没有继续排队的GPU作业。逐arm数见[新计时结果](experiments/libero-four-suite-performance-20261019/ISOLATED-RESULT.md)。计时重复不增加独立质量N。

GPU0 Goal→Object、GPU1 Spatial→Long，每卡单worker/server与client各2线程，同case同卡；05:34:31停新、05:44:31清理（10/20），12 GPU生命周期h/1452000尝试/3GiB，不延期。计时仍有首两S组件FAILED，只作未准入算法工程成本观察，不是旧正式入口放行。核心ΣT_round/Σg；暖episode含reset/policy/cache/env，不含env.close/落盘，冷setup另列，client_total不叫纯E2E。[计时边界](experiments/libero-four-suite-performance-20261019/TIME-BOUNDARY.md)。

## 接口与已有强基线

TF64/每套选定权重、normalizer、tau=.007839680526450805、absΔσ×L2、BF16方向/FP32评分、K2B2/bonus/prefix_then_cost冻结。PF2=SST完整policy请求，T50更新/51 action调用含终端；reset/cache不计，不是bonus或状态恢复。所有臂同用512-token BF16 reset bank，首次构建/冷启动另列。actual796与失败罚分800分开。

D3072/H24/d128/FFN14336，action Q16、GEMM M32→224已有实际aten::addmm证据；attention实例仍Q16。action KV288/1152/2160，video本批272/1136/2144，cross512；candidate private KV不共享。[旧强cache/Graph与FFN组件](experiments/operator-profile-device-20261019/RESULT.md)已有0.7%–3.2%有限收益，不作新调度创新。M34未接成熟cache强路径。

M31静态子图、M32阶段内persistent、M33退化profile旧结果保留；M32新增persistent更慢，M33调216 CTA仍无正收益，NCU权限失败/硬件计数UNKNOWN，不借旧有限精度给新算术签证。[主张矩阵](experiments/single-layer-warp-specialized-20261019/CLAIM-EVIDENCE-MATRIX.md)：H1有限支持；H2同冻结7-query对照缺；H3无新增调度正收益；H4质量约束闭环证据未完成。

## 下一步与范围

先[内部FA2](memory/global/requirements/decision-先在自有整层算子内接入FA2.md)已完成有限核验；后续直接在optimized处理FFN2/Q/Norm2数值与短Q资源压力，再考虑跨阶段ready调度。沿现有记录，不另建实验登记；后继未启动。原[M38慢点](experiments/whole-layer-bottleneck-20261020/RESULT.md)与M39双缓冲保留。

M36[RoboTwin调研完成](experiments/robotwin-migration-survey-20261019/RESULT.md)：官方LingBot-VA权重/50唯一任务，100有效seed/task，三臂15000局/condition。Q32/16有效动作通道/pool静态9792需新draft/cache接口，不继承LIBERO阈值；0第二环境实验。CUDA仍私有12.4/Torch2.9 cu126，远端driver535不升级CUDA13/系统。

## 保护与日历

engineering_quality_screen=false；independent_test_complete=false；noninferiority_established=false；formal_performance_permission=false。旧FAILED/retention/NO_GO/paused72/96、MLP0、22缺收据/default A_off不改。

baseline及依赖冻结，仅optimized变化；投机源/配置/运行依赖、原tile/HW、state/cycle/rules保护，不建缺失cycle.yaml、不升级/旧sync。BAAI仅~/cxz；AutoDL不再SSH/关机UNKNOWN。用户允许性能统一远端，替代旧地点限制；不杀他人/改服务/时钟。无自动新训练/120/独立TEST/旧45chunk/第二环境/30层大融合。

M17原10/21 16:56:31停新、17:11:31清理、18723000尝试/15GiB不重置。只main公开审查源码/HW/小证据，无权重/PT/KV/DSO/凭据。10/22 go/no-go、10/23冻结、10/28主表、11/01 18:00中文包；正文由用户飞书整理。官方注册/正文/补充北京11/11、11/17、11/24各19:59，未单列abstract截止，作者profile UNKNOWN。[日期与三情形](experiments/libero-four-suite-performance-20261019/DATES-AND-PLAN-20261020.md)。旧方案见Cycle history，不由历史“下一步”重启任务。

M37（2026-10-20T09:33:50.273752+08:00）已完成：Long全尝试1.107×/共同成功1.344×，同一失败case三重复拉长；Object1.216×受不变非action开销与多3.4%chunk影响。成本完整保留。远端唯一几何匹配CUTLASS Q/FFN2四项原逐位FAIL；4block/12原语/8heuristic，185.698秒，owned退出；异步整层集成未触发。见experiments/device-contract-20261020/RESULT.md和TASK-COST-DIAGNOSIS.md。旧四false/FAILED与M34/M35终态不变，无GPU排队。
