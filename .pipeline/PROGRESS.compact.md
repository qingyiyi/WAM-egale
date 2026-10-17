# 当前研究恢复入口

更新时间：2026-10-17 08:43，北京时间。派生短入口；原始证据/FAILED/tile保留。先读[索引](INDEX.md)、[上下文策略](CONTEXT-POLICY.md)、[约束记忆](memory/INDEX.md)，再恢复当前Cycle；不默认展开全部旧索引。

## 当前暂停与最新结论

**用户10/17早明确暂停实验，先写网页版文献/开源调研Prompt并同步GitHub。GPU已清理，不自动恢复；旧10:00上限不是继续运行授权。** [暂停决定](memory/global/requirements/decision-暂停TF实验并先外部文献复核.md)仅允许必要CPU交接/发布。TF64仍为主draft研究结构，MLP只作等容量ablation，不新开架构/seed/训练搜索。

[本轮结果与五报告索引](experiments/tf-effect-closure-20261016/RESULT.md)是当前事实入口；[运行历史](experiments/tf-effect-closure-20261016/RUNBOOK.md)只负责追溯。原固定96目前**72条完整机器人终态、24条未完成**。TF64结构/API可暂冻供系统研究，当前TF-on权重/配方不符合无新增风险失败要求，不能推荐独立TEST或正式性能。

|分层|T|TF-cont|TF-on|MLP-on|
|---|---|---|---|---|
|fresh init36/37，成功/完整|20/20|19/20|20/20|0完整，未测|
|known-risk，成功/完整|6/8|未登记|0/4（计划8）|未登记|

TF-on/T fresh共同成功20组原记录环境步数比P50/P90/max=.990745/1.020326/1.027027；这些不是严格质量证书。TF-cont失败task7/init37。不能以fresh20/20抵消风险反例，也不能在MLP0/20时宣布TF架构胜出。

风险task1/init29 T264成功→TF-on796失败；task3/init26 T240成功→TF-on796失败；task4/init26 T249成功→TF-on796失败；task5/init28双方796失败。前三对新增失败均有保存物理/controller/reset/真实CPU-CUDA RNG/双相机exact证据；前两对独立CPU组件审计、后两对新namespace逐局审计。未保存仿真内部UNK2026-10-17 08:43N，失败根因UNK2026-10-17 08:43N，不硬编码task回退。

原DEV 70局预算终态FAILED_OR_PARTIAL，缺旧TF-on局部arm cleanup，22完整成败保留为严格审计缺口；全局退出证明不能补造局部收据。一次pending26接续仅完成task4/task5两局后用户暂停；下一case只有client启动元数据、没有完整终态/COUNTER，已清理，不算算法失败。双根CPU报告72 raw/50严格组件完整/22原缺口；暂停非预算completion另列issue，整体仍PARTIAL/NO_SELECTION，不改冻结汇总器放行。

剩24：TF-on风险6/27、8/25、0/28、6/25及MLP-on fresh20。两个未闭合attempt实际forward UNK2026-10-17 08:43N，各按原4000上界保守扣账；原280k/3GiB不重置。新job60个观察到的自有pins全退出，GPU compute列表已空；没有AutoDL连接或新实例。全轮已准入GPU作业墙钟合计10.4647h（含失败/setup/等待/cleanup，非CUDA active），精确账本见结果索引。

## 这轮实际完成了什么

1. CPU监督/覆盖诊断57.14s：当前到达parent误差较同step target FIT高约48%，联合覆盖20.52%超过旧CAL对应P95；与旧mixed误差接近，不能称严重分布失控或根因。
2. collector资格r1/r2失败保留；r3原完整OFF＋新ON50chunk/2712行审计完成，未声称原双新pass/所有中间KV合同。一次30完整采集9383.734s、23685 forward、额外teacher0；生成池29成功/1失败不是新TF-on质量。pack49.37s：FIT11416/CAL6805实际到达行，11170反事实留原档。
3. TF-cont/TF-on/MLP-on固定500更新完成。原CPU1765.165s FAILED保留；CPU1完整校验续接1061.888s PASS。CONT复用已完成权重不重训；ON/MLP首次各500。全旧20237＋新6805 CAL CPU核验完成；TF-on两次完整27042行GPU接口核验通过，MLP生产GPU CAL未运行。
4. 新CAL CPU1 scaled-L2均值TF-on .003567824/CONT .003734550，低4.46%，逐query通过90.389%/90.066%；没有推导出闭环风险消除。一次新数据/公平训练机会已用完，当前暂停不再追加。

## 固定接口、训练和权限

TF64 53070参数：16token×17维，state7＋previous direction7＋σ＋nextσ＋t/1000；1层pre-norm、D64/H4/FF256、GELU/drop0、学习位置、双7维残差头，无显式视觉/prompt cross-attn；previous方向含隐式条件。MLP96为52928参数。draft内部FP32，adapter返回current dtype；生产BF16方向/FP32打分。

旧target FIT174=159691行、CAL20237；旧mixed14997=3906 A_off＋11091历史TF，不是当前on-policy。新30由旧TF_CONT3000生成，不是新TF-on自身on-policy。child2500同起点/seed20260923，各500更新、b512(256旧target＋256新FIT或历史mixed)、lr3e-5/AdamW/wd.01、norm/loss固定。TF-on SHA be2606d1…cb302；MLP-on f8d6a9e4…bf0d25，权重不公开上传。

默认旧**A_off是learned draft变体，不是原始target-off模式**；current_speculative_candidate.json不改。K2/B2bonus，7状态×CFG2=B14，每attention实例Q16；prefix_then_cost/shared-prefix/O1/O2/O3；τ=.007839680526450805，PF关闭，bonus不同于PF。共同reset512-token BF16 bank；新投影缓存/调度未集成本轮质量路径。

**四权限始终false**：engineering_quality_screen、independent_test_complete、noninferiority_established、formal_performance_permission。120/独立TEST/新TF正式45chunk未启动。旧retention/FAILED/阈值不变，ΣT_round/Σg和新TF正式E2E加速UNK2026-10-17 08:43N。新网页建议不是执行授权。

## 系统主线与历史指针

主线：投机改变工作负载→多候选并行性→新调度超过最佳B14→质量约束下有效提交与完整任务成本降低。H1有真实shape、H2有批处理经济性、H3只有有限pilot、H4未成立。n_v7不等于g7，K2bonus最多3合法更新；target调用减少不等于E2E。

WAM30层D3072/H24×128/FF14336；Linear M32/96/224库已合并，attention仍独立Q16。七冻结query逐B2对B14约2.45/2.91/3.36×经济性，不是机器人加速。路线B G1→G2 L2请求−32.81%、DRAM仅−1.31%、occupancy降，不能写HBM减半；共同缓存静态pilot不替代新TF闭环收益。原tile只读，不自动恢复融合/30层大融合。

旧[50局结果](research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)、[M11](../review/OMISSIONS-20261017.md)、[M12同query/全KV](experiments/draft-teacher-path-20261016/RESULT.md)均保留；M12首差在时间MLP，单项规范时间未恢复最终方向，不能据单query解释整局。旧child50 T48/TF44、max长度比1.4；fresh30任务/像素FAILED等不混新DEV/TEST。旧A_off45chunk1.60/1.54/1.50×及四任务1.15×不代替新TF速度。

公开历史固定WAM-egale@b0dfac3a；用户现授权新增paused72审阅分支，发布状态与新版commit以会话及新发布回执为准，不以main最新覆盖历史。原始PT/NPY/相机/KV不发布。网页版请优先读新版RESULTS/START_HERE/Prompt，旧“正在运行”按时间追溯。

## 下一步与日期

先完成文献/官方源码复核并回传一条有限、可证伪主线。没有明确恢复指令不启动新实验；若重开质量研究须新前瞻协议，不能清旧STOP复用已结束job。TF不是必须优于MLP的预设，架构偏好不放宽质量门。算子正式性能仍须本机SXM4独占且先质量放行；AutoDL双卡仅独立episode吞吐用途，实测加速UNK2026-10-17 08:43N。

10/18作者OpenReview profile检查；10/22质量/系统go-no-go；10/23冻结只在证据允许时；10/28主表；**11/01 18:00中文稿/证据给老师**。10/17再核[CVPR官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)：北京时间注册11/11、正文11/17、补充11/24各19:59；未另列abstract，AuthorGuidelines链接404，未知规则不套往年。

HW固定v15/legacy，不升级/重置/旧全量sync。state/cycle/rules、结构化规则、原tile/HW与失败证据保护。中文论文在对话讨论、用户整理飞书，不新建本地论文稿。
