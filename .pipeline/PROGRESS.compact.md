# 当前研究恢复入口

更新时间：2026-10-16 15:16，北京时间。派生短视图，原PROGRESS/保护状态/FAILED/tile保留。先读[短路线](research/speculative-sampling/WAM-COORDINATION-20261016.md)，按需读[今天50完整结果](research/speculative-sampling/DRAFT-FREEZE-RESULT-20261016.md)、[晨间80](research/speculative-sampling/TF-USABILITY-RESULT-20261016.md)或[夜间系统](research/speculative-sampling/OVERNIGHT-RESULT-20261016.md)。

## 当前决定与资源

今天50完整DEV已收尾，**TF retention HOLD**。TF64结构及B14接口暂冻作算子研究，3000权重/训练配方未accepted。明天120局入口SOURCE_READY但真实前置拒绝，未注册、无CPU waiter/GPU分片。AutoDL已归还，不再SSH，实例关闭未知。今天58个已观察自有pins均退出；8056.511秒、2.207GB、61859前向计数，4h/3GiB/100000内；不是正式性能。

用户资源偏好：先质量验证，需要算子正式计时时通知在完整任务分片末暂停。已备入口每片30完整episode，非立即暂停。当前没有获准启动的扩大链。默认旧A_off不变，TF正式45chunk受阻；工程筛选、独立TEST完成、非劣建立、正式性能许可四项均false。

## 本轮已观察

fresh init35：T/TF_CONT/等预算MLP_CONT各10/10。TF/T步数比P50/P90/max=1.021212/1.074884/1.092593，三长度条件过；全任务失败800均值257.5/265/267.7。风险8组T/TF均5/8，配对00/01/10/11=2/1/1/4：task4/init26 T249成功→TF796失败，触发risk_no_new_TF_failure=false；task0/init28反向救回不能抵消事前否决。风险共同成功max1.322449仅描述，不新增判据。

3184PT/34776数组、50完整commands与原RESULT全量独审一致，173源/两小权重/prereg/491保护记录不变。28对保存物理/controller/实际初始target Torch RNG一致，相机25/28；三处均task8腕相机，幅度≤2。否决task4两相机也exact，不能归因初始像素差。未保存仿真内部仍UNKNOWN。旧task7/init24不在本轮，不能称回归。

target重复4局独列：task2/init33成功258/263步，task8/init33成功536/382步；命令不同、腕相机1/2pixel不同。重复性与像素共现不证明根因，不把4局算4独立初态。

task4有限CPU诊断：首环境command索引0差1通道/max.0078125，保存actions_0差62个BF16元素/max.015625；第一份保存obs最早帧3不同。失败前14完整chunk/252记录与同task成功init35深度和sigma分布相近；changed_route日志0不代表packed/sequential重验。缺teacher velocity到环境命令映射；D01/阈值/PF/cross-attn均HOLD。

下一步先登记2–4对完整风险诊断，保存已算teacher输出与提交/命令映射，不加teacher调用、不存逐query30层KV。运行粗估20–40分钟，入口实现/审查另0.5–1.5h；尚未登记/启动。仅证据触发一次独立FIT/CAL数据与等预算训练，不能按失败task硬编码回退或改旧RESULT放行。

## 固定研究接口与训练

TF53070参数，16×17：state7、previous-direction7（根已有方向、递归父draft）、sigma、next-sigma、t/1000；1层pre-norm、D64/H4/FF256、GELU/drop0、学习位置、两7维残差头，无cross-attn。MLP52928等容量对照。小模型内部FP32，adapter随current_state dtype返回，本轮BF16；接受分数FP32。

seed20260923/final3000：FIT174 target159691行2000更新/b512/lr1e-4；再500＋500，各新AdamW lr3e-5/wd.01、256target＋256历史candidate，固定norm，最后500重复原抽样表；不是连续3000。TF CPU125.154秒，新MLP54.622秒。旧14997池=3906 A_off＋11091历史TF，不是当前TF-on-policy。两新模型完整20237实际GPU CAL同设备逐位/RNG核验均过，不代替闭环。

TF SHA3657388f…d64ac、MLP3dc01a48…103b0，研究接口见[STRUCTURE-FIXED](../投机采样/results/runtime/p0-draft-freeze-v1-20261016-r1/OPERATOR-INTERFACE.STRUCTURE-FIXED.json)。K2/B2bonus、7状态×CFG2=B14、Q16/状态、prefix_then_cost、shared-prefix/O3、τ=.007839680526450805；PF关闭，bonus不同于PF。共同reset512-tokenBF16bank，本轮未集成新调度/文本投影缓存。

## 系统因果主线

投机重组连续动作负载→多候选可调度→新增调度超过最佳B14→质量约束下单位有效提交与任务成本下降。H1实际shape，H2批处理经济性，H3有限新增pilot，H4未收口。普通batching/Graph/共同缓存不算新增机制。

Linear M32/96/224库已合并，D3072/H24×128/FF14336/30层；attention实例Q16。七冻结query逐个B2与B14约2.45/2.91/3.36×经济性，不是机器人加速；n_v7不等于g7，K2bonus最多3更新。

路线B guarded_v2不增加BN64块数才G2，否则G1；旧G4长/G2中数值FAILED保留。未缓存两进程全档−4.98%，L2请求−32.81%、DRAM仅−1.31%；六A_off完整chunk/300提交状态通过。共同文本投影缓存单进程36行静态pilot全档−10.27%，payload2.859GiB，未独立复测/动态生命周期/cached完整chunk。旧A_off45chunk1.60/1.54/1.50×及四任务1.15×不代替新TF证据。

明天算子可用新命名空间固定结构/B14接口推进最佳缓存基线复测、生命周期、短中长真实KV及完整chunk状态/ΣT/Σg；原tile和撤回融合不改，不先做大融合。

## 历史与日历

晨间80：T19/20、child19/20、CONT20/20、旧MLP20/20，主长度1.004184/1.032907/1.172996；不是今天等更新MLP对照。60物理/RNG一致、像素56/60，原远端receipt命名触发FAILED、原聚合PARTIAL保留。旧本机13不混池；T task8旧393成功→796失败，原因UNKNOWN。晨间3.47674h含等待/失败，两个本机目录1.923GB，AutoDL小档约65MB，未镜像全部obs。

旧child50 T48/TF44、−8pp、共同成功max1.4；原fresh30 T10/TF和MLP9、task7失败和task8像素FAILED；旧T/A_off100严格FAILED和描述T48/A47全部保留，不借新结果覆盖。

默认关键路径：风险诊断→条件性一次数据/训练→有限DEV→10/22 go/no-go→10/23候选/引擎冻结→10/26 TEST→10/27获准计时→10/28主表→11/01 18:00中文稿交老师。最迟节点不满足则收缩可选范围，不扩种子/树/环境。TEST40–49未跑/未读结果，仅核初态容器数；上游target训练来源UNKNOWN。350局计划未启动，无新增采购授权。

10/16再核CVPR官网：北京时间论文注册11/11、正文11/17、补充11/24各19:59，未另列abstract；10/18查作者profile、10/25处理。[官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)。HW既有v15/legacy保留，本轮不升级/重置；state/cycle/rules、原大PROGRESS、tile/FAILED不动。中文论文只在对话由用户整理飞书。
