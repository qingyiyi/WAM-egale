# 固定PF2的接受器比较与CVPR2027计划

状态：本轮隔离实现、开发集校准、20个真实GPU完整chunk及16个完整任务、独立审计均完成；本轮新候选质量门结果见下表。日期：2026-10-14，Asia/Shanghai。

## 研究问题与固定条件

用户授权继续比较当前步长加权L2（WL2）、FlowCast风格速度MSE（MSE）和原scheduler实际后继状态的归一化残差（RES）。冻结A_off双头MLP、target、PF2=spec/spec/target、bonus、O1/O2/O3、video20/action50、完整800 horizon及环境执行协议。只改变接受规则与开发集校准出的对应阈值，不同时训练或调整周期。

PF2表示两个动作chunk开启投机，再让第三个动作chunk完成原大模型的50次去噪更新。bonus表示一个投机tree接受到完整两步前缀后，使用target方向再推进一个去噪更新；它属于chunk内部，和PF2周期分别记录、分别冻结。

MSE仍用当前学习型双头draft，是接受器适配，不是完整FlowCast恒速draft算法复现。RES调用原scheduler/scatter/BF16/projection得到同parent的target后继，再检查候选后继；使用DPM-Solver的max绝对/相对尺度归一化，是数值残差适配，不是高阶嵌入求解器。相关原文依据见[文献核查](ACCEPTANCE-CRITERIA-CCFA-20261014.md)。

\[
e_i=|\Delta\sigma_i|\|v_D-v_T\|_2,\quad
m_i=\operatorname{mean}((v_D-v_T)^2),\quad
R_i=\sqrt{\operatorname{mean}\left[\left(\frac{x'_D-x'_T}{\max(0.0078,0.05\max(|x'_T|,|x|))}\right)^2\right]}.
\]

每个规则用一个冻结全局阈值。RES的三次target状态探测没有额外transformer前向、不提交动作；相应scheduler工作需要记入成本，不能把它隐藏成CPU计算。

文献支持上述判据形式。RES的绝对/相对尺度0.0078/0.05、完整任务总步数5%与单对变长10%的门，是本项目在本轮结果出现前声明的实验设置；不是论文给出的LIBERO成功率保证。接受阈值只在train标签上配平，闭环质量结论仍来自后面的真实配对任务。

三投机臂均沿用既有最长合法前缀选择器，同长前缀按各自规则的接受边分数排序。因此更改scorer同时影响通过身份与分支排序；本轮是整体接受规则对照，没有单独隔离排序变化的因果贡献。

sigma（σ）是每个动作chunk内部从噪声走向最终动作的去噪进度：从1附近开始，降到0结束；下一个chunk重新开始。这里的velocity表示latent随sigma的变化，不是机器人在环境中的物理速度。50个action去噪更新与实际执行的环境步数是两种计数。步长Δσ不均匀，因此相同方向误差在较大步长下会产生更大的状态偏差。

[验证集sigma图](../../../review/OMISSIONS.md)统计阈值冻结后旧策略访问状态的3780个head标签。低sigma区间WL2/MSE/RES分别通过2725/2134/2732个（共3128）；中区间25/581/27个（共584）；高区间0/68/0个（共68）。它说明MSE把接受位置大量移向中、高sigma；不能据此判定任务是否成功、trace是否接近或是否提速。两个head来自相关状态，不能当3780个独立任务。

## 已完成的CPU实测

只用已有teacher train task0/1×init0/1的3906条same-query行、7812个head标签，将MSE/RES通过数量配平到WL2的5812。整episode val8的1890行在阈值落盘后计算，不用于调参。生产逐行predict，state/previous/target回到实际BF16；RES用原FlowMatchScheduler并保留frame0投影。旧数据是OLD访问状态，不能称新候选的on-policy数据。

| 规则 | 固定阈值 | train head通过 | val head通过 | val旧通过丢失／新增 |
|---|---:|---:|---:|---:|
| WL2 | 0.007839680526450805 | 5812/7812 | 2750/3780 | 0/0 |
| MSE | 0.013428529258817434 | 5812/7812 | 2783/3780 | 934/967 |
| RES | 0.06256367266178131 | 5812/7812 | 2759/3780 | 37/46 |

配平数量不保留身份。train中MSE交换1852条旧通过标签、RES交换91条；离线总体通过率提高不能据此推断trace变短或成功率提高。完整11592条分数与身份、1932份teacher来源SHA、阈值区间/ties见[校准报告](../../../review/OMISSIONS.md)。

三个scorer的78项CPU检查通过。独立集成审查发现绑定API实例进入全局cache，可能使MSE继续使用旧WL2；修复为只缓存unbound代码、每请求创建独立API后42项检查通过，旧37PASS/2FAIL证据保留。审查验证六边索引、BF16输入、真实3次scheduler探测、输入/树/RNG不变及回调隔离。

24个tiny30层CPU完整chunk控制通过，包括MSE/RES当前O3路径重复。RUN1的CPU native SDPA与O3 float64参照差异保留：video相同、MSE action最大差0.0234375、路径/KV不同。这仅是CPU后端诊断；不能写成GPU数值通过。真实GPU仍要求同规则RAW/optimized完整输出、route、KV、RNG、phase和forward精确相同。真实GPU门20chunk，不因CPU增补改变GPU预约。

最终收尾日志增补了实际scheduler probe计数：用已有完整phase计数减去树展开、普通target和bonus，核对RES恰好每tree3次、其他规则0次。24个已闭合CPU chunk的实际新metadata代码回放与负例通过；数值函数无改动，没有重复完整chunk。原CPU报告与metadata绑定分别保留。[CPU控制](../../../review/OMISSIONS.md)、[绑定审计](../../../review/OMISSIONS.md)。

## 固定完整任务实验

T/PF2_WL2/PF2_MSE/PF2_RES，各task0/1×init15/16，16个任务，双独立EGL。15/16曾出现在历史target-only，未用于当前teacher/draft拟合；只称冻结接受器后的初态holdout开发pilot，不称全局未见test或总体非劣证明。

预声明对T和PF2_WL2分别检查：参考成功→候选失败=0、成功数不降、总惩罚步数比≤1.05、最差配对变长比≤1.10；失败按800计，最大绝对变化另报。候选阈值不根据这些结果重新选择，不追加或删掉不利任务。新GPU硬限6300秒，累计36000秒；carry25684.616692936048秒/108894 target intents，新有限预约65600 intents。共享GPU只采质量，不把墙钟作为新加速结果。

完整16任务自然终止，计数、初态/图像/RNG与PF2周期由两份独立分析读回。20个真实GPU完整chunk的同规则RAW/O3输出、route、RNG、phase/forward及现场完整KV严格对照通过；离线保存的是动作/视频和KV轻量metadata，不声称离线重新逐字节校验了全部KV。研究执行与审计通过，和新候选通过任务质量门是两种结果。

| task/init | T | PF2_WL2 | PF2_MSE | PF2_RES |
|---|---:|---:|---:|---:|
| 0/15 | 成功/320 | 成功/288 | 成功/317 | 成功/324 |
| 0/16 | 失败/796 | 成功/271 | 失败/796 | 成功/301 |
| 1/15 | 成功/257 | 成功/252 | 成功/254 | 成功/309 |
| 1/16 | 成功/263 | 成功/333 | 成功/284 | 成功/273 |

每格为自然结果/实际执行命令数；失败按800计入以下trace门。客户端另有原协议的5个稳定动作，完整horizon按环境timestep判定，因此失败的实际命令数可能小于800。

| 配置 | 成功 | 实际环境命令总数 | 惩罚步数总和 | policy chunks | target Transformer调用 |
|---|---:|---:|---:|---:|---:|
| T | 3/4 | 1636 | 1640 | 105 | 7764 |
| PF2_WL2 | 4/4 | 1144 | 1144 | 75 | 4057 |
| PF2_MSE | 3/4 | 1651 | 1655 | 106 | 5929 |
| PF2_RES | 4/4 | 1207 | 1207 | 79 | 4302 |

target调用来自intent/返回闭合计数，包含实际模型工作；次数下降本身不是速度提升。共享GPU墙钟仅作资源与诊断记录。

| 候选 | 参照 | 成功丢失对数 | 总惩罚步数比 | 最差配对变长比 | 最大绝对变化 | 预声明质量门 |
|---|---|---:|---:|---:|---:|---|
| PF2_WL2 | T | 0 | 0.697561 | 1.266160 | 66.125% | 未通过 |
| PF2_MSE | T | 0 | 1.009146 | 1.079848 | 7.985% | 通过 |
| PF2_MSE | PF2_WL2 | 1 | 1.446678 | 2.952030 | 195.203% | 未通过 |
| PF2_RES | T | 0 | 0.735976 | 1.202335 | 62.375% | 未通过 |
| PF2_RES | PF2_WL2 | 0 | 1.055070 | 1.226190 | 22.619% | 未通过 |

MSE和RES均未通过双参照开发pilot门，本轮不晋级。PF2_WL2相对T也未通过本轮门，先处理质量稳定性，再做verifier提速验收。原preset仍保留；完整条件仅4个，不能推出全LIBERO成功率或trace保真保证。失败和超出门槛的条件全部保留，没有根据结果调阈值或加局。

[主分析原始结果](../../../review/OMISSIONS.md)、[独立原始数据审计](../../../review/OMISSIONS.md)、[配对质量图](../../../review/OMISSIONS.md)。独立审计初版将保守收费policy标志误读为当前有未闭合预约，错误断言与初版失败保留；资源补充逐一验证17份预约实际全部闭合、1018+7764+4057+5929+4302=23070次实际调用，不改变质量门或费用。

本轮新GPU计费4393.431秒（1.2204小时）；累计30078.048秒（8.3550小时），10小时预算剩余5921.952秒（1.6450小时）。累计target intents=131964，本轮模型启动17次；任务自有GPU进程已退出。预算与资源闭合以[GPU-COLLECTION.json](../../../review/OMISSIONS.md)为准。

## 仿真瓶颈与时间安排

对上一轮12个完整任务的客户端elapsed范围复算，非模型RPC总剩余228.493/3611.428秒，约6.327%；这是包含仿真、处理与日志的上界，未单独测得纯MuJoCo成本。WAM服务reset RPC总1502.961秒，平均约125秒，包含prompt编码、Transformer/VAE cache与mask等准备；没有各子项独立计时。它不属于Flash物理仿真可直接替换的部分，0次target Transformer前向也不等于没有GPU工作。构造/close另外约6.6秒/episode，不在上述elapsed内。每次env.step的相机图像均用于cache，保持原20Hz、物理步、相机输入、稳定动作和终止行为，不能通过跳渲染或减物理步取得等价提速。[证据](../../../review/OMISSIONS.md)。

CVPR2027官方全文11/16 AoE；内部老师稿硬节点为北京时间11/01 18:00，内部全文提交目标11/15。用户的MegaTransformer是自行设计、借鉴MegaMoE思路的verifier系统优化。本轮仅安排，不计未实现kernel收益。10/17冻结接受器候选、10/21质量表、10/26原型正确性与E2E、10/30主证据；写作从现在并行。[完整倒排](CVPR2027-SCHEDULE.md)。

## Flash-LIBERO能否替代当前仿真

已核对[Flash-LIBERO](https://github.com/Catigeart/Flash-LIBERO/tree/4bcc9dad5b0eda384dea66379e2bfb01b32e4dc7)固定提交4bcc9dad5b0eda384dea66379e2bfb01b32e4dc7。它将物理计算迁至CUDA FP64，使用cuBLASDx/cuSOLVERDx；保留LIBERO的任务、常用Python入口和OpenGL/EGL图像链路。它是加速后端候选，作者尚未提供可核查的整应用加速倍数或与原版配对成功率。我们本轮只审查源码，没有运行Flash硬件测试。

静态逐字节核对130个BDDL以及LIBERO-10全部20个原始/裁剪初态资源git blob，均匹配当前原版提交。代码保留20Hz控制、25个物理子步、稀疏success/done语义与128RGB输入；当前客户端的图像翻转仍需保留。这些静态对应不证明长时动力学、像素或成功率等价。[完整审查及来源](../../../review/OMISSIONS.md)。

目前存在具体接入阻断：`acceptance_criteria_v1/client_entry.py`的状态记录读取`sim.data._data`，Flash GPU facade对这个私有CPU句柄直接抛`UnsupportedGpuOperation`。因此不能把正式采集环境的`libero`包直接替换就开始出论文主表。仓库[测试说明](https://github.com/Catigeart/Flash-LIBERO/blob/4bcc9dad5b0eda384dea66379e2bfb01b32e4dc7/tests/README.md)明确不执行真实GPU物理、OpenGL或性能基准。

使用建议：原版继续作为主评估参照；Flash放独立环境，先建立公开字段的状态记录适配，固定资产、init、RNG、20Hz/子步/稳定动作、相机及成功判定。依次比较原CPU/FlashCPU/FlashGPU接口与状态、相同动作序列回放、同一冻结策略的配对完整episode，再测冷启动、稳定吞吐和包含WAM的E2E。通过相应验证后可以用于采样；在物理等价尚未确认时，成功率需明确标注Flash后端，并保留原版复核。当前6.327%的非RPC时间上界不足以推断Flash会大幅改善完整任务时延；同卡GPU物理与verifier资源竞争也需实测。

## 证据与当前边界

隔离实现：`投机采样/experiments/acceptance_criteria_v1`；[工作清单](../../../review/OMISSIONS.md)。原公共代码、模型、preset、protected HW和旧证据保留。本轮采集和审查已完成。Flash仅完成静态审查；MegaTransformer当前为倒排规划，尚无实现或性能结果。下一步先据真实质量门冻结研究候选，区分开发选择与独立验收，再profile原verifier并限定一个优化原型。扩大质量样本或运行Flash硬件验证须先估算剩余GPU预算，不能把本轮pilot当论文全benchmark结论。
