# FastWAM迁移夜间完整简报 · 2026-10-27

覆盖10月26日晚至10月27日上午已收口实验；补充算子会话10:10完成的Cross-KV结果。此页是公开快照，不是实时服务器。先固定Git commit，再读[网页版讨论Prompt](WEB-DISCUSSION-PROMPT.md)、[版本及逐文件SHA](MANIFEST.json)、[数据说明](DATA-README.md)和[源码导航](CODE-MAP.md)。本次不生成论文正文。

**结论：十步FastWAM上的真实投机路径已经跑通，并取得动作去噪的实际收益；完整策略收益较小，闭环质量和研究增量尚未证明充分。** LIBERO四suite、40任务、每任务50初态的三臂全量共6,000局已完成，target/TF/TF+PF分别97.50%/97.25%/97.20%。同条件prompt缓存的5090结果，TF动作去噪1.150–1.242×，完整policy chunk 1.057–1.080×。RoboTwin首四canonical完整成功，尚无Robo投机质量或速度结果。

完整面板含本次FIT/CAL/DEV及历史暴露，不能称独立TEST。四项`engineering_quality_screen`、`independent_test_complete`、`noninferiority_established`、`formal_performance_permission`均false；扩量工程检查通过不等于这些科学结论成立。

## 1. 已完成的迁移及固定身份

|项目|本次实际采用|
|---|---|
|官方代码|FastWAM `7faa71108368fbb3b6885649f112af607427a2d4`，工作区核对为clean|
|官方权重revision|`8eaceeb24c3cc92ff2a9c9a9d266a4941b836705`|
|LIBERO|Base `libero_uncond_2cam224.pt` + 同revision统计量；不是Optional-IDM/Joint/IDM|
|RoboTwin|另一份Base `robotwin_uncond_3cam_384.pt` + 对应统计量；不能共用LIBERO动作头或normalizer|
|历史方法参考|WAM `89ba5882a7e7185ff6178541782bfb70731603bf`；旧LingBot代码/权重/失败/结果保留|
|推理协议|双方10次scheduler更新、sigma shift 5；同seed42每次重新初始化噪声；BF16 target，FP32打分|
|视觉协议|当前图像VAE编码、视频分支一次prefill建立视觉KV；不生成未来视频、不跑视频多轮去噪|
|动作协议|LIBERO预测32×7、每次最多执行10条；Robo预测32×14、最多执行24条；去噪步、动作命令、chunk是不同单位|
|网络/条件|30层dense action、hidden1024、attention宽3072、FFN4096；128文本位置+1proprio位置；单路条件|
|缓存/公平基线|官方视觉KV与torch.compile保留；文本缓存独立版本，三臂共同使用；当前投机默认未接persistent或Cross-KV候选|
|资产|LIBERO与Robo各自权重/统计量均在本机、双A100、5090三处保留；不上传GitHub|

已迁移canonical调用、真实teacher采集、轻量预测器训练、B7批验证、真实接受/拒绝/fallback/bonus、闭环评估及5090成本计量。七个verifier状态为根1+第一层2+第二层4，新候选只有6个；各自32-token动作序列隔离，不能无掩码合成224-token互相注意。首步target bootstrap、末尾和bonus全部计费，每chunk实际提交G=10；目前剩余不足3步仍走canonical，尚未应用B3尾部改进。

部署模型和数据契约见[初始固定manifest](../../.pipeline/experiments/fastwam-base-small-closedloop-20261026/MANIFEST-v2.json)、[接口映射](../../.pipeline/experiments/fastwam-base-small-closedloop-20261026/INTERFACE-MAPPING.md)。这些是按创建时间冻结的历史记录，其中“尚未测速/暂不PF”等状态由本页及后续结果覆盖，不能当最新状态。模型并非迁入LingBot的CFG双路、full512历史bank或末端t=0额外提交。

## 2. 采集、训练与新阈值

每suite独立100 FIT episode、10 CAL episode、20组DEV三臂：FIT初态0–9、CAL初态10、DEV初态47/48，按完整episode分组；四套共400 FIT、40 CAL、80组三臂DEV。全量0–49随后覆盖这些已暴露初态。没有随机拆query行，也没有将已用过的面板改名TEST。

教师冻结；标签是真实target在canonical访问状态、对应timestep和条件下的去噪方向。当前未补采draft rollout分布。小预测器54,094参数，32个动作token、width64/heads4/FFN256、双头残差方向；显式输入action state、previous target/已提交方向及时间，完整文本/图像/proprio仍由target处理。不是训练或替换target backbone。

FIT-only输入/残差标准化；任务→episode→query分层采样，失败采集episode不丢弃。FP32 AdamW，固定final3000 checkpoint；loss为双头方向MSE的min项 + 0.1×双头平均项 + 0.01×多样性hinge项，详见[训练代码](../../投机采样/fastwam-base/fullcode-v1/train.py)。CAL上取两头最小`abs(actual delta_sigma) × L2(v_draft−v_target)`的Q80；去掉无previous的首步，阈值使用索引1–8的可投机位置。BF16 proposal转换后、归一化动作空间所有32×7通道FP32评分。阈值没有按全量成败逐task调节，也没有强制线上80%接受。

|suite|FIT方向query|CAL用于阈值query|新tau|全量TF逐步接受率|TF深度0 / 1 / 2|
|---|---:|---:|---:|---:|---|
|Spatial|9,576|824|0.0151355192|79.953%|22.06% / 13.61% / 64.33%|
|Object|11,943|1,016|0.0137320096|80.409%|19.86% / 15.43% / 64.71%|
|Goal|10,008|824|0.0145392660|79.905%|23.11% / 12.43% / 64.46%|
|Long|23,787|2,296|0.0154123530|80.443%|14.65% / 21.60% / 63.75%|

总计55,314 FIT query、4,960个用于阈值的CAL score。CAL总保存query与用于阈值的query不同，见[采集汇总](data/teacher-collection-summary.json)。逐步接受分母为实际到达的路径验收边，不是所有候选边，也不是“80%整树通过”。各checkpoint SHA/选择规则见[四套选择记录](data/draft-selection.json)。TF+PF按完整policy chunk执行S,S,T循环，不能恢复已经偏离的机器人状态；也不是每树的bonus。

## 3. 6,000局闭环质量和trace

|suite|target|TF|TF+PF|全体平均实际动作命令 T / TF / PF|
|---|---:|---:|---:|---|
|Spatial|481/500，96.2%|481/500，96.2%|482/500，96.4%|110.622 / 109.960 / 110.800|
|Object|499/500，99.8%|497/500，99.4%|498/500，99.6%|126.958 / 128.356 / 127.684|
|Goal|493/500，98.6%|490/500，98.0%|491/500，98.2%|108.772 / 109.722 / 109.534|
|Long|477/500，95.4%|477/500，95.4%|473/500，94.6%|265.976 / 265.660 / 269.820|
|合计|1950/2000，97.50%|1945/2000，97.25%|1944/2000，97.20%|153.082 / 153.4245 / 154.4595|

每suite是10个任务；每臂500局，共4×500×3。全部完整有效，缺失/PARTIAL/无效0；失败分别50/55/56，保留完整horizon与真实执行命令。计数、G=10、实际动作前缀、身份配对和训练后B7五项审计通过。总体点估计TF比target低0.25个百分点，PF低0.30个百分点，不能因此直接宣称非劣。

|suite / 臂|双方共同成功N|共同成功平均命令 T→候选|>1.3倍trace数 / 最大比|救回target失败 / 新增失败|
|---|---:|---|---|---|
|Spatial / TF|475|99.02→98.55|4 / 1.697×|6 / 6|
|Spatial / PF|474|99.08→99.33|2 / 4.645×|8 / 7|
|Object / TF|497|126.29→126.72|2 / 1.460×|0 / 2|
|Object / PF|498|126.42→126.59|0 / 1.260×|0 / 1|
|Goal / TF|490|104.35→103.80|2 / 1.863×|0 / 3|
|Goal / PF|490|104.57→104.13|4 / 1.728×|1 / 3|
|Long / TF|468|244.07→243.85|4 / 1.749×|9 / 9|
|Long / PF|464|242.15→243.50|5 / 2.221×|9 / 13|

多数成功任务长度接近canonical，长度比中位数均约1，P90约1.014–1.021；但均值接近不能保证每项任务trace不变。PF没有跨suite稳定改善。配对SR差的95% bootstrap区间（百分点，TF / PF）为Spatial[-1.4,+1.4]/[-1.2,+1.6]，Object[-1.0,0]/[-0.6,0]，Goal[-1.4,0]/[-1.2,+0.4]，Long[-1.6,+1.6]/[-2.6,+1.0]；这是暴露面板的描述性不确定性，不是预注册非劣检验。

与论文报告canonical点估计比较：Spatial 96.2% vs 98.2%，Object 99.8% vs 100%，Goal 98.6% vs 97.0%，Long 95.4% vs 95.2%。Spatial低2个百分点仍需定位；不能称已复现论文完整成功率。Object初始像素500组均一致，其他suite存在渲染像素不完全一致；物理初态/seed配对与像素exact需分开，不能把新增失败全解释为渲染差异。

数据：[6,000逐局CSV](data/episodes-6000.csv)、[120逐任务/臂CSV](data/tasks-120.csv)、[4,000配对CSV](data/paired-quality-4000.csv)、[原始审计与逐suite区间入口](../../.pipeline/experiments/fastwam-full-triarm-20261027/FOUR-SUITES-RESULT-v1.md)。CSV从原JSON派生，SHA保留；未上传原相机/动作/KV张量，因而本公开包可重算标量结果，不能独立重放完整机器人实验。

## 4. RTX5090速度：共同成功条件、真实十步

以下为三臂共同prompt缓存版，与上面原未缓存的6,000局质量有独立版本身份。速度只用原DEV三臂共同成功来源组，不借后续全量重跑的成功替换原失败。Long/Spatial/Object/Goal分别16/19/20/20组，每组3个观测、5次重复，共75组、225观测、1,125对主计时。所有慢调用保留，GPU按登记独占。

|suite|动作 T / TF ms|动作加速|完整chunk T / TF ms|chunk加速（条件性95%区间）|
|---|---:|---:|---:|---|
|Long|60.952 / 49.064|1.242×|158.212 / 146.473|1.080× [1.074,1.086]|
|Spatial|61.504 / 51.055|1.205×|163.467 / 153.100|1.068× [1.061,1.074]|
|Object|61.300 / 49.878|1.229×|164.682 / 153.729|1.071× [1.065,1.078]|
|Goal|61.344 / 53.366|1.150×|164.297 / 155.500|1.057× [1.048,1.064]|

完整policy调用即一个chunk，包括图像/文本/状态条件、VAE、视觉prefill、动作十步及后处理；一个任务包含多个chunk。T_denoise包含draft、pack、verify、accept/commit、bootstrap、fallback、bonus和尾部。按`G=sum(g)`、`C_update=T_denoise/G`统计；四套T约6.10–6.15ms/实际更新，TF约4.91–5.34ms/实际更新。不是把树的6次展开计成6次推进，也不是以减少总去噪步数制造收益。

|suite|任务trace共同成功配对N|同长度policy回放 TF / PF|各臂自然长度policy回放 TF / PF|
|---|---:|---:|---:|
|Long|3|1.081× / 1.054×|1.078× / 1.056×|
|Spatial|3|1.067× / 1.042×|1.040× / 1.029×|
|Object|4|1.071× / 1.045×|1.095× / 1.071×|
|Goal|4|1.059× / 1.015×|1.020× / 1.013×|

**任务trace数字是保存观测后的policy回放，不含仿真步进；目前没有5090真实仿真环境E2E加速比。** 配对N仅3/3/4/4，是成功条件下的小样本结果。A100并行评估wall时间只作资源记录。

每臂另有75次同步profile：TF完整chunk耗时约38.74%在GPU VAE、35.12%在动作循环、11.38%视觉准备、7.93%视觉prefill；文本缓存约0.058ms/0.04%。VAE已经在GPU BF16，当前没有“把CPU VAE搬GPU”的增益。TF动作内部约57.74% verifier、18.33%尾部target、12.96%bootstrap、8.79%draft、1.78%accept/commit、0.22%pack。profile和主计时独立，嵌套分项不能重复相加。

![完整调用与动作耗时分布](../../.pipeline/experiments/fastwam-full-triarm-20261027/TEXT-CACHE-v1/REVIEW-4suites-v1/FIGURES/LATENCY-DISTRIBUTIONS.png)

![完整policy和动作循环分解](../../.pipeline/experiments/fastwam-full-triarm-20261027/TEXT-CACHE-v1/REVIEW-4suites-v1/FIGURES/PHASE-BREAKDOWN.png)

冷成本另列：各suite模型加载约40–43s，首次canonical compile调用约67s、首次speculative compile调用约76–78s；每个prompt首次填充及其他冷检查在原JSON记录，不包含在warm毫秒表。缓存容量64、只缓存T5文本输出，proprio每次重新追加。

225输入在同一加载模型中cached/uncached条件、T/TF动作、路径和计数全部exact。但跨独立作业，相同fixture的路径变化Long4/48、Spatial15/57、Object9/60、Goal23/60，原因未定；Goal最慢动作约139ms保留。另有Long canonical重复同观测/seed动作差异，及一例原失败→成功。**同进程缓存等价不等于跨进程可复现；不能把旧/新速度差全归因于prompt缓存，也不能给原6,000局贴上缓存版质量标签。**

来源：[缓存完整报告及原始JSONL](../../.pipeline/experiments/fastwam-full-triarm-20261027/TEXT-CACHE-v1/REVIEW-4suites-v1/REPORT.md)、[1,125对便携CSV](data/cached-speed-pairs-1125.csv)、[重复运行差异](../../.pipeline/experiments/fastwam-full-triarm-20261027/TEXT-CACHE-v1/CROSS-RUN-COMPARISON-v1.md)。原未缓存速度独立保留，其完整chunk约1.059–1.072×。

## 5. RoboTwin、失败与资源

首轮canonical采用独立14维Base/3相机/32预测24执行/十步shift5/官方eager，首四任务：adjust_bottle 118命令5chunk、beat_block_hammer 105/5、blocks_ranking_rgb 436/19、blocks_ranking_size 445/19，4/4完整成功。actual action prefix、网格、strict加载、三相机图像工程检查通过；图片末帧是动作前观测，成功以官方环境记录为准。

前三attempt分别被GL/EGL渲染依赖、ffmpeg入口阻塞，均0policy，记录保留；只修用户目录的既有依赖路径。adjust_bottle候选seed4300000一次expert setup异常保留，实际有效seed4300001；其他为4300000。成功attempt232.508秒，加前三失败及两次诊断共502.120秒，原5400秒预算未重置。

14维collector/predictor/verifier代码已准备并做CPU契约检查，但Robo尚未训练draft、校准tau、运行TF/P闭环或5090测速。四局pilot不能替代50任务×100有效初态×clean/randomized×三臂的30,000局面板。见[Robo审阅](../../.pipeline/experiments/fastwam-robotwin-canonical-20261027/CANONICAL-REVIEW-v1.md)和[四局CSV](data/robotwin-canonical-4.csv)。

三张A100按每卡一模型server、两隔离环境client推进LIBERO；全量12个作业登记生命周期合计约16.647 A100小时，**仅全量阶段，不含采集/训练/冷启动前置等总账**。逐job真实起止见[资源CSV](data/full-eval-resource-jobs.csv)。5090投机测速原版+缓存版累计3725.658/7200秒，均已退出并释放；不能和另会话算子预算混算。原36h/72 aggregate A100h和24h/card约束仍在，截止10/28 12:43；5090本轮原截止10/27 21:32。

本次发布前检查三张A100无模型进程；Cross-KV算子作业后来于10:10收口并报告5090自有进程全部退出。发布不启动新实验。其他夜间工程故障也保留：本机初载SIGKILL后以T5 meta/assign+mmap恢复；初版审计维度错误、控制器只读SSH中断、文本测速因协调文件身份变化而分版本接续，均未当作成功数据或重置预算。

## 6. 算子侧新增证据与下一步方向

算子由另一会话负责，本次只同步其已完成证据，未把算子候选合入主投机版本。严格TMA/1T4M3E/mbarrier完成eager数值对齐，但尚未胜过官方compile。FFN2连续FP32试验局部约1.060×、整层无净收益，72接受边1翻转/1路径变化，因此未采用。

10:10新完成Cross-KV候选：chunk内分别构建B1/B7条件投影和30层K/V，不能无证明跨chunk复用proprio或把B1广播成B7。六个代表层/shape及四个固定Long观测的eager动作、接受路径与teacher比较exact；compile缓存动作仍有约0.25–0.37%相对L2变化，四观测路径相同不代表闭环质量已过。

|四固定观测、完整policy平均ms|target|TF|
|---|---:|---:|
|官方compile，共同prompt缓存|162.990|151.074|
|compile再加Cross-KV缓存|153.649|148.778|
|自有persistent+Cross-KV缓存|184.124|177.400|

这四观测来自一个task、两个init。共同Cross-KV优化后，compile下TF相对target的倍数从约1.079×变为1.033×；target收益比TF大。自有persistent仍比同缓存compile慢约19–20%。这是重要风险：后续必须用更强的公平公共基线衡量方法余量，不能只在旧基线上讲收益；也不能把这4观测外推为四suite新主表。[完整算子报告](../../.pipeline/experiments/fastwam-crosskv-cache-20261027/REVIEW_RETURN.md)和原计时已同步。该候选目前只保留研究入口，主投机默认未切换。

建议网页版优先讨论以下工作顺序，均需独立版本和原预算内登记，不能覆盖已冻结结果：

1. **重复运行/公平基线先收口。** 固定输入、加载顺序、RNG、eager/compile与B1/B7路径，定位跨进程差异；把prompt与Cross-KV公共优化和原质量版本对应起来，复查Spatial canonical差距，确定可比较的最终baseline。
2. **用最小消融判断预测器增量。** CAL-only对照显示previous-direction carry在后段可能更合算，但没有线上carry闭环/速度；补公平同阈值校准、同成本的carry vs小预测器，不能只凭“冻结target+小模块”宣称创新。
3. **修十步尾部、再研究深度。** 已交接剩余两步B3单层验证并保留合法bonus，剩余一步target；目前代码尚未实现。四套在索引7起树时均没有depth2接受。前期4–5层属于候选方向，二叉全树会变31/63个verifier状态，要共同比较深度/宽度/剪枝与`T_denoise/G`，不能外推高接受必带来收益。
4. **Robo分阶段补齐。** 登记自身episode隔离的FIT/CAL/DEV，先验证真实14D采集与批验证，再训练/新tau/小三臂；过实际工程门后按资源预算进入完整面板。不要把LIBERO阈值搬过去，也不要现在报告Robo三臂成功率。
5. **算子围绕实际B7成本。** 先改善候选验证、视觉KV布局与复制、投影/FFN2的并行组织，同时向双方提供相同缓存；必须在官方compile强基线和完整chunk上验证净收益。单层倍数与投机倍数不相乘。
6. **随后补跨模型证据。** 同一篇论文面向flow-based robotic action models；FastWAM优先，有时间和资源再核对StarVLA-PI/StarVLA-GR00T精确版本和接口，尚未启动。

论文主张仍是候选方向/状态、target在线验证、合法scheduler前缀与bonus提交及实际执行成本的组合证据。FastWAM已有cache/compile不是新贡献；[1Step-WAM](https://arxiv.org/pdf/2610.15625v1)已涉及冻结backbone、轻量方向校准与流水，[FLASH](https://arxiv.org/html/2605.13778v1)已研究连续动作并行验证，需围绕预测对象、验收对象、合法提交和计算组织逐项比较。连续近似接受不写成严格分布保持或无损保证。当前适合讨论下一轮可证伪实验，不适合宣布完整方法闭环或开始改写论文正文。
