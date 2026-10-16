# 10月16日夜间执行结果

更新时间：2026-10-16 05:21，北京时间。状态：本机有限机制、完整chunk正确性、缓存共同基线全部完成并清理；AutoDL新104局完成并清理，原始小档已归档。TF质量门未通过。本文件为研究执行报告，旧FAILED和既定协议均保留。

## A. 动态事实

本机实际完成三档KV、1/3/7状态的完整30层机制矩阵，补齐静态CUDA Graph以及同一7个真实候选逐个B2的整体Graph基线。Q02完成真实TF task7前10个完整policy/cache chunk两遍重放。新跨候选attention原型v1数值失败；保留负结果后，最后一次固定几何规则v2通过有限门，完整30层的长KV局部改善在两个独立进程复现。本夜本机已登记GPU作业均已终止，自有GPU进程均清理；全局A_off、原tile、撤回融合和TF正式45chunk质量门未改。

旧AutoDL T/A_off 100局完整但原严格相机配对FAILED。独立描述性结果：T48/50、A_off47/50，target胜2、spec胜1、双方成功46、双方失败1；双方成功长度比P50=1、P90=1.0454、max=1.5939；全任务800失败罚分均值285.50/298.64。50对物理/环境RNG相同，4对腕相机hash不同；旧像素数组及target初始RNG缺失，不能修补旧证书。它不是TF、不是最终盲TEST、不是正式性能。

新AutoDL独立链于北京时间05:04:01签发完整研究终态，104/104局（4条reset对照+T50+TF50）及自有进程清理通过，未复用旧T。原4小时帽从01:19:05开始，05:14停止工作、05:19完成归档清理，早于用户09:00授权上限；实际新链约3小时45分。脚本清理不等于租卡实例已关机，实例未关闭。

|新50对主结果|T|TF64 child2500|
|---|---:|---:|
|成功数|48/50|44/50|
|全部实际环境步数合计|14301|16255|
|失败记800的均值|286.18|325.58|
|实际policy chunks|931|1049|
|目标模型调用/完整chunk（含video/cache）|73.897|43.513|

配对共同成功43、仅T成功5、仅TF成功1、共同失败1，任务等权成功率差为−8个百分点；固定Long10任务内成对init bootstrap 2000次，描述性95%区间[−18,0]个百分点，不能建立3pp非劣。43组共同成功的TF/T长度比P50/P90/max=0.99617/1.05430/1.40；最大来自task6/init25（245→343步），初始相机逐位相同。预定max≤1.30门与成功数门均未过。TF失败为task1/init29、3/26、4/26、5/28、6/27、8/25；T失败0/28、5/28。本次task7双方均5/5，不能把旧init24失败硬编码成task7问题。

50对实际物理/控制器/RESET与目标初始CPU/CUDA RNG逐位相同；原相机47/50逐位，3对腕图分别task2/init25差2pixel、task8/init25和8/26各差1pixel，最大幅度1，不设容差。严格像素47组仅作敏感性子集：T45/47、TF42/47，差−6.383pp；不能替代50对主表或修补旧FAILED。独审已经重读全部100局原始结果、50对实际初始数组和两相机及真实RNG tensor，见[统计独审](../../../投机采样/results/runtime/p0-autodl-terminal-quality-independent-stat-review-v1-20261016-r1/REVIEW-RESULT.json)和[全原始覆盖增量](../../../投机采样/results/runtime/p0-autodl-terminal-quality-independent-stat-review-v1-20261016-r1/RAW-COVERAGE-INCREMENT.json)。

真实20237条旧CAL的GPU adapter/decode同设备BF16逐位、finite、RNG检查通过；旧查询离线接受率约94.3%–95.5%，并未换来新闭环质量达标。18896棵实际树depth0/1/2=2543/813/15540，平均接受深度1.6878。目标调用从68798降至45645，却增加实际chunk数量并降低SR，不能把调用下降直接报为任务加速。原始cost构造未重算，CAL接口逐位也不是递归质量证明。

时间仅为共享PCIe机器描述：bank428.38秒，4条reset验证阶段622.17秒；T50阶段6704.53秒（平均134.09秒/局，含阶段间开销），TF50阶段5491.40秒（109.83秒/局）。不是本机正式chunk/task/E2E速度。4条reset原/bank物理及相机数组等价，双方质量均用同一bank；不得将旧reset比例沿用到新账本。

## B. H1–H4判断

|假设|现有证据|边界/缺口|
|---|---|---|
|H1 小action存在shape效率限制|真实输出投影M32/96/224；NCU三次实际launch的SM吞吐约24.19%/38.71%/47.65%，局部行吞吐随M改善|单权重暖缓存profile，只支持局部机制；不能外推全部307个Linear或称HBM饱和|
|H2 同组候选批量验证有经济性|同7个真实query，完整30层顺序B2 Graph与B14 shared Graph比较，约2.45/2.91/3.36倍（短/中/长）|跨独立job的静态fixture机制参照，非正式机器人速度；不同batch的BF16结果不一定相同|
|H3 新调度超过最佳已有B14|未缓存强基线下两进程长档下降9.64%、全档4.98%；新增同形状文本缓存共同基线下，单进程完整pilot短/中/长下降3.86%/6.45%/18.40%，全六query10.27%|新共同基线结果尚待独立复测；缓存生命周期与完整chunk性能未验证；新引擎1/3状态、shape交互R和cached G1控制缺失；不能把原有共享prefix思想当新贡献|
|H4 单位有效提交与闭环收益|新TF50目标调用减少，但SR48→44/50、最大成功轨迹比1.4，当前质量条件未满足|新缓存/调度仍无完整chunk及episode联合成本，需实际ΣT_round/Σg；当前TF不能开放正式性能或宣称完整机器人加速成立|

## C. 实际shape和DAG

target隐藏3072、24头×128、FFN14336、30层。112维紧凑action不是Transformer隐藏维。实际action embed输入为[14,16,30]；普通投影输入[14,16,3072]，库已展平M224。n1/n3/n7的M分别32/96/224，CFG未拆成独立模型调用。attention每实例仍为Q16，单纯batch增大没有将其改成一个互相可见的Q112序列。

每query实际307个Linear：30 block各10个，加7个embedding/projection；305个输入BF16、2个time embedding输入FP32。文本投影/cross K/V输入有[14,512,3072]，M7168，不能将所有GEMM都说成M224。按2MNK统计，文本相关占线性运算量约81%，不是81%墙钟。

|算子|实际M/N/K（B14）|
|---|---|
|action embedding|224/3072/30|
|self Q/K/V/out、cross Q/out|224/3072/3072|
|cross K/V|7168/3072/3072|
|FFN up/down|224/14336/3072；224/3072/14336|
|time embedding两层（FP32输入）|224/3072/256；224/3072/3072|

block依赖：norm/modulation→QKV→QK norm/RoPE→合法历史+私有16-token self attention→out/residual→norm/cross attention(text)→residual→norm/modulation→FFN→gated residual。不能跨未完成归约随意流水。CFG两组公共历史不同，候选间私有KV不可互见。

旧shared kernel已去除历史materialization，但一个CTA处理一个candidate/head，没有CTA内跨候选共享；KV只存一份不代表只从HBM读取一次。未缓存原六份完整profile中cross占kernel duration54.4–62.2%，主要KV投影；旧eager prompt cache负结果保留。该profile不代表本轮新静态缓存后的热点，下一次路线选择需更新profile。

## D. 本轮实际测量

以下以first/last两真实query中位数给范围，单位ms；各arm保留6次样本，profile不混入计时。只有静态真实fixture，不含输入更新/Graph重建/生产cache生命周期。

|总KV|同7query顺序B2 Graph|已有B14 shared Graph|整组验证批处理经济性|单状态Graph|
|---|---:|---:|---:|---:|
|288|166.58–166.62|68.09–68.17|约2.45×|23.86–23.87|
|1152|213.76–213.77|73.49–73.63|约2.91×|30.63|
|2160|265.85–265.97|79.01–79.11|约3.36×|38.03–38.07|

必须采用合理优化的单状态baseline。B14 shared/单状态Graph局部成本比约2.855/2.402/2.078；三个真实A_off chunk每树平均合法推进g为2.526/2.333/2.286。K2+bonus每树最多推进3步，因此短档即使忽略draft/pack/accept等开销、每轮均推进3步，按这里的局部成本也只有约1.05倍上限。该比较是固定first/last成本的敏感性分析，不是已经测得的ΣT_round/Σg或chunk负收益，但显示更强target基线会显著收窄投机收益。

上述1.05倍仅是未缓存静态Graph的局部上限；下节同形状缓存改变了双方成本，不能将旧上限沿用到新配置。

完整三chunk各50次scheduler更新；树数19/21/21，树内Σg48/49/48，加起始普通target更新2/1/2才各为50。bonus/fallback只按真实推进计一次。 [既有完整episode日志审计](../../../投机采样/results/runtime/p0-wam-existing-episode-kv-field-availability-v1-20261016-r1/AVAILABILITY.json)检查TF10局3611树及A_off10局4738树；现有小日志没有逐轮有效KV长度，不能据frame/chunk编号补造真实短中长权重。本表继续三档等权，下一版采集应记录实际validKV与合法提交数。

Q02：32真实树、224候选、256额外target forward；方向32/32树不逐位，最大差0.3359375，但接受路径/深度翻转0/32；原action/video、完整有效KV/RNG复现。只覆盖task7前10chunk的记录重放，未重建机器人环境反事实。局部方向差异存在，当前样本不支持将其写成task7失败根因。

实验故障保留：完整矩阵r1资源中断，r2入场BLOCKED未启动GPU，r3draft-mode入口错误，r4完成；NCU r1包装pin失败、r2实际采得3kernel但CSV单位行解析使原监督器FAILED，独立CPU报告解析真实counter而不改原状态；顺序Graph r2在首次replay前误读capture输出，修复后r3完成。714为顺序Graph保守Python forward intents，物理设备等价forward为672，不能混写。

## E. 原型决定

本夜只实现路线B的最小数值原型，独立于原tile和生产入口。G1为不跨候选的BM64控制，G2/G4为两种登记分组，共享同CFG prefix，私有suffix逐行mask；不做跨CFG合并、不加稠密mask、不改action horizon、不恢复旧融合。

长KV三层：G1/G2均逐位；G4每层不同822/2050/1577元素，max差0.00390625/0.0078125/0.00390625，停止晋级。扩大G2检查：短KV first/last共66对照全部逐位；中KV first的第0层G2不同1152/688128元素，max0.001953125。监督器完成该原native完整query并核冻结输出/有效KV/RNG后停止：实际67候选对照、90 native calls、3完整model forward，36.76GPU作业秒并清理。

v1版本NO-GO，未计时、未集成。中档prefix1136、G2虚拟K1168与原K1152使BN64分块从18增至19是源码事实；尚未实验隔离唯一数值根因。之后仅做一次独立v2修订：当ceil((P+32)/64)=ceil((P+16)/64)时用G2，否则G1。它是固定几何判据，不按task/失败名单回退，也不构成普遍BF16等价证明。三个真实prefix272/1136/2144分别使用G2/G1/G2。

v2实际完成180层probe及180层实际替换，共12个完整30层forward、540次attention；六个实际替换完整输出与冻结原输出逐位、有效KV/输入/slots/RNG不变，58.45GPU作业秒并清理。随后两个独立进程各完整六query×三臂，不挑长档、不挑最好轮：每进程198 Python intents（其中18 capture不执行）、180设备等价forward，全部18次eager准入先过才计时，所有216个测量replay及36个保存输出的CPU字节核验通过。三臂为native BM128、G1 BM64、guarded_v2；native→G1不能全归因于tile尺寸。

|真实KV|两进程合并native→guarded延迟下降|解释|
|---|---:|---|
|288|1.826%|未达到5%投入筛选线|
|1152|2.895%|guarded本身使用G1，无跨候选分组收益|
|2160|9.644%|native→G1下降4.178%，G1→guarded再下降5.705%，长档增量复现|

六query等权Σ中位数下降，r1为4.8997%、r2为5.0593%，合并4.9796%；不能挑r2写全档稳定超过5%。停止第三次重复和分组调参；随后仅做一次实际attention counter归因和一次独立完整chunk正确性验证。输入更新、Graph构建、动态cache生命周期均不在本表中。[Cascade原始说明](https://flashinfer.ai/2024/02/02/cascade-inference.html)已有共享prefix/query复用、prefix/suffix统计合并及A100评测；不能将本原型的复用思想直接写成新贡献。

随后一次真实long2160/layer00 NCU归因完成并独审：native/G1/G2的CTA为336/336/192，寄存器255/168/168，shared memory每block66.560/50.176/50.176KB；L2请求508.791/477.613/320.893MB，DRAM访问70.193/67.076/66.197MB。G1→G2减少32.81% L2请求，DRAM仅降1.31%，achieved occupancy反而从17.86%降到11.15%。它支持跨候选减少重复片上请求和CTA工作，不支持“HBM减半”或“occupancy越高越快”。该NCU每launch实际11个counter passes，暖输入且cache/clock未控制；407.232/296.480/171.296µs是profile耗时，不能混入Graph表或机器人性能。输入/全部有效KV/CFG/slots/RNG/版本保持，profile和离线导出全部自有进程已清理，GPU作业计费11.13秒。

完整chunk S04目前的三次独立登记如下，旧失败没有被重写或拼接成通过：

|登记|实际结果|原因与后续边界|
|---|---|---|
|v1|FAILED；19 forward、42.8064 GPU作业秒、0/6完整、guarded尚未调用；已清理|观察器误把普通target临时槽metadata当成B14严格只读；失败后差异数组缺失，不能补判“仅metadata变化”或有效KV通过|
|v2观察语义修正|超时FAILED；268 forward、785.8769秒、5/6完整、250提交；已清理|短/中两对严格一致；long/native完整，long/guarded仅普通video21次、无B14/提交；全量状态审计耗时估计不足，不是已记录的模型数值差异|
|同数值预算续测|COMPLETE；6/6完整、300提交、294 forward、938.4385秒并清理|三档全量新配对通过，未拼旧通过臂；三job累计581 forward、1767.1217秒、93,178,320B，均在登记预算内|

普通原生路径会写空闲槽、结束撤销mask；v2核验准确的空闲槽id/is_pred变化，同时保持全部原有效KV、有效metadata及mask不变。B14仍检查全池metadata和版本严格只读。v2部分结果实测40个guarded B14×30层及108普通query×30层；21600个id变化全部位于原无效槽。后续完整预算版补齐长档实际配对，全部六臂均各完成50次提交。

完整预算版已经[独立实际结果审查](../../../投机采样/results/runtime/p0-wam-s04-budget-independent-result-review-v1-20261016-r1/REVIEW-RESULT.json)：276逐query输入/输出及18历史准备，共294次target；122个B14×30层，其中guarded实际1830层；130普通query×30层准确核验scratch。9对保存的actions/latents/obs共4,793,760元素另用CPU逐字节及finite核验；全量有效KV、RNG、scheduler、VAE/server状态一致。范围仍是三个原A_off snapshot的完整policy/cache重放、0新环境episode，不是TF质量或正式性能许可。三job累计上限600 forward/128MiB，GPU预留2028.68秒小于用户S04总3小时。原900秒协议与v1/v2 FAILED保持，详见[观察纠错决定](../../../投机采样/results/runtime/p0-wam-s04-observer-correction-decision-v2-20261016-r1/DECISION.json)、[同实现预算续测决定](../../../投机采样/results/runtime/p0-wam-s04-budget-continuation-decision-v1-20261016-r1/DECISION.json)、[资源续测独审](../../../投机采样/results/runtime/p0-wam-s04-budget-independent-source-review-v1-20261016-r1/REVIEW-RESULT.json)。

static prompt缓存共同基线v2已经COMPLETE并清理，[独立实际结果审查](../../../投机采样/results/runtime/p0-wam-static-prompt-baseline-v2-independent-result-review-v1-20261016-r1/REPORT.md)通过。六真实query×六臂=36行，所有36次eager准入先于任何Graph；216次六顺序平衡测量、36保存输出逐位/finite通过，410 Python intents/374设备等价forward，输入/全部有效KV/slots/RNG/参数版本及缓存输出不变。424个实际NVML样本未见foreign进程。单作业325.346秒，新增约0.88MiB，原blocked v1与全部旧失败保留。

|KV|cached单状态C1|cached原B14|cached新B14|新调度对共同cached B14延迟下降|
|---|---:|---:|---:|---:|
|288|19.119ms|31.203ms|29.998ms|3.861%|
|1152|25.949ms|35.737ms|33.431ms|6.452%|
|2160|33.451ms|41.422ms|33.800ms|18.402%|

各档是first/last两query各六样本中位数的均值；全六queryΣ中位数下降10.274%。仅一个独立进程，不能称两进程复现。缓存本身让C1下降20.554%/15.880%/12.648%，原B14下降53.481%/50.217%/46.722%；这是普通共同工程收益，不是MegaTransformer新贡献。cached新B14/C1局部成本比1.569/1.288/1.010，不能替代实际ΣT_round/Σg或完整chunk速度。未测cached G1独立控制，18.4%不能全部归因跨候选分组。

缓存分别保存原B2和B14完整61个投影输出，禁止从B14切片冒充原B2数值。payload合计2.859375GiB，另有64MiB prompt副本；实际allocated peak17.707GiB。首填wall分别B2为92.364ms、B14为697.575ms，包含完整第一次原forward、clone与库first-use，并非独立cache构建成本。旧reset bank保存512-token编码，本实验缓存的是Transformer内部投影；两者须分开计账。当前Graph固定输入已准入，实际输入更新、Graph构建、prompt/权重/CFG变化与cache失效均不含在表中；未接生产，未完成cached完整chunk/机器人质量。旧eager缓存慢17.33%/23.79%/6.48%记录继续有效，不用新静态条件覆盖。

## F. 质量收口与放行

TF64结构可暂冻，child权重只是研究候选。原fresh30的task8腕图像素配对FAILED保持；独立描述性诊断为T10/10、TF/MLP均9/10，TF成功子集长度条件通过但成功数门失败。新AutoDL50对扩大测量也观察到成功数下降与max长度门失败，TF权重/接受器不能作为全量生产版本冻结；TF64结构和7状态接口可暂冻供系统线继续。

96个真实teacher查询与target-only历史CAL同σ对照时，root通过率37.5%对37.24%，第一层反事实候选37.5%对57.78%；较大差异集中未提交节点。进一步查原树，40个不通过的first查询对应父边40/40已经超过τ；在固定父边下仅改善这些子预测不能增加合法提交深度，不代表联合重训绝无收益。

之后补齐[更同类的旧candidate审计](../../../投机采样/results/runtime/p0-wam-old-candidate-support-cpu-v1-20261016-r2/REPORT.txt)：14997行实际上保留4999完整三节点树、teacher/previous/selected/depth/commit。历史TF aux与当前风险96行按精确σ链及root/first层级匹配，root旧/当前通过率均37.5%、误差均值0.01393/0.01509；first通过率也均37.5%、误差0.02266/0.02146。加入已记录reach类别后仍能匹配；不同generator的head编号并非稳定语义，严格node编号+reach匹配的0分母单独保留，不伪造同节点等价。之前target-only对照的明显劣化在同类candidate对照中没有复现。旧FIT与当前风险DEV仍存在选择/初态混杂，这只削弱“新数据缺口”论证，不证明没有分布问题。

完整当前10局3611树的depth0/1/2为492/155/2964；若只修子边而固定父预测，每树额外合法推进上限155/3611=0.0429。相同风险96行CPU诊断中常速度v_prev局部误差均值0.01830、通过40/96，TF best-of-two为0.01934、36/96；不是GPU接受或闭环效果证明。[FlowCast（ICLR2026）](https://proceedings.iclr.cc/paper_files/paper/2026/hash/93e06b762457778f3e9832b9c90d5ad2-Abstract-Conference.html)已采用常速度外推和MSE接受，因而必须保留常速度基线；当前96行诊断不等于其完整算法复现，也不能把生成轨迹误差界外推为机器人成功保证。上述两项官方来源于10/16再次核对。D01/D02继续HOLD，不为Transformer外观启动全量训练。扩大质量和原型完整chunk正确性已收齐；下一步核查严格配对的失败/最长病例，并补强target缓存生命周期及单位有效提交成本；若以后有明确可达父边监督缺口，再按一次30完整episode与等预算三训练条件推进。

新50局的[实际提交路径审计](../../../投机采样/results/runtime/p0-tf-heldout-path-cost-descriptive-v1-20261016-r1/REPORT.md)完成：1049 chunks、18896树、31893条实际提交draft边，113376个候选FP32记录的选路与提交守恒全部通过。只累计选中的合法路径，不加拒绝分支、不重复计算bonus/fallback；没有原velocity，不声称重算score。先逐episode再组平均，44成功/6失败的每chunk累计score为0.0643402/0.0633901，每边为0.00211407/0.00208932；六个有成败的task内，失败局每chunk均值也全部较低。失败episode总和较大同时伴随50chunks，不能由此认定累计误差根因，当前结果不支持直接选累计预算阈值。预登记σ段实际提交边在≤0.2、(0.2,0.4]、(0.4,0.6]分别27939/3244/710，>0.6为0；高σ仍可执行target。此审计不触发新训练或新接受器，也不改变质量FAILED。

实际3,840,000训练抽样索引、FIT/CAL/aux及当前远端来源未见init30–49；benchmark500只有prepared/preflight。可以预留本项目FIT30/31、CAL32、DEV33/34、TEST40–49；上游target预训练完整来源未知，不能声称绝对未见。25–29扩大结果解盲后不能改称最终盲TEST。

当前派生状态为`engineering_quality_screen=false`、`independent_test_complete=false`、`noninferiority_established=false`、`formal_performance_permission=false`；其中独立TEST指未来冻结后的主TEST，不能把已解盲的25–29研究测量改名。冻结候选/接受器/树/引擎和TEST清单→独立TEST→按原协议质量放行→本机独占正式性能。工程筛选、独立TEST完整性、3pp非劣结论、正式性能许可四项分开；50对即使零新增失败，也不能仅凭非显著宣称3pp非劣。

## G. 默认关键路径

10/16已收齐扩大质量及终态；S04完整预算续测已通过并清理，新静态缓存共同基线也已测得收益；10/17–18先做一次独立复测、缓存生命周期与完整chunk验证，再更新profile和单位有效提交账本，同时完成质量风险收口。仅证据触发才做新数据/等预算训练；不展开cross-attention、PF2/3/4或多树搜索。最多一个成本支持的小树对照也必须独立登记和重验，n3诊断不等于已验证生产树。

10/20前条件性训练完成，10/21有限DEV，10/22明确quality+系统go/no-go；10/23冻结后进入TEST，10/26 TEST收齐、10/27获准正式性能、10/28冻结主表。10/29–31整合中文稿，11/01 18:00前交老师。主线未通过时收缩系统主张及可选环境，不增加新方向；不能保证按时形成正论文结论。

后续运行量已从实际归档核验：[预算估算](../../../投机采样/results/runtime/p0-wam-future-q03-q04-volume-plan-review-v1-20261016-r1/PLAN-ESTIMATE.json)。条件性Q03为6条件×20组=120局；Q04为T100+spec100+3个baseline各50局=350局，三个baseline共用事前固定的主T50子集，不额外再跑150条T。当前PCIe单worker的阶段跨度估计：Q03常规4–5.5小时；Q04均值代理12.5小时、加25%余量约15.6小时，另留1–2小时启动/CAL/归档；失败长尾可超过18小时预算。仅是未来规划，不能当SXM4性能或运行上界。小draft历史六组2000步CPU2训练约15.6分钟、六组500步约9.8分钟，完整episode才是主要墙钟开销；新数据采集和新训练均未启动。

乐观：1–2天内从少量严格病例找到可检验修复，同时缓存生命周期验证通过，10/21完成DEV。常规：先用1–2天补实际分歧证据与强baseline，随后2–3天做有限必要baseline及条件性训练DEV，10/22评审；新TF−8pp和max1.4意味着该安排有实质质量风险，并非承诺按时通过。若10/22仍无质量或B14增量证据，停止扩展，保留机制/负结果与明确缺口给老师判断。RoboTwin、LIBERO100、真机和第二架构均不占当前保证路径。

[CVPR2027官方日期](https://cvpr.thecvf.com/Conferences/2027/Dates)：注册11/10 AoE、正文11/16、补充11/23，对应北京时间11/11、11/17、11/24的19:59；当前页面没有另列abstract截止。[官方CFP](https://cvpr.thecvf.com/Conferences/2027/CallForPapers)需作者OpenReview资料；10/18检查、10/25处理遗留，现无作者清单，不能声称已检查。11/02起按老师反馈修订/翻译/排版，内部提交至少提前一天。

## H. 原始证据与维护

- [短索引](WAM-COORDINATION-20261016.md)、[短状态](../../PROGRESS.compact.md)、[Cycle](../../cycles/wam-research-20261016/PROGRESS.md)。实际HW为已授权切换的v15，原v12/legacy保留，不再次升级/回退。
- [三档机制矩阵](../../../投机采样/results/runtime/p0-wam-shape-audit-v1-20261016-r1/full-matrix-r4/MATRIX-RESULT.json)、[静态Graph](../../../投机采样/results/runtime/p0-wam-shape-audit-v1-20261016-r1/graph-matrix-r1/MATRIX-RESULT.json)、[同7query顺序Graph](../../../投机采样/results/runtime/p0-wam-same-seven-sequential-graph-v1-20261016-r3/SEQ-RESULT.json)。监督器终态及PLAN在各自同目录。
- [机制独审及NCU](../../../投机采样/results/runtime/p0-wam-mechanism-results-independent-review-v1-20261016-r1)、[Q02](../../../投机采样/results/runtime/p0-wam-quality-q02-replay-v1-20261016-r2/RESULT.json)、[96行匹配对照](../../../投机采样/results/runtime/p0-wam-quality-q02-matched-cpu-v1-20261016-r1/REPORT.txt)。
- [原型源码](../../../投机采样/experiments/wam_candidate_tile_v1/PLAN.md)、[G2扩展数值负结果](../../../投机采样/results/runtime/p0-wam-candidate-tile-fixture-v1-20261016-r2/FIXTURE-RESULT.json)、[独立来源审查](../../../投机采样/results/runtime/p0-wam-candidate-tile-fixture-independent-review-v1-20261016-r1)。
- [v2有限数值门](../../../投机采样/results/runtime/p0-wam-candidate-tile-guarded-v2-20261016-r1/GUARDED-RESULT.json)、[两进程三臂合并独审](../../../投机采样/results/runtime/p0-wam-candidate-tile-graph-pilot-independent-review-v1-20261016-r2/TWO-PROCESS-COMBINED-REPORT.md)；原始r1/r2各目录保留所有轮，未择优。
- [实际attention counters独审](../../../投机采样/results/runtime/p0-wam-candidate-tile-profile-independent-review-v1-20261016-r1/R3-TERMINAL-COUNTER-REPORT.md)，原CSV与NCU report见被引用的profile r3目录；r1/r2均为未启动GPU的准备/审查修订，记录保留。
- [S04失败与原生临时槽合同核验](../../../投机采样/results/runtime/p0-wam-s04-independent-result-review-v1-20261016-r1/REPORT.txt)，实际job的CHUNK-RESULT/SUPERVISOR-RESULT与完整冻结源码分别保留。
- [新50对完整结果](../../../投机采样/results/runtime/p0-autodl-failed-old100-research-dispatch-v1-20261016-r1/remote-evidence/fixed104/ANALYSIS.json)、[小原始证据归档](../../../投机采样/results/runtime/p0-autodl-failed-old100-research-dispatch-v1-20261016-r1/COMPLETE-SMALL-TRACE-ARCHIVE.json)。六tar共3196件/60,528,640B，含全部104局结果、命令/动作、接受/阶段计数、初始相机/状态/RNG；不含大KV、obs_data或视频，不能称完整大张量轨迹。
- [四完整诊断episode视觉保全](../../../投机采样/results/runtime/p0-autodl-failed-old100-research-dispatch-v1-20261016-r1/OBS-DIAGNOSTIC-PRESERVATION.json)：TF失败task1/init29及其T配对、最长共同成功task6/init25及其T配对；全部102个obs_data只含两相机uint8数组，逐件内容/名字/大小审计，七tar合计173,834,240B，52.95秒只读传回、05:14前完成。仅供后续诊断，不是FIT/CAL/TEST；没有KV/latent/权重/视频。远端原件未删，本机该dispatch所有证据共244,470,836B（约233.1MiB）。
- [50局路径score诊断](../../../投机采样/results/runtime/p0-tf-heldout-path-cost-descriptive-v1-20261016-r1/REPORT.md)、[全部原始质量字段独审增量](../../../投机采样/results/runtime/p0-autodl-terminal-quality-independent-stat-review-v1-20261016-r1/RAW-COVERAGE-INCREMENT.json)。
- [AutoDL动态终态入口](../../../投机采样/results/runtime/p0-autodl-failed-old100-research-dispatch-v1-20261016-r1/status.json)、[原质量协议](STATISTICAL-QUALITY-PROTOCOL-20261015.md)。

[最终本机作业账本](../../../投机采样/results/runtime/p0-wam-shape-audit-v1-20261016-r1/NIGHT-LEDGER-FINAL-v2.json)共28个GPU/CPU准备namespace，已知GPU作业生命期3030.814秒（50.51分钟，非CUDA活跃时间），主要产物逻辑约2.77GiB。491项原tile/保护源/原记录size+mtime守卫不变，未重做整套模型hash、未删除失败、未复制逐query30层完整历史。旧派生ledger漏映射两个旧job的`own_process_reaped`字段；最终v2只修读字段，原监督器均为true，所有原job终态不改。AutoDL另计，不混入SXM4速度或本机GPU预算。

最终[报告只读独审](../../../投机采样/results/runtime/p0-overnight-final-local-independent-review-v1-20261016-r2/REVIEW-RESULT.json)PASS：51个本地证据链接存在、491项原保护记录不变，完整报告/短视图数字与权限一致；短入口5997B、本轮索引3279B。此处追加复核回执及Cycle本轮完成标记，不改已核验研究结论。
