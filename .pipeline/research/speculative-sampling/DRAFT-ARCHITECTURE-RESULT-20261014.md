# Draft结构筛选结果与verifier推进状态

2026-10-14，Asia/Shanghai。承接用户“可以的你继续做”的授权，完成有限CPU结构对照、独立结果审计、旧训练数据兼容性审查，以及隔离verifier profiler的实现和CPU预检。这是研究记录，不是论文正文。

## 本轮决定

**保留当前A_off和K2/B2部署接口；单层Transformer作为下一轮研究候选，尚未定版或开始扩大训练。** 18组从零训练中，Transformer在6/6组同学习率、同seed配对中优于容量相近的MLP，但没有任何一组通过预先冻结的全部准入条件。不能用全部样本平均改善掩盖root退化，也不在看过结果后放宽门槛。

MegaTransformer线已完成真实CFG14调用入口、生命周期、CUDA活动校验、账本和资源隔离的实现与独立静态审查。GPU监测期间持续有其他用户的进程占用，**真实profile、block捕获、融合CUDA原型与新的系统收益均未完成**。本轮没有GPU模型启动；有限10分钟只读资源监测已结束，没有后台自动启动队列。

## 已完成的18组实验

原始任务目录：[CPU18实验](../../../review/OMISSIONS.md)。[训练前plan](../../../review/OMISSIONS.md) SHA256：`3302d32ffc61d3f29bdbf6b047fb447283793e53c60d22483b5aa90a09838e08`。

- 三结构：MLP51（227→51→51→224，25,928参数）、MLP96（227→96→96→224，52,928参数）、Transformer64（53,070参数，16位置、1层、4头、FFN256、dropout0）。后两者参数量相差约0.27%。输入输出及双候选方向含义一致，所有结构从零初始化。
- 数据：同一3906行train、1890行validation，来自两任务；train为init0/1，validation为init8。validation只有2条相关episode，且已用于先前开发，不能当成新最终测试集或1890个独立试验。没有读取test数据。
- 预算：两档学习率1e-4/1e-3 × 三seed20260922/23/24 × 三结构，共18组；每组1000更新、batch512、同seed相同采样序列、AdamW和同一双头损失。相等的是训练样本次数和更新数，不是FLOPs。归一化只用train计算。
- 验证：使用记录的BF16查询逐行调用预测接口；每组checkpoint重载后全部1890行逐位一致。每步梯度有限、输入/RNG不变、16位置映射等检查通过。
- 用时：准备、训练及评估共1017.82秒（16.96分钟）。限制CPU0、单线程、nice19，CUDA不可见且未初始化。18枚checkpoint、逐组指标、日志和冻结数据保留。

下表新结构为三seed分量中位数；A_off是已有部署参照，其历史预训练与本轮scratch不同，不能据此声称某结构在等历史数据下优于A_off。

| 模型 | 学习率 | 全部局部通过 /1890 | root局部通过 /630 | A_off旧root通过丢失 /527 | root平均代理L2 | 全部平均代理L2 |
|---|---:|---:|---:|---:|---:|---:|
| A_off | 历史模型 | 1458 | 527 | 0 | 0.005331 | 0.006867 |
| MLP51 | 1e-4 | 1430 | 484 | 46 | 0.007172 | 0.007462 |
| MLP51 | 1e-3 | 1415 | 482 | 46 | 0.007743 | 0.007966 |
| MLP96 | 1e-4 | 1415 | 476 | 51 | 0.007410 | 0.007590 |
| MLP96 | 1e-3 | 1401 | 473 | 55 | 0.007777 | 0.007966 |
| Transformer64 | 1e-4 | **1483** | **515** | **21** | 0.005663 | **0.006507** |
| Transformer64 | 1e-3 | 1478 | 509 | 26 | 0.005695 | 0.006353 |

Transformer相对匹配容量MLP的6个配对root增量为17、41、42、24、36、39条。低学习率Transformer相对A_off的全部平均代理误差下降5.25%，但root平均误差上升6.23%，root通过少12条，旧root通过丢失21/527=3.98%。这说明需要同时看总体与root，不能只按全局loss选模型。

冻结准入包括root总通过及逐任务root通过不低于A_off、全部和root平均代理误差不增、旧root通过丢失不超过1%；配置还要求三seed中位数门通过且至少两个seed全门通过，拟选checkpoint本身也必须通过。**18/18均未入围，最终决定KEEP_FROZEN_A_OFF。** 1%是事先约定的工程筛选线，不是文献保证或成功率保证。

这里字段名`successor_l2`实际计算的是`abs(next_sigma-sigma) × ||BF16方向预测−教师方向||`。应称为步长加权方向L2或Euler后继代理；没有执行真实scheduler加BF16投影，也没有生成新树递归轨迹。这些数字不是实际投机接受率、闭环成功率、trace长度或GPU加速比。

## 退化定位：中sigma的root

[分组诊断](../../../review/OMISSIONS.md)针对Transformer lr1e-4；低sigma<1/3，中sigma在[1/3,2/3)，高sigma≥2/3。

| 样本组 | A_off通过 | Transformer三seed中位数 | 旧通过丢失中位数 |
|---|---:|---:|---:|
| root低sigma | 502/508 | 507/508 | 0 |
| root中sigma | 25/88 | 7/88 | 21 |
| root高sigma | 0/34 | 0/34 | 0 |
| child | 931/1260 | 968/1260 | 2 |

各分组中位数不必相加等于总体中位数。已观察到的模式是child及低sigma收益与中sigma root退化并存；尚未证明原因只来自loss、结构容量、训练时长或数据覆盖。下一轮应针对这组做预先冻结的采样/保留质量对照，不能把事后诊断当成已验证的修复，也不能直接部署sigma切换策略。

独立[结果审计](../../../review/OMISSIONS.md)924项通过，复算18个run、准入、配置选择、文件哈希和预算。审计重算保存的结果及源码合同，没有重新运行模型；原始训练评估的checkpoint回读检查另行保留。[代码审查](../../../review/OMISSIONS.md)说明代理指标、初始化历史及GPU接口等限制。

## 旧训练数据的可复用范围

[完整兼容性报告](../../../review/OMISSIONS.md)确认旧数据的state、previous direction、teacher direction和时间字段足以构造227维输入，无须因为旧模型名为context而丢弃。

但旧67,277行train跨四个suite，**其中libero_10只有4,802行、6个任务**，并且task0的833行已有历史使用，须去重。旧validation中的任务不能并入train。实际episode到init的映射，以及每份原trace使用的target权重身份，仍需原始启动记录认证；汇总provenance声明不能替代逐源证据。

优先按旧train原trace的`right.x_i / left.d_cfg / right.d_cfg`重建。旧mmap把`left.x_next`当state，裁为compact112后是否等价尚未做张量验证；全480维连续性误差可能涉及被置零的通道，不能据此宣判compact112错误。原target-only数据可以作为历史监督，不能当作当前投机候选访问状态上的same-query标签。

本次仅查代码、manifest和一个明确train文件的111KB pickle元数据，未读取tensor storage、旧留出张量或运行教师。尚未产出通过验收的扩展训练集，不应直接宣布“全量训练已开始”。

## Verifier已经准备好什么

新代码隔离在[megaverifier_profile_v1](../../../投机采样/experiments/megaverifier_profile_v1)。[冻结job plan](../../../review/OMISSIONS.md) SHA256：`806d217ae33ad3e59c1705d9a80f5013f3fd9f58829347df24e33a994de89201`。计划短/中/长各RAW、CURRENT捕获、CURRENT profile共9完整chunk，6次candidate profile和6个block0输入捕获；上限900秒、1500次target forward intents，仅一次模型加载。

[CPU预检](../../../review/OMISSIONS.md)11项、[CPU mock](../../../review/OMISSIONS.md)6项通过；[独立静态审查](../../../review/OMISSIONS.md)绑定214个来源文件和冻结plan。核查包括动态shared-prefix wrapper关闭与签名、异常恢复、CPU-only trace拒绝、30层标记、真实CUDA活动要求、同算法输出/路径/KV/RNG检查入口、累计预算和自有进程清理。它们不是GPU执行证据。

profile测量范围是candidate forward（包括其shared-forward准备），不含此前candidate packing/lease setup，更不代表整chunk延迟；带profiler耗时不能作为正式加速数字。各层包容时间不可直接相加，实际归因须避免重复统计CUDA活动。未来block kernel需保留scale/shift/gate原始非连续stride及FP32/BF16舍入语义。

仍待：独占窗口实际profile → 依据测得开销决定最小block融合 → 数值/路径/有效KV/RNG验收 → 完整verify及完整chunk正式计时。之前静态选出的四处modulation/gated-residual只是候选，尚未证明是主要瓶颈。

## 资源、保护与下一步

本轮新增GPU计算0秒，累计仍为30550.48840228282/36000秒，剩5449.51159771718秒（90.83分钟），target intents仍为144550/174494。10分钟资源观察结束时及收口复查都有外来GPU进程，未启动profile，没有新增reservation或GPU-ATTEMPT。CPU训练和只读观察均已退出；不终止他人进程。

当前部署模型、preset、受保护state/cycle/rules/config、原tile代码及HW保留。所有新实验为隔离代码；未删除资料或生成论文初稿。历史完整chunk结果仍是短/中/长KV **1.602×/1.544×/1.500×**，本轮不更新该数值，也不新增成功率结论。

[HW收口审计](../../../review/OMISSIONS.md)567项通过：216条旧Knowledge记录、47份tile证据、13个受保护路径保持原样；13个导航文件只新增内容，移除新增段后与旧文件哈希完全一致。新增记录`CHAT-20261014-DRAFT-ARCHITECTURE-SCREEN-RESULT`已进入references/pitfalls/decisions索引及compact视图。[进程收口记录](../../../review/OMISSIONS.md)确认训练PID已不存在、无匹配的自有worker和GPU进程；仍存在他人GPU进程。

下一阶段按依赖推进：

1. 数据线先核验旧train教师来源、init映射、实际查询状态及重复行，输出逐suite可用清单；无法认证的来源单列，不当作当前教师数据。优先利用现存数据，降低GPU采集成本。
2. 在匹配的数据和有限预算下，保留Transformer64与容量相近MLP对照，预先冻结中sigma root质量保留方案；扩大train时保留validation/test边界。模型入围后再做真实递归、完整chunk及成对闭环的成功率/trace验证。当前18组只是结构初筛，不必为了形式上的“最终结构”停止数据工作。
3. 系统线在独占GPU窗口启动已经冻结的profile；如账本或源码有变化，先重新准备并审查job。启动须实际设置CPU1–7，不能只在plan中声明。拿到profile后决定融合范围，再实现和验证。

内部里程碑沿用10/17首版候选与协议、10/21更广质量表、10/26系统原型验收、10/30主证据冻结、11/01 18:00交老师。当前结构未入围、数据来源未完全认证和GPU窗口不确定都是实际风险；这些日期是工作目标，不是完成承诺。剩余GPU预算仅够有边界的验证，不能默认为覆盖全任务训练/闭环/消融。

复现入口：[源码模型](../../../投机采样/experiments/draft_architecture_v1/models.py)、[实验runner](../../../投机采样/experiments/draft_architecture_v1/run.py)、[最终结果](../../../review/OMISSIONS.md)、[执行清单](../../../review/OMISSIONS.md)。
