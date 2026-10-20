# RoboTwin迁移核查：有官方权重，但不是直接换环境

状态：READ_ONLY_SURVEY_COMPLETE；0 GPU、0模拟器安装、0权重下载、0机器人episode。源身份见[SOURCE-MANIFEST.json](SOURCE-MANIFEST.json)，50个唯一任务见[TASKS.csv](TASKS.csv)。以下方案不是第二环境实验授权。

## OBSERVED

- RoboTwin 2.0官方benchmark含50个任务；论文采用Aloha AgileX双臂，区分Easy/clean与Hard/domain-randomized。任务涵盖交接、双臂抬物、叠放排序、工具按压、开合与摆放；这里的分组是我们便于分析的归类，不是LIBERO那种四套官方suite。[论文§4.5](https://arxiv.org/html/2506.18088v2#S4.SS5)、[官方任务](https://robotwin-platform.github.io/doc/tasks/)。
- LingBot-VA确有[官方RoboTwin checkpoint](https://huggingface.co/robbyant/lingbot-va-posttrain-robotwin)，revision `8c9dea8abbc5c91cc9e18bc3264b8915083bbe70`；是LingBot-VA，未与LingBot-VLA混用。推理需要该策略权重，环境本身和专家动作生成不要求神经策略权重。无需从头训练5B Target。[官方模型入口](https://github.com/robbyant/lingbot-va)。
- 对应官方适配源固定LingBot-VA `7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb`。README要求RoboTwin `2eeec322d95799f537cbfe5f291a8220d965ccb8`；不是当前主分支。最新版已引入XPolicyLab，接口与该旧适配不能视为相同。[固定README](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/README.md#evaluation-on-robotwin-20)。
- 官方单卡脚本为`demo_clean`、eval seed0、`test_num=100`。多卡脚本为了8 GPU×7组将50任务补成56项，最后两任务有重复；不能报成56个独立任务。单条件完整规模是50×100=5000局/方法，T/C/P三臂15000局；如果两个条件各100，则30000局，不含训练与调试。[固定脚本](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/evaluation/robotwin/launch_client_multigpus.sh)。
- 不是直接取LIBERO的50个固定init。client从`10000*(1+eval_seed)`开始逐seed建场景，先运行专家检查，跳过不稳定/不可解种子，再评估100个有效场景。三臂必须共享预登记的有效seed、指令及初态；跳过数与原因单独记录，不能把基础设施筛选当模型成功。[固定client 393–504行](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/evaluation/robotwin/eval_polict_client_openpi.py#L393)。

## 真实接口差异

| 项目 | 当前LIBERO | 官方RoboTwin配置/源码 |
|---|---|---|
| 相机 | 两相机 | head＋左右wrist三相机，256×320 |
| 动作有效通道 | 7 | 16：两组EEF pose7＋gripper1，放入30维padded action |
| action token/chunk | 16 | 2 frame×16 action/frame=32 |
| video token/chunk | 本轮真实fixture Q128 | 固定tshape配置推导240，尚未运行捕获 |
| 历史pool容量 | 2160 | 固定配置推导9792；当前整层kernel的4096上限不覆盖 |
| compact latent元素 | 112 | 16×2×16=512，按源码维度推导 |
| video/action更新 | 20/50 | 25/50 |
| action SNR shift | 0.05 | 1.0 |
| guidance | 当前冻结CFG2协议 | video guidance5、action guidance1；原server按use_cfg走两路 |
| 归一化 | 当前LIBERO normalizer | RoboTwin quantiles/q01/q99 |
| Transformer主体 | D3072/H24/d128/F14336/30层 | HF小配置相同；action输入输出30维 |

配置与维度来源：[va_robotwin_cfg.py](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/configs/va_robotwin_cfg.py)、[server 398–414行](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/wan_va_server.py#L398)、[模型action展开](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/modules/model.py#L830)。512仅指16有效通道的紧凑表示，是静态接口推导，未运行张量捕获。原server实际上按`[1,30,2,16,1]`分配action噪声，即960元素；`_prepare_latent_input`每次teacher查询前清零inactive通道。不能把服务器原分配说成512，也不能直接改为只抽512个随机数而声称RNG等价。见[噪声分配456–462行](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/wan_va_server.py#L456)与原mask调用。client还做初始EEF pose叠加、四元数归一化，再`take_action(..., action_type='ee')`；不能沿用LIBERO的命令映射。

当前TF64的7维state＋7维previous velocity＋3标量、16个位置，不能直接加载为RoboTwin draft。若沿同思路，应重新登记32-token/16-channel接口与权重、RoboTwin数据、normalizer、FIT/CAL及接受阈值；不能默认沿用LIBERO tau。算子主体维度可借鉴，但本轮整层kernel只登记Q16/Q128，也不能自动声称已支持Q32。

容量推导已核对共享配置`patch_size=(1,2,2)`：tshape latent为24×20，2×24×20/4=240 video token，pool=(72/2)×(240+32)=9792。若CFG2、30层、D3072且K/V均BF16，固定pool的K/V静态容量约6.72GiB；这是形状预算，不是实测总显存，也不能再与官方总显存描述直接相加。来源为[shared config](https://github.com/robbyant/lingbot-va/blob/7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb/wan_va/configs/shared_config.py)、server 390–407行和model 662–670行。

RoboTwin循环使用各任务`TASK_ENV.step_lim`；不沿用LIBERO horizon800或800失败罚分。未来需按RoboTwin原任务上限另登记长度/失败成本口径。

## UNKNOWN

HF列出的Transformer约10.179GB、text encoder约11.362GB、VAE约2.819GB，合计约24.36GB（十进制）；只读metadata，未下载或验证与现有组件可共享。RoboTwin资产、SAPIEN/Vulkan在BAAI的安装/渲染兼容性、实际空间及每局耗时尚未测量。官方旧适配要求同机server/client；换成当前XPolicyLab分机接口是额外工程工作，不能混称开箱可用。未核验所给权重的精确训练划分/Hard覆盖。

## RECOMMENDATION

先收口四套LIBERO时间账本和本轮单层算子，不立即加15000局。若之后批准第二环境：优先固定官方兼容版本和现成Target，先登记6–10个不同机制任务的小面板，再单独采集/训练draft、审计动作/初态/质量；小面板明确是迁移DEV，不能代替50任务标准结果。新环境单局耗时UNKNOWN，现阶段不提供伪精确总时长。稳定后按实际单局分布估算5000局/臂预算，再决定是否扩全套。

没有安装系统依赖、启动第二环境或新增训练；本调研不会自动接入现有GPU队列。

## 2026-10-20补充：与LIBERO的规模及迁移工作

本轮重新浏览官方任务页、LIBERO仓库、LingBot-VA仓库及RoboTwin权重页；未启动RoboTwin。下列比较以现有固定适配源码为准，未用公开main覆盖本地科研代码。

| LIBERO套件 | task数 | 控制的变化/用途 |
|---|---:|---|
| Spatial | 10 | 空间布局和相对关系 |
| Object | 10 | 操作对象 |
| Goal | 10 | 任务目标 |
| LIBERO-100 | 100 | 综合任务；拆为LIBERO-90（90）与LIBERO-10/Long（10） |

总共130种；我们的四套Spatial/Object/Goal/Long覆盖40种，每任务50个init不是50个task。[LIBERO官方说明](https://github.com/Lifelong-Robot-Learning/LIBERO#benchmarking-knowledge-transfer-for-lifelong-robot-learning)。LIBERO当前单臂Franka/MuJoCo-robosuite，RoboTwin固定适配为双臂Aloha/SAPIEN；后者的双手协同、三视角、EEF位姿动作和域随机化增加了迁移内容。

RoboTwin没有与上述四套一一对应的官方分类；50唯一任务在Easy/Hard条件下评测，不能计成100种。为了回答任务类型，按既有[TASKS.csv](TASKS.csv)互斥归类如下；**这是我们的工程归类，不是官方suite或论文分层指标**：

| 分析类别 | 数量 | 例子 |
|---|---:|---|
| 抓取、摆放与容器操作 | 26 | place_object_basket、put_object_cabinet |
| 叠放与排序 | 6 | stack_blocks_three、blocks_ranking_rgb |
| 工具与按压 | 6 | beat_block_hammer、press_stapler |
| 姿态调整与摇动 | 4 | adjust_bottle、shake_bottle |
| 双臂搬运 | 3 | lift_pot、pick_dual_bottles |
| 关节物体与开合 | 3 | open_microwave、open_laptop |
| 双臂交接 | 2 | handover_block、handover_mic |

总和50，任务名单源于固定官方评测脚本并去除多GPU补齐重复；类别之间实际行为可以重叠，上表只是固定的单标签分桶。[官方任务说明](https://robotwin-platform.github.io/doc/tasks/)。

迁移建议的依赖顺序：

1. 独立环境固定官方兼容commit，核查SAPIEN/Vulkan、资产、三相机渲染；现有535驱动兼容性待实机验证。
2. 使用官方RoboTwin Target和原动作后处理，先复现完整任务baseline并保存严格匹配的有效seed、指令与初始化。官方已提供Target，无需从头训练大模型。
3. 将draft输入/输出、位置编码改到32个action token和16有效通道；保留30维padded teacher/RNG协议。重新采集该Target监督、训练draft、校准normalizer和阈值，隔离FIT/CAL/DEV/TEST。不能直接搬LIBERO权重与tau。
4. 适配S/T/PF2路由、完整scheduler计数、reset text bank及三相机cache；环境终止/长度按原step_lim。单层算子另需Q32/videoQ240/更长KV支持与真实输入核验，不与draft迁移同时改变数值路径。
5. 先用6–10个跨机制任务的完整配对DEV判断迁移可行性；只有获准后扩50任务×100有效seed×三臂=15000完整episode。新DEV不是独立TEST，不改当前四项false。

工程估时（推测性排期，不是已测耗时或执行授权）：环境与原Target完整复现约1–2工作日；投机/数据接口及审计约1–2工作日，资源或渲染不兼容会更久；新数据/训练/质量时间待测。15000局总GPU工作量可按单局均值t分钟估算为250t GPU小时；两卡各单worker理想墙钟125t小时。例如仅作为容量情景，t=2/5/10分钟时为500/1250/2500 GPU小时，两卡理想250/625/1250小时；这些不是RoboTwin实测，更未计数据采集、专家筛选、准备、失败长尾及争用。并行worker收益必须测量，不能按进程数线性外推。

默认建议：先完成LIBERO算子数值与速度收口，第二环境作为随后有预算的迁移验证。RoboTwin可检验双臂和不同动作表示的泛化，但不会自动修复当前GEMM误差或使现有质量门通过。
