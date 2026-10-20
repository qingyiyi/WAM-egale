# 自有整层kernel内FA2：attention局部通过，整层仍未准入

2026-10-20T11:28:22.776486+08:00。沿已有M39记录直接迭代optimized，没有新建实验/Cycle。原双缓冲[RESULT.md](RESULT.md)及240/900登记不一致FAILED保留；本阶段使用最新用户明确授权的FA2接入工作，不复用旧预算。状态：`FA2_LOCAL_CONTRACT_PASS_WHOLE_BLOCK_NO_GO`。

## OBSERVED：已经实现和验证

当前[whole_layer.cu](../../../算子研究/wam_ops/optimized/csrc/whole_layer.cu)在一个cooperative persistent kernel内完成整层。self/cross attention改为固定Torch2.9所用FA2的设备主循环、split-KV和combine；不是host调用SDPA或再启动attention kernel。固定上游FA2/CUTLASS没有修改；[fa2_codegen.py](../../../算子研究/wam_ops/optimized/fa2_codegen.py)提取原函数，调整worker索引、named barrier和KV地址。[地址加载](../../../算子研究/wam_ops/optimized/csrc/fa2_load.cuh)保持公共历史和候选私有suffix的原可见性，没有复制14份历史；[设备阶段](../../../算子研究/wam_ops/optimized/csrc/fa2_stage.cuh)保持原split heuristic和算术顺序。

9个真实block0输入：action B2/B14、video B2，各短中长KV；同输入self/cross共18个对照全部与Torch Flash SDPA有限逐位相同。9次重复、输入/RNG/参数身份、tile恰好一次、metadata和有效KV，以及18个合法update1→2→0状态转移通过。native对原Torch9/9通过；整层FA2版本对原Torch仍0/9通过，所有四项质量权限保持false。无30层、chunk、闭环或新训练。

GPU实际108 CTA×256线程；GEMM仍warp0领任务、1/2双缓冲加载、3–6 WMMA、7写回。attention由warp3–6执行完整4-warp FA2 load/MMA/softmax/store，其他warp等待；尚未把FA2 loader额外拆给warp1/2。普通FA2 tile为Q128/KV64，split为Q64/KV128，短Q使用合法padding/mask；B14始终14个独立Q16实例，未跨候选注意。split counts为B2 action self短3/中9/长4，cross4；B2 video self/cross均2；B14均1。combine仍在同一kernel内通过grid barrier接续。

运行时函数资源为255 registers/thread、82464B dynamic shared、2168B localSizeBytes，理论1 CTA/SM。编译中间对象日志显示242 registers/704B stack，与最终加载函数资源不同；报告采用运行时值。localSizeBytes是函数本地存储资源，不等于实际spill流量。NCU计数/实测occupancy仍UNKNOWN。

## OBSERVED：误差到底多大

相对L2定义为||new−Torch||₂/||Torch||₂；不等于每个元素的相对误差。单位是单层hidden，不是机器人动作或接受分数。旧列是同次复测的M39 async，当前列是内部FA2整层。

| 真实输入 | 旧相对L2 | 当前相对L2 | 当前最大绝对误差 | 当前不同元素数 |
|---|---:|---:|---:|---:|
| short-B2-action | 0.2373% | 0.2671% | 1 | 28485/98304 |
| middle-B2-action | 0.3599% | 0.3600% | 1 | 28420/98304 |
| long-B2-action | 0.3173% | 0.3172% | 1 | 28358/98304 |
| short-B14-action | 0.5597% | 0.5632% | 1.5 | 273706/688128 |
| middle-B14-action | 0.3438% | 0.3495% | 1.5 | 242659/688128 |
| long-B14-action | 0.4274% | 0.4227% | 1.5 | 268935/688128 |
| short-B2-video | 0.1307% | 0.1329% | 0.5 | 106915/786432 |
| middle-B2-video | 0.1698% | 0.1491% | 0.5 | 107701/786432 |
| long-B2-video | 0.1372% | 0.1368% | 0.5 | 108701/786432 |

所有case的绝对误差P50为0；当前P90为0.000244140625–0.0009765625。最大误差发生在大幅值hidden，例如B14短91→92.5、长90→88.5，B2 action短124.5→125.5，video长64.5→65.0。不能因均值小或多数元素相同而放宽逐位门，更不能直接推断闭环安全。

## OBSERVED：已定位边界 / HYPOTHESIS：原因

1. **attention设备数学路径已局部对齐**：18同输入逐位PASS。原差异可通过采用相同FA2 tile/归约/split路径消除，但本结果只覆盖这些真实输入，不证明任意输入。它不保证上游Q/K/V已有误差时完整attention输出也相同。
2. **Q投影独立误差仍在**：相同modulated输入下，B2 action为68/98304元素不同、最大0.0625、相对L2约0.01560%；B14为132528–141176/688128不同、最大0.0625–0.125、相对L2约0.2012%–0.2048%。video Q投影逐位相同。不是全由前序误差传播。
3. **FFN2独立误差仍在**：相同GELU输入下，action最大0.25，相对L2约0.1988%–0.3310%；video最大0.0625，相对L2约0.2690%–0.2695%。自有WMMA递增K和cuBLASLt的内部归约/分块/舍入路径不同是有源码支持的候选机制；尚无证据将其唯一归因于split-K，旧split-K匹配挑战者仍失败。不能声称精确指令级根因已找到。
4. **FA2之后的逐阶段定位更清楚**：补充18个完整block将native三维/四维debug视图和workspace二维视图按相同逻辑元素顺序比较，原shape mismatch不再误称数学差异。video三点整个self-attention、self输出投影与第一次残差均逐位一致；长video直到FFN1/GELU全部一致，仅FFN2/最终输出不同，直接定位该case后段误差。
5. **另有小量LayerNorm2差异**：短/中video输入残差已经逐位相同，Norm2分别12/786432和8/786432元素不同；最大0.000244140625/0.0009765625、相对L2约6.91e-7/2.78e-6。共享Welford公式不代表编译后归约和affine指令完全相同。具体差异指令尚UNKNOWN；不把后续cross差异误归为FA2失败。

[DAG逻辑视图](fa2-evidence/dag-1/RESULT.json)、[同输入与状态审计](fa2-evidence/validate-1/RESULT.json)、[完整精度CSV](FA2-PRECISION.csv)。原validate中的DAG_native shape mismatch原样保留，逻辑视图补测补充而非改写原记录。

## OBSERVED：整层速度

同一空闲远端A100-SXM4-40GB GPU0，CPU4/5；两个独立进程，3路径交错，每cell合并8次，共216计时样本。CUDA event覆盖完整forward，缓存准备/NVML资源检查在计时外；18份独立profiler trace各1 compute kernel，profile不计入表。固定原Torch/扩展版本、相同输入和canonical模型参数；这是未准入原型的uncached eager单层测量，不是最强cached/Graph B14或机器人E2E。

| 真实输入 | native ms | 旧async ms | 内部FA2 ms | 相对旧耗时变化 | 当前/native耗时 |
|---|---:|---:|---:|---:|---:|
| short-B2-action | 1.484 | 4.923 | 5.319 | +8.0% | 3.58× |
| middle-B2-action | 1.637 | 5.992 | 5.534 | -7.6% | 3.38× |
| long-B2-action | 1.761 | 7.152 | 5.641 | -21.1% | 3.20× |
| short-B14-action | 2.722 | 18.909 | 20.678 | +9.4% | 7.60× |
| middle-B14-action | 3.027 | 21.779 | 21.044 | -3.4% | 6.95× |
| long-B14-action | 3.453 | 26.383 | 21.510 | -18.5% | 6.23× |
| short-B2-video | 1.937 | 9.779 | 9.475 | -3.1% | 4.89× |
| middle-B2-video | 2.048 | 13.247 | 9.638 | -27.2% | 4.71× |
| long-B2-video | 2.186 | 17.305 | 9.842 | -43.1% | 4.50× |

短action退化约8%–9%，长action改善约18%–21%，长video改善约43%。FA2增加单层寄存器/本地存储需求、短Q tile补零及split-combine工作，有可能抵消短KVattention收益；这是待隔离机制，不是本轮硬件计数证明。全部中位数与逐进程结果见[JSON](FA2-RESULT.json)、[全部计时CSV](FA2-TIMINGS.csv)，不选最快轮。整层仍慢于native约3.20–7.60倍，不报告生产加速。

## 资源、失败、清理

实际387/500完整block、18同输入SDPA、18同输入Linear；0机器人局。GPU worker生命周期合计69.979秒（不是CUDA active秒）；含CPU构建的7个job合计221.183秒。新阶段从T0到最终清理836.9秒，低于原3600秒上限；本阶段新增文件约6.79MiB，低于512MiB帽。首次两次CPU编译失败分别为缺失flash参数头和const指针赋值，已修正；原失败日志保留，无GPU超时或自动机器人重试。

[真实清理收据](fa2-evidence/CLEANUP.json)：7组父子共14个PID/birth均已退出，两卡无compute context。75项本地保护源码/依赖校验不变；当前whole源/入口是用户授权改动，不拿旧冻结字段判失败或改写旧manifest。未改baseline/投机代码/原tile/state/cycle/rules。旧M39资源登记错误与本阶段预算分列，未补造收据。

## RECOMMENDATION

下一步优先对齐FFN2，再处理Q投影和小量Norm2差异；保留当前已通过的attention对照。长video给出了最清楚的FFN2隔离输入，可用于限定归约/舍入定位。数值边界清楚后再处理寄存器压力、短Q FA2策略和跨阶段ready调度；不能仅删除grid barrier掩盖基础计算慢点。

RoboTwin调研见[既有报告10/20补充](../robotwin-migration-survey-20261019/RESULT.md)：50双臂任务，有官方Target，但需重做draft接口/训练/校准；本阶段没有安装或运行第二环境。
