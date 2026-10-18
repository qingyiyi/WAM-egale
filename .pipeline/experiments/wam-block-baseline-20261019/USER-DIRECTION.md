# WAM / MegaTransformer：先建立完整 block 基线，再研究 tile 调度

你是服务器上的研究工程 Codex。请使用中文审查并准备实现。本任务不是让你立即开发整层 megakernel，也不是仅再做一轮 attention microbenchmark。

## 0. 本轮目标和权限

本轮要回答并尽可能完成：

> 当前是否只有 attention 核心的独立实现？是否已存在可单独调用、准确复现原始语义、性能接近原版的完整 Transformer block 顺序基线？若缺少，就补齐最小基线、测试和测量链；在门检通过前，不启动 persistent scheduler 开发。

长期目标是单个 Transformer block 内 self-attention、cross-attention、FFN、norm、残差和调制的依赖感知 tile 调度，不是30层整体融合。

默认权限：

```text
MODE = CPU_AUDIT_AND_BASELINE_PREPARE
ALLOW_NEW_GPU_JOBS = false
ALLOW_SIMULATOR = false
ALLOW_TRAINING = false
ALLOW_NEW_EPISODES_OR_TEST = false
ALLOW_MIGRATION = false
ALLOW_PRODUCTION_PATCH = false
```

可在独立新 namespace 内进行非破坏性 CPU 源码审查、清单生成、测试/基线代码准备和受限编译。若现有权限不覆盖文件写入或编译，则输出补丁和命令提案，不执行。GPU 数值、性能、profiling、模型加载、CUDA context 创建均需明确有效的授权；转发本 prompt 不构成新增 GPU 授权。没有授权也继续完成只读审查与 CPU 交付，不等待、不虚构执行。

本机模拟器已退出，禁止恢复。本机开发不再迁移；新机继续既定评测，不向其运行进程注入本轮代码。不要取消、重启或改变既定评测。

四项字段保持 false：

```text
engineering_quality_screen=false
independent_test_complete=false
noninferiority_established=false
formal_performance_permission=false
```

不改旧 FAILED、阈值、retention、质量门、draft 权重/训练、候选树、接受器或 PF 周期。没有闭环质量保证时，只能报告限定输入下的算子结果。

## 1. 固定版本和真实证据

```text
repository = https://github.com/qingyiyi/WAM-egale
REVIEW_COMMIT = ca5f37e0482c2e38e3738e3d8e71e3b185990a23
```

不得用 main 或更晚结果替代此提交。用只读 `git show REVIEW_COMMIT:path` / `git ls-tree` 检查固定内容。记录当前 HEAD、dirty diff 和正在执行任务的身份；不 reset、clean、强制 checkout、覆盖或 stash 用户工作区。派生实现保存在单独 namespace，记录 base SHA 与 patch hash，不冒充固定快照内容。

先完整读取：

1. review/CURRENT-PROGRESS-20261018.md
2. review/PROJECT-REVIEW-PROMPT-20261018.md
3. .pipeline/cycles/wam-research-20261016/STATUS-20261018.md

随后沿它们的真实引用读取：最新对齐结果、原始 CSV/JSON、测试脚本、编译/来源清单、资源账本、完整 block/performance 报告及限制条件。

至少检查实际存在的下列入口，并沿导入链追到函数和 CUDA launch：

- 投机采样/wam-p0-worktree/wan_va/modules/model.py
- 投机采样/wam-p0-worktree/wan_va/wan_va_server.py
- 投机采样/experiments/operator_torch_alignment_v2/
- 当前实际使用的 integration / install / restore 路径
- 当前 aligned / shared-prefix / guarded / cache 实现
- 当前 baseline / block / full-model / saved-input validator 和 benchmark

目录缺失时标 UNKNOWN，从固定提交的索引和引用定位；不要编造替代文件。旧 review/RESULTS.json 是 paused72 历史快照，不用它覆盖当前进度。

网页端未成功读取本次固定提交。因此，本 prompt 不断言新结果具体通过几例，也不预设当前仍失败。服务器实际源码和已归档结果优先；已有严格证据应继承，不重复无目的旧测试。

每项关键判断记录：文件、行号、函数、源码 hash、输入 manifest、产物/日志身份。缺少原始 tensor、权重或二进制时，区分“读取已发布结果”与“独立验证”，标 UNKNOWN，不声称重算。

## 2. 首个交付：不要把替换范围与验证范围混为一谈

输出 BASELINE-COVERAGE.md，分别回答：

### 2.1 替换范围

对下列实际运算逐项标注：TORCH_ORIGINAL / OWN_STANDALONE / OWN_DEVICE_PRIMITIVE / MIXED / NOT_APPLICABLE / UNKNOWN。

- self-attention 的 SDPA 核心
- cross-attention 的 SDPA 核心
- self Q/K/V 与输出投影
- cross Q/K/V 与输出投影；区分已有 context K/V cache
- QK norm、RoPE 等实际存在的处理
- 各处 norm、时间调制、gate、residual、cast、布局变换
- FFN1、真实激活、FFN2
- KV 读取、写入、metadata 和 read-only verifier 路径

不要把 `install()` 接管 attention 函数误写成接管整个 block。不要把 Python `forward` 封装误写成 device-side 原语。

### 2.2 验证范围

为每个实现单独列出：

```text
kernel -> 子层 -> 完整block -> 完整模型 -> 完整policy chunk -> 机器人闭环
```

记录支持的模式、batch/CFG/candidate 组织、Q/K 长度、hidden/head/FFN 维度、dtype、stride、层号、缓存状态、只读/写入分支。

“完整模型中仅替 attention 后通过”不等于“完整模型所有计算已自有化”。

测试报告至少拆分：

```text
collection_complete
finite_pass
bitwise_pass
tolerance_pass
state_contract_pass
performance_status
scope
```

COMPLETE 不等于 PASS。只读 KV 不变与合法写路径的状态等价分别检查。已有有效证据能覆盖的项目不重跑。

## 3. 明确定义四种基线/候选，禁止跨层冒称

### R0：原模型完整 block reference

直接调用固定提交真实原 block；不手写“标准 Transformer”替代。不更换原权重、norm/激活、epsilon、mask、RoPE、CFG组织、投影 bias、cast、cache 或 scheduler 语义。

R0 是原始完整 block，不是随机张量上的 attention gold。允许用 R0 调两次验证测试框架自复现，但这不是自有实现验收。

### R1：仅安装已对齐 attention 的混合完整 block

原 block 的投影、norm、FFN 等保持原实现，只在已经验证的范围内安装 aligned attention。命名为混合/attention-replacement block。

若已存在就复用，不重新写一个副本。R1 需相对 R0 有适用范围明确的数值/状态证据。

### B0：显式分阶段的、未融合顺序 block 基线

本轮核心交付。在独立模块中按真实 block 顺序组织阶段，暴露中间张量边界、权重引用、依赖与生命周期；便于替换某一基元和测量完整执行。

第一版允许：

- attention 使用已验证实现；
- Linear/GEMM、norm、激活等暂时复用原版 Torch 或原已验证库调用；
- 保留必要中间缓冲，不引入新调度，不改变 batch 分组。

明确叫 B0_HYBRID_SEQUENTIAL，不谎称全自有 CUDA block。B0 应使用原参数、原运算和实际阶段组织，不能只是 `return original_block(...)` 后声称完成分解；R0/R1 wrapper 本身可以这样调用，但角色必须分清。

不要求为建立 B0 从零重写所有 GEMM、norm、softmax。复制固定原 block 控制流并保留来源/许可，通常比凭记忆重构更合理。

### D0 / P_seq / P_sched：后续设备级阶段

- D0：将计划改动的基元做成可独立测试的 device-side tile 原语。
- P_seq：相同设备原语，按严格阶段顺序执行。
- P_sched：相同设备原语与数学配置，仅改变就绪任务分配/执行顺序。

B0 通过不等于 D0 已存在。host 侧 Torch/cuBLAS 调用不能直接作为 persistent kernel 内的普通 device 函数。需要确认实际可嵌入 CUDA 的实现、资源和调用约定。

本轮不自动开发 P_sched。先完成 B0；只针对第一个拟研究子图列出 D0 缺口，不全面重写所有子层。

## 4. 从真实源码提取 block 契约与依赖图

输出 BLOCK-CONTRACT.md / block_contract.json：

- 真实模块类型、参数对象/hash、bias、norm epsilon、激活精确模式；
- 实际 eval/inference/autocast 设置、dtype 转换和显式舍入边界；
- TF32、FP16/BF16 reduction、SDPA backend、determinism 等当前配置；只记录，不擅自修改；
- 每个 Linear 的逻辑 M/N/K、leading dimension、stride、布局及是否已有合法静态缓存；
- query/context、时间条件、mask、candidate/CFG 映射；
- KV 有效范围、元数据、输入 alias/view、cache 读写副作用；
- 各阶段输入、输出、完整归约条件、可重叠计算与内存生命周期。

不得假设 video/action 维度相同，不从 tensor 总元素数猜 shape。不得把7候选描述成可跨候选attention的Q112/224。线性层能否flatten batch和token需由真实布局决定；必要copy必须记录。

标记哪些消费者需要整行 hidden 完成，哪些需要该候选完整可见 K/V，哪些可流式消费。归约尚未结束不能发布完整 ready。

特别保持：

- FP32 -> BF16 -> 后续运算等显式转换；
- 原 approximate GELU 等激活，不替换为更容易融合的函数；
- 原 norm，不把 LayerNorm 换成 RMSNorm；
- 原投影组织、split 和接受语义。

从参数/布局推算 tile 数量时，标成估算，不冒充实测。K增加不必然增加独立输出tile；不要只凭总FLOPs判断可调度并行度。

## 5. 输入与测试准备：真实 fixture 优先

复用既有真实模型/块输入、权重和状态快照。只需 block 时不加载完整模型；既有 fixture 足够时不重新采集模拟器数据。

真实 fixture 缺失：

- 明确缺字段与可恢复路径；
- 可写 synthetic unit test 骨架，结果标 SYNTHETIC_ONLY；
- 不用 random input 冒充真实分布；
- 不自动发起新 episode、模型查询或新机数据搬运。

fixture 至少描述：输入、条件、时间、权重身份、mask、CFG/candidate映射、KV有效区/metadata、RNG、dtype、stride、alias。

R0/R1/B0 各臂从相同逻辑状态开始。独立保存可变状态，但保持每臂内部的 alias/view 语义；不能用随手 `.contiguous()` 或独立clone破坏原布局/别名契约。只读权重可复用。

若内部格式变换是设计的一部分，明确“外部契约”和“内部规范化”；prepared与integration开销分列，不把准备工作无偿移出测量边界。

首轮以真实 B14 action 短/中/长 KV 为核心最小集合；对 B2 action/video 和不同分支按契约补必要控制。先固定最多6个有辨识力的完整block fixture；通过后预算允许再扩到最多12个。需要更多才能覆盖的范围标 PARTIAL，不自动扩无限矩阵。

层/查询身份依据已有 manifest 预登记，不硬编码聊天中的旧失败点；诊断点与性能确认点分别登记。

## 6. GPU验收阶段：仅在明确授权后执行

### G0：参考自身和测试框架

- 先 R0 vs R0 确认该fixture、状态恢复和执行环境可复现。
- 若参考自身不稳定，停止严格替换判断，定位原因。
- 不通过私自改精度/确定性配置来制造与旧gold一致；必要变更另建契约和授权。

### G1：变更基元的隔离验证

只对本轮变更/未覆盖部分逐项比较。可复用 R0 真实中间输入，定位实现本身差异。

记录：finite、logical_shape/dtype、字节不一致计数、max_abs、rel_l2、误差分位数，必要时ULP，使用事前确定的阈值。

逐位指同逻辑元素/同dtype的比特内容，不把不同布局的padding或未初始化存储纳入；`torch.equal` 对数值相等的判断不足以单独证明所有字节模式一致。非有限数先检查，不让相同NaN伪装通过。

### G2：完整B0顺序执行，不使用gold中间结果喂给后续阶段

真实传递 B0 自身输出到下一阶段，比较最终 block 输出与状态。

隔离单算子时可teacher-force输入；完整block测试中不得每一步重新替换为R0中间tensor，否则会隐藏误差传播。

失败时开启debug tap定位首次分叉；通过后的正常计时关闭debug tap/hook/逐元素检查/CPU日志。边界可包括真实存在的norm/modulation、QKV/QK norm/RoPE、self SDPA、self投影/残差、cross query/context KV/SDPA/投影、FFN前norm、FFN1/激活、FFN2、最终残差。

首次分叉前输入若已不同，不把差异全部归因于当前kernel；只改一个证据支持的因素，最多两次纠错尝试。

### G3：状态契约

只读 verifier：有效 KV、metadata 与输入按原契约不变。

合法写路径：比较R0与B0的最终有效KV、写入范围、metadata、必要RNG状态；不是要求一概“不变”。不要比较无语义的未初始化cache区域。

### G4：胜出版本最小集成回归

若B0真正改变生产计算路径，使用已有fixture在原模型安装/还原入口做完整模型query回归。按现有权限/输入条件决定是否做完整policy chunk；不恢复模拟器。

只读完整query不代替policy chunk；完整chunk包括原scheduler、终端cache工作和真实返回动作边界。新基线未通过的范围标未覆盖，不用局部通过补齐。

不要把所有kernel单独通过当作完整block通过，也不要把block/模型通过当闭环质量通过。

## 7. 性能验收：先“可比较”，再“接近”

建立下面明确的对照，而非只比较Torch eager与新Graph版本：

1. R0 eager；
2. R0在相同合法cache/Graph条件下；
3. R1同等条件；
4. B0同等条件；
5. 最强合格B14/shared-prefix/guarded/cache组合（仅使用对应精度已通过的实现）。

兼容的最强组合及shape dispatch在确认测量前固定，不能逐点事后取最好噪声样本拼表。不兼容的cache/guarded不能凭名字相加。

区分计时范围：

- prepared kernel/subgraph GPU execution；
- 完整block GPU execution；
- 真实接口端到端墙钟，含实际准备/布局/缓存维护；
- 完整verifier/model；
- 完整policy chunk。

同边界比较。不得用Graph attention时间除以eager block时间推瓶颈占比。

测量要求：

- 先A/A噪声检查；warmup、build、加载不计steady-state，但单独记资源账。
- CUDA Event在正确stream依赖完成处计GPU时间；墙钟明确同步边界。
- ABBA/BAAB交错，至少3个独立执行块/轮次；预算允许时跨进程复核。
- Graph replay不是独立统计样本；记录每轮原始结果与独立样本单位。
- 只读输入可重复；会写KV/latent时每次正确恢复状态，prepared与含恢复的integration成本分列。
- Graph保持合法地址、shape、状态和同步；不能把CPU cache更新被capture忽略当作优化。
- 正常计时不带profiler、不逐算子同步、不在热区打印或刷盘。
- profiler另开有限诊断窗；先时间线，再只对已定位热点采计数。
- 资源争用/时钟异常且无法排除，标PERFORMANCE_UNKNOWN，不宣称稳定对齐。
- 输出均值、中位数、配对比值/几何平均、范围和适合独立样本的区间，不只报最好值，不用少量样本给精确P99。

建议的新B0性能筛查门（不是旧质量/retention；GPU阶段执行前登记）：

- 在核心fixture集合上，B0相对已匹配的原路径，完整block配对时间比几何平均 <= 1.05；
- 不出现可复现的 >10% 的单个分层退化；
- A/A噪声应足以区分这些阈值，否则PERF_PENDING而非强判通过；
- B0的数值PASS与性能FAIL/PENDING分开，不抹掉正确性成果。

这些是“是否值得在该基元上加调度”的工程筛查，不是证明统计非劣性或任务级加速。最强候选基线单独保留，B0接近原Torch不等于打败最强shared/cache。

若B0只是拆分原调用但慢，先检查额外分配/clone/contiguous、Python分派、同步、后端/shape变化、cache失效及实际计时范围，不先重写大kernel。

## 8. B0之后的决策：只给设计，不默认启动调度GPU实验

输出 SCHEDULER-READINESS.md，列出：

- 已有可嵌入device的attention/GEMM/row-op原语；
- 哪些只有host调用；
- 哪个边界的实测可优化份额足够；
- 理想收益上界与当前GEMM损失风险；
- 哪些输入就绪粒度、归约和显式舍入限制了推进。

默认长期设计：完整block调度范围；静态任务结构+轻量ready计数；GPU常驻CTA领取tile；优先M/N输出tile，首版不增加跨CTA split-K；允许必要global scratch。

首个实际原型可选真实FFN1->激活->FFN2行组链，但必须以profile和device基元可用性为依据。若缺少可对齐且不明显慢的GEMM原语，先验证原语，不宣称scheduler已可实现。最多选择一个子图，候选配置最多两个。

后续研究必须同时提供：

```text
强基线 R_strong
P_seq：相同设备原语，严格阶段顺序
P_sched：相同设备原语，只改变就绪任务调度
```

P_sched超过P_seq说明机制有增量；仍慢于R_strong则不能宣称系统胜出。普通缓存/Graph/pointwise融合收益单独归因。

“跑满GPU”只是手段；最终看完整block/verifier时间以及全部成本下的sum(T_round)/sum(g)。不将occupancy或target-call减少当最终成果。

## 9. 24–48小时计划、资源与停止规则

T0从本任务在服务器开始计，阶段耗时为规划，不是必须运行到满预算。

- T0–4h：完整读入口、真实状态/coverage、固定source/artifact身份。
- T0+4–12h：B0最小顺序基线、fixture adapter、验证/benchmark代码、CPU契约测试。
- T0+12–24h：仅在授权覆盖时完成最小GPU门检、性能诊断。
- T0+24–48h：必要一次证据驱动修正、范围内回归、输出是否可进入scheduler的结论。

新资源申请建议（不自动获得，不重置历史账）：

```text
CPU <= 2核
CPU累计墙钟 <= 8h
GPU = 既有单张A100，至多1个本任务worker
GPU生命周期累计 <= 120min
新增持久化文件 <= 1GiB（含报告、有限trace、构建产物）
新的机器人episode = 0
新的训练/采数 = 0
GPU纠错尝试 <= 2
```

GPU120min建议分配：启动/环境15、最小数值门25、完整block计时与局部profile35、胜出版本范围内回归30、清理15。若模型加载/输入体量更大，缩减覆盖并报告，不超额或挤掉清理。

若原账本、存储cap或权限更严格，遵守更严格者。预算不包含未来scheduler GPU开发；不能将本阶段剩余数额默认为那一阶段授权。

停止条件：

1. 无GPU授权：停止在CPU准备和待运行清单，不调用CUDA。
2. 原始fixture/权重/依赖缺失：UNKNOWN/BLOCKED，不虚构输入补证明。
3. R0无法自复现：不进入严格替换判定。
4. 局部首次分叉未定位：不跑大矩阵、完整闭环或改多个数学因素。
5. 严格精度/状态失败：保留失败；不放宽容差/改dtype/改gold。
6. 两次有依据纠错仍失败：停止该路线，输出首次分叉与缺口。
7. 基线稳定明显慢或计时不可比：先输出原因/UNKNOWN，不用scheduler掩盖。
8. 预算/存储/清理deadline将不足：不启动下一job；只清理身份验证过的owned资源，不影响其他任务。

允许结果是“完整block数值基线已建立，性能/集成仍待验证”，不要求为了交付而制造全PASS。

## 10. 具体代码与交付

优先复用已有目录、helper和manifest；不要重复造通用框架。确有缺口时，在独立namespace准备：

```text
block_baseline.py          # R0/R1/B0入口，清晰阶段，原参数/语义
fixtures.py                # 最小必要数据契约、可变状态恢复
validate_block.py          # 逐基元/完整block/状态/首次分叉
benchmark_block.py         # 同边界eager/Graph、原始配对记录
README.md                  # 真实可执行命令、构建与权限说明
```

名字只是建议；不要为匹配本prompt重构已有合理代码。变量可读、注释充分；避免过度封装和大量无意义helper。所有未知路径从配置/manifest解析，不能写死猜测目录。CPU静态检查可做；实际未运行的命令写NOT_RUN。

必需报告：

1. FACTS-AND-UNKNOWNS.md
2. BASELINE-COVERAGE.md
3. BLOCK-CONTRACT.md + machine-readable manifest
4. ACCURACY-RESULT.json（或PENDING，禁止预填PASS）
5. PERFORMANCE-RESULT.csv + 边界与噪声解释（或PENDING）
6. SCHEDULER-READINESS.md
7. RESOURCE-AND-STOP-REPORT.md
8. SOURCE-EVIDENCE-PACK.md：固定源码引用、关键字段、hash，不含原tensor/权重/相机

最终中文回答必须直接回答：

A. 当前到底只有attention baseline，还是已有混合/顺序完整block baseline？分别有哪些证据？
B. 本轮新增哪些真正执行代码？哪些仍是Torch库调用？哪些能进入device kernel？
C. 输入、输出、有效状态、dtype/stride契约对齐到哪个范围？
D. 完整block性能是否接近原版，是否存在异常慢点？计时是否可比较？
E. 下一步是否可以开发tile scheduler？若不能，唯一优先补的基元/证据是什么？

不要把“架构设计合理”写成“代码已经存在”，不要把“报告收齐”写成“基线通过”。

## 11. 官方依据与来源要求

以下是方法参考，不是仓库执行证据。若实际复用外部源码，另固定兼容commit、文件hash、许可证、编译选项；不能默认升级依赖或下载main覆盖现有vendor。

- PyTorch 2.9 Numerical accuracy：
  https://docs.pytorch.org/docs/2.9/notes/numerical_accuracy.html
- PyTorch 2.9 CUDA semantics（异步计时、stream、Graph）：
  https://docs.pytorch.org/docs/2.9/notes/cuda.html
- PyTorch 2.9 Linear / LayerNorm API：
  https://docs.pytorch.org/docs/2.9/generated/torch.nn.Linear.html
  https://docs.pytorch.org/docs/2.9/generated/torch.nn.LayerNorm.html
- CUDA12.4.1 Best Practices：
  https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-best-practices-guide/index.html
- CUTLASS grouped scheduler（仅借鉴persistent任务分配；不是WAM完整DAG实现）：
  https://docs.nvidia.com/cutlass/4.2.1/media/docs/cpp/grouped_scheduler.html

本轮顺序：真实原block参考 -> 已对齐attention混合block -> 显式顺序B0 -> 精度/状态/性能范围收口 -> 必要device原语 -> 后续另行授权的scheduler。
