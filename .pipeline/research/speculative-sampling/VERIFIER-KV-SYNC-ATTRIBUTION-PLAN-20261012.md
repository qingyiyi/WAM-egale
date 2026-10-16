# 投机采样 verifier：共享历史、重复物化与同步的最小归因方案

2026-10-11夜间提出、2026-10-12整理（Asia/Shanghai）。**本步仅CPU制定方案、读取既有结果并追加HW；未实现实验包装/mega Transformer，未启动GPU、训练或新队列。非tile算子实验。** 静态核查结果见同名证据目录；静态通过不等于新包装CPU回归或GPU正确性已经通过。

## 1. 当前事实及本步问题

既有固定负载任务已完成：0完整episode、2次EGL、168固定chunk、4118.325秒。正式OFF动作分支中，长历史单需求A_off较T慢4.7%/11.2%，双需求慢30.2%/28.2%；所以本轮以**长历史单需求**为主，双需求只检查同卡并发压力。单双需求同fixture的接受路径/工作计数一致；不能把双需求额外慢化解释成接受率下降。

长单两份ON诊断中，verifier占动作时间81.96%/83.65%，draft约2.20%/2.19%，tree scheduler约0.98%/1.04%。这是带同步诊断的线索，不是纯GPU kernel占比；tree_build包含draft/scheduler，不能相加。视频另约1.93秒，未计入动作分母。我们尚未隔离KV复制、同步与Transformer数学各自的真实净贡献。

需要回答：①每层重复物化历史是否值得优先消除；②不必要的元数据计算/host同步是否值得外提；③剩余是否主要在attention/GEMM/FFN/文本投影，才考虑更大系统优化。**不通过本轮改变候选树、CFG、阈值、scheduler、bonus、模型权重或训练目标。**

## 2. 用户追问：已经batch验证，为什么有14份？

确实是batch验证：K=2、B=2树有 `1+2+4=7` 个逻辑节点；CFG条件/无条件两路形成14行。**一棵树一次大模型verifier前向，不是7次或14次串行target前向。** 一次batch内部仍有30层及多个CUDA kernel，也不能把“一次前向”说成“一次GPU kernel”。

服务器映射为 `[0]*7 + [1]*7`：前7行共用条件历史，后7行共用无条件历史；每个候选的当前Q/K/V不同。现在每层的实现是：

```text
两路持久历史 KV [2, Lhist, H, D]
  → mask.nonzero + 有效槽 gather
  → index_select(cache_rows)：物化 [14, Lhist, H, D]
  → cat(各候选当前16个token)：[14, Lhist+16, H, D]
  → 一次批量 attention，14行互相隔离
```

因此，“14份”是同一个batch内部的**历史KV存储副本**，不是14份独立历史，更不是14次模型调用。每一路历史复制7份，两路合起来14份；相对原来的两路数据是7倍，而不是14倍。batch并不会自动将相同前缀去重。不同候选的Q需要分别和历史交互，**共享存储也不等于数学工作缩到1/7**。

真实元数据为BF16、24头×128维：Q=[14,16,24,128]，长K/V各=[14,2160,24,128]，其中历史2144、当前16。K+V两路历史payload为50.25MiB，扩成14行后351.75MiB，再cat后的输出为354.375MiB。短历史对应272+16，扩展历史44.625MiB。以上只是张量输出字节，**不是实测HBM流量、峰值显存或耗时**；不能把几种payload直接相加当峰值。

真正共享前缀应保留两路历史并提供候选行映射，同时维持各候选当前KV的隔离。仅写`expand`可能仍被后续`cat`物化；把7个候选直接拼成一个序列会引入错误的跨候选attention。分段历史/当前attention也需要正确合并softmax归一化，不能直接相加。本轮不实现这些新attention接口或kernel。

## 3. 最小实验：先用四臂区分“重复物化”与“元数据/同步”

在**一个真实长历史verifier的第0层self-attention输入**上做固定负载微对照。模型/SDPA backend、dtype、shape、stride、Q/当前K/V和原历史槽布局均冻结；不使用全模型30层持久展开缓存。

| 臂 | 元数据/检查 | 历史有效槽gather及14行展开 | 保留的工作 |
|---|---|---|---|
| R原始 | 每次原样nonzero/行范围检查 | 每次原样gather/index_select | cat及原批量SDPA |
| H外提 | 每个29调用序列只预计算/检查一次 | 每次仍重新物化 | cat及同一SDPA |
| C展开缓存 | 每次仍保留原动态元数据/检查；不添加新OFF检查 | 序列开始展开一次、后续复用 | cat及同一SDPA |
| HC组合 | 预计算/检查一次 | 展开一次、后续复用 | cat及同一SDPA |

**所有准备成本计入主sequence耗时**，不得给H/C/HC免费预热历史。C首次构建需要额外元数据准备也收费；它刻意保留每次动态检查，以构成控制臂。记录每臂实际metadata/gather/index_select/cat调用计数，防止混入未申报优化。

H测的是元数据/检查/同步的组合成本，不声称差值就是“纯同步秒数”。C测的是重复gather/展开是否值得消除，**C仍存14行历史，不是最终两路共享KV实现**，也仍每次cat。H/R、HC/C看外提效果；C/R、HC/H看重复物化效果；HC/R看组合；报告交互，不能把两项收益简单相加。

C在第0层仅额外常驻长历史351.75MiB/worker。若贸然全30层常驻展开，会达10.305GiB/worker、双worker20.61GiB，可能改变显存竞争甚至OOM，因此不在最小实验里实施。如果C有稳定净效果，再单独规划真正的共享前缀/去cat方案及完整verifier等价性。

## 4. 输入、负载、顺序和计数

### 4.1 复用原长fixture，不生成新episode

仅用原任务 `T-init3-chunk14.pt`（1,534,661,387字节）及同名JSON；快照有30层、frame56、有效history2016。先按已修复的原reset→A_off预加载顺序恢复，重放保存请求的视频分支后，首次candidate的有效history才是2144。**2016不是本轮attention最终历史长度。**

旧结果保存了shape/状态摘要，但**没有保存可直接重放的candidate原始QKV**。未来须先实现私有、默认关闭的捕获包装：从快照做有界partial replay，21个video前向（20更新+terminal）、1个action anchor、1个首次candidate，共23个target前向/worker；完整跑完这次candidate再在首次tree提交前停止。不得为方便改成完整action chunk，不调用LIBERO/EGL/render，不执行任何环境动作。

捕获第0层已经投影/RoPE后的Q、当前K/V、原历史pool/mask/row map，保存输入与原attention输出用于等价检查；只读candidate不得写持久缓存。不把30层全部展开或导出大buffer。新包装和capture运行均**尚未实现、尚未验收**；如果真实接口不能在此合同内完成，就停止重新申请，不补跑完整chunk。

### 4.2 四种固定负载

- 主负载：同一真实长QKV，历史2144、单需求；两模型驻留，peer空闲。
- 辅助长度控制：保持同一长fixture的Q/当前K/V、历史pool容量/槽布局，私有mask只选原有效顺序中最后272个历史槽，得到KV288。它是**合成局部长度对照**，不是短历史真实任务，不修改公共policy缓存、不评价成功率。
- 压力辅助：上述两种长度分别双需求；同一A100两个独立worker、相同fixture/臂，执行窗口重叠至少90%。不是树B=2、CFG两路、双GPU或T/A混跑。

每个sequence重复29次同一不可变QKV的attention准备+调用，对应既有长fixture的29棵树次数；**不是实际29棵树不断变化的QKV，更不是一个完整policy chunk**。第0层局部收益不乘30当全模型收益。初筛只有一个init，不能宣称跨状态稳定。

每个负载用四个平衡顺序，各臂4次主测：`R,H,C,HC`；`H,HC,R,C`；`C,R,HC,H`；`HC,C,H,R`。各臂每个位置一次，每个有序相邻配对一次。主测profile OFF，仅sequence边界同步，setup含在timer内；保留host wall及边界CUDA event，不能把event跨度称作纯kernel忙时。双需求启动门在timer外，等待时间另报，统计primary，peer不是独立样本。

| 用途 | worker-sequence数量 | 每sequence调用 | micro attention调用 |
|---|---:|---:|---:|
| 主测：4负载×4臂×4重复，双需求含peer | 96 | 29 | 2784 |
| 每个条件一次warmup，双需求含peer | 24 | 29 | 696 |
| 独立ON微诊断：仅长R/HC的单/双需求 | 6 | 29 | 174 |
| 每worker-condition一个等价调用和一个单行扰动隔离调用 | — | — | 48 |
| **micro attention硬上限** | **126个sequence，另48检查调用** | — | **3702** |

另做**原始完整candidate verifier**的诊断，不向全模型注入H/C：捕获46个target前向，2个candidate warmup（每worker一个），长单原始candidate trace1个、长双trace2个，共**51个完整target Transformer前向硬上限**。其内部attention与上表micro调用分别计账，3702不是所有模型内部attention的总上限。诊断Nsight trace与OFF计时分离，不用ON−OFF推算同步，不累加重叠kernel时间当wall time。

## 5. 未来GPU预算（提案，未获本次启动授权）

- **0新完整episode，0次EGL初始化，0新完整policy chunk；至多2个GPU模型worker。**
- target前向≤51；额外micro attention≤3702；四臂×两长度×单/双需求，不追加init或重复。
- 预计GPU占用15–25分钟；**安全硬限45分钟，从第一个模型worker启动计时，包含加载/捕获/诊断/必要比较与清理**，不含事前等待。预计时间不是保证，超时就停、不补跑。
- 资源门需重新确认独占/空闲且无未知持续负载，不能沿用历史独占口头结论；不启动后台排队、不干预其他进程。CPU内存按两份fixture及实际模型加载路径复核，至少10GiB余量；磁盘至少40GiB；两模型加载后GPU余量至少8GiB。micro附加峰值目标≤2GiB/worker；实际检查不满足就停。
- 原监督器属于168chunk/2EGL合同，不直接复用启动。未来新建任务专用计数/timeout/归属监督器，采用已修复私有ownership guard并重新CPU回归；只能清理自己的PGID/PID/端口。
- ON使用已有nsys的CUDA/CPU范围（不收集ncu硬件计数、不做多轮replay），trace总量≤1GiB；仅确认工具路径，不意味着profiling已运行。缺失/不兼容工具停在诊断关，不安装系统依赖、不把带ON结果伪作OFF。

## 6. 通过标准和失败行为

| 关 | 必须满足 | 不满足时 |
|---|---|---|
| 当前CPU方案核查 | 源码/既有metadata引用一致；14行映射、payload、平衡顺序、51/3702预算算术一致；保护文件不变 | 修订方案，不能称GPU通过 |
| 未来包装CPU预检 | 真实reset/预加载/partial-stop接口回归；默认关闭/输入只读；mask槽顺序、CFG行隔离、各臂调用账；预算/超时/owned-cleanup故障路径，无CUDA执行 | 不启动GPU |
| GPU恢复/等价/隔离 | 与捕获原attention及R输出逐bit相等；dtype/stride/backend一致；单行Q/当前K/V扰动不改变其他13行输出；有效history/metadata/RNG不变；不比较无效pool未定义字节 | 保留失败并停止，不放宽容差/换backend/补样本 |
| OFF有效性 | 每条件4个primary序列MAD/median≤7%；双窗口重叠≥90%；准备成本收费、全部raw和峰值保留 | 判无效/不确定，停止扩展，不加轮次 |
| 诊断完整性 | 覆盖原candidate全部30层；区分复制、同步/D2H、CPU间隙、attention、文本K/V、GEMM/FFN、cast；独立ON有明确scope | 不声称找到唯一瓶颈，缺失如实NOT_RUN |
| 结果和清理 | 正常/失败均按关写HW；owned进程/端口退出，保护及公共源码保持不变 | 未完成/失败，不称全部通过 |

**性能正结果不是有效性通过条件**：长单某项含准备成本的sequence中位改善≥5%，且四个顺序块同方向，才标“局部有价值的pilot”；否则有效负结果或不确定结果同样有意义。分别报告长度交互、单双需求交互；没有正式统计功效或闭环准确率保证。双需求收益不能覆盖主负载长单失败。

当前完成的是**静态CPU规划**。新包装CPU回归、GPU恢复/bitwise/成本/trace/清理全部待执行；本轮不能证明共享KV已加速，不能证明成功率、环境步数或E2E改善。

## 7. 除KV复制之外的重点嫌疑（尚未测定占比）

1. **批量行量扩张与未用候选工作**：长T为51前向/102 CFG行调用，A_off为32/412。forward减少但行量约4.04倍；不能把行量比例当FLOPs比例，也不在本轮删叶节点/改bonus。
2. **反复文本投影与cross-attention K/V**：真实prompt与negative embedding各[1,512,4096] BF16，candidate包装重复到7+7。model每次调用text_embedder；每层attn2重新执行to_k/to_v，cross-attention没有该历史缓存。文本token512远多于当前action query16，可能重要，但token数不能直接变成耗时占比。完整verifier ON优先给出这些GEMM计数/范围，若它们占主导，再规划文本KV复用，不自动改源。
3. **device→host同步与CPU小操作**：CUDA mask.nonzero、每层bool(all)行范围检查，server/scheduler的cpu/item，以及有限性检查/6条边误差item。T也有部分共同同步，不把所有同步都归于投机；H只隔离attention元数据一组，其余由完整trace另看。
4. **完整Transformer数学/类型转换**：QKV/output线性、FFN、cross-attention及norm/gating，RoPE当前还有float64/complex路径和回转。30层×14行会重复处理；本轮不改RoPE精度、不赌数值可忽略。
5. **可能可省但不在本轮改的CFG行**：本动作配置guidance scale=1，而视频仍使用CFG；不能直接从整个服务删负路历史/terminal/video工作。后续如单独测试，T和A都须公平优化、逐bit验收。
6. **并发资源、host dispatch和E2E固定地板**：双worker争GPU/带宽/CPU调度；视频约1.93秒另计。artifact hash/保存耗时在计时外，不算模型推理。局部微对照不冒充服务最优吞吐或自然episode时延。

## 8. 按结果分支，不自动实现下一步

- C/HC长单稳定有净改善：优先另行规划真正共享两路前缀、去cat及完整candidate等价；不要将全30层14行展开常驻当最终方案。
- H/HC改善、C无改善：优先元数据/同步外提；完整trace再核对tree/scheduler等其他CPU等待。
- 四臂基本无改善，数学/文本投影占主导：先规划文本KV复用或GEMM/FFN/attention执行优化；只有证据支持才进入mega Transformer。
- 仅双需求有差异：结论是压力敏感性，不替代长单部署收益；再单独规划调度/准入。
- 任何错误/不稳定/超限：记录原因停，不改阈值、不补局、不训练。

**下一步：用户先确认本合同及私有默认关闭包装/CPU回归的实现范围；包装通过后，再单独确认45分钟GPU启动预算。本步没有申请立即启动，也不继承上一168chunk任务的授权。** 后续系统收益成立后，仍需另行检查完整policy及冻结target闭环的成功率、环境步数和E2E；RL/KL、联训与Fast-WAM迁移保持未授权候选。

## 本地证据

- [完整固定成本结果](FIXED-LOAD-COST-RESULT-20261011-213036.md)
- [上步系统解释](VERIFIER-SYSTEMS-FASTWAM-DUAL-DEMAND-20261011.md)
- [CPU规划核查](../../../review/OMISSIONS.md)
- [机器可读提案](../../../review/OMISSIONS.md)
- [attention/model源码](../../../投机采样/wam-p0-worktree/wan_va/modules/model.py)：302–331、608–660、774–793、1074–1098。
- [CFG batch及tree server](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)：747–779；[树构建/打包](../../../投机采样/scripts/p0_batched_verify.py)：506–596。
