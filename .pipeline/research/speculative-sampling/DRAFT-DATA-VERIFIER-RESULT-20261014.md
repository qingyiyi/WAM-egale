# Draft 数据、root 保留与 MegaTransformer 首版验证结果

> 最新夜间结果（2026-10-15T05:45:28+08:00）：45次训练无稳定新权重，保留A_off；30条开发质量任务、新TRAIN与PF2/3/4已独审。算子融合已撤回，原tile不改。见[统一报告](OVERNIGHT-DRAFT-ACCEPTANCE-RESULT-20261015.md)、[扩大准备](NEXT-SCALE-READINESS-20261015.md)，HW记录 CHAT-20261015-OVERNIGHT-DRAFT-ACCEPTANCE-RESULT。


本轮承接用户授权，使用多个 CPU 核及 GPU，完成历史数据重建、15 组有限训练、真实 verifier profile、CUDA 逐元素融合及六个完整 chunk 的数值检查。现用 A_off 保留：新模型局部指标改善，但没有一个配置通过预先冻结的多种子准入条件。新融合已通过完整 chunk 数值验收，正式性能等待独占窗口。

## 已完成的训练与数据工作

历史采集日志确认 98 条所选 train trace 实际加载 LIBERO-Long checkpoint 路径；client/run metadata 对应 init0。67,277 条历史 train 数据横跨四个 suite，仅重建其中 Long 的 4,802 条。实际右侧 `x_i` 的 compact112 投影与左侧 `x_next` **4,802/4,802 位级相等**；full480 不相等来自未使用通道，不能将 full-state 相等当作先决条件。原 A_off 使用的 833 条数据与 17 条原 trace SHA、行张量及标量逐项吻合。与当前 3,906 条 TRAIN 在状态、查询和标签等组合键上的重复为零。

新 scratch 训练可使用全部 4,802 条；旧 A_off continuation 的新增视图为 3,969 条。本轮使用 scratch，不把已用过的 833 条伪称新增。历史 node/head provenance 缺失的字段保留为 unknown；它们只提供历史教师标签，不冒充当前 CFG14/B14 同查询 oracle。历史权重路径有实际日志依据，无法追溯证明过去权重文件的字节 SHA。本轮未混入原留出集。

- [数据重建报告](../../../review/OMISSIONS.md)
- [原始重建与合并清单](../../../review/OMISSIONS.md)

固定 Transformer64、lr=1e-4、1,000 updates、batch512，三个固定 seed。先进行 9 组 A/B/C 消融，再进行 6 组当前/混合历史数据对照。归一化、准入门及验证集均不回改；CPU checkpoint/prediction 被独立重算。9 组训练准备及执行 583.38 秒，6 组 430.30 秒，二者独立审计完成。以下为三个 seed 的中位数，Euler 分数沿用生产接受标准：

| 条件 | Euler 全部通过 /1890 | Euler root 通过 /630 | 原 527 个 root 通过项丢失 | CPU 实际投影 root 通过 /630 | 所有准入门通过 seed |
|---|---:|---:|---:|---:|---:|
| 现用 A_off（参考） | 1458 | 527 | 0 | 374 | — |
| A：同数据基本 loss | 1483 | 515 | 21 | 366 | 0/3 |
| B：root/mid 加权 | 1495 | 524 | 15 | 370 | 0/3 |
| C：B + root 保留蒸馏 | 1498 | 529 | 10 | 391 | 0/3 |
| 混合历史数据 + C | 1509 | 533 | 9 | 399 | 1/3 |

加权因子仅从 TRAIN root 低/中 sigma 数量计算，为 1049/182=5.763736；保留蒸馏仅针对 TRAIN 上原 A_off 同 head 已通过的项，系数为 1。加入历史数据使用每 batch 256 current +256 historical，固定混合 loss；它同时减少 current 标签及 current KD 的训练占比。因此只能证明这套混合方案的净效果，不能把提升单独归因于数据量。

混合组 seed20260924 满足全部旧门，但配置仅 1/3 seed 通过，少于冻结要求的至少 2/3。配置中位数仍丢失旧通过 root 9/527=1.71%，超过原 1% 保留界限；部分分任务/root 误差门也失败。**不能挑一个 seed 替换现用模型，也不按结果放宽门。** 1% 是预先登记的工程条件，不是论文证明的成功率保证。

加入了使用原 `FlowMatchScheduler.step`、scatter/project/compact 及 BF16 的 CPU 实际投影评价。5796 行、1932 源查询和 82 个 step/frame 条件校验通过。A_off 的同阈值 root 数从 Euler527 变为 CPU实际投影374，说明两个量不能混称；生产接受标准仍为 Euler 加权方向差，本轮没有修改。实际 CPU 投影也不等于完整 GPU 递归 rollout。

当前验证集 1,890 行来自两个有关联的开发 episode，已经重复用于比较；这些数字不是新独立最终测试，不是成功率、递归接受率或轨迹长度保证。A_off 和新 scratch 模型的历史训练量不同。

- [9 组原始结果与独立审计](../../../review/OMISSIONS.md)
- [6 组历史混合结果与独立审计](../../../review/OMISSIONS.md)

## Verifier 的真实瓶颈与首版融合

原 profile v1 将 CPU/GPU 两类同名 marker 重复计数，首次真实 trace 在校验时失败；保留失败与费用。在独立 v2 中只选 CPU `user_annotation`，再执行 9 个完整 chunk、6 份 profiler trace、6 份真实 block0 输入/输出捕获。RAW/CURRENT/PROFILE_CURRENT 同算法输出、路由、RNG 以及现场全部30层 GPU KV 检查通过；50次更新完整闭合。独立 CPU 审计重读保存输出、9840 个实际 unique kernel、235 个 source SHA、参数和 capture metadata。

按唯一 CUDA kernel 归属、避免嵌套 inclusive 时间双计后：

- cross-attention 为 **54.40%–62.22%** 的累计 kernel duration，每个候选 verifier forward 约40.98–43.39ms；120个 cross GEMM 约35.64–37.91ms，主要在 K/V 投影。
- self-attention 随历史增长，从约8.3–8.9ms 增至18.6–19.1ms。
- FFN 约8.8–9.2ms；模块外 pointwise/cast/copy 约4.95–5.24ms。

这里是 GPU kernel duration 累计值，不是完整 forward/chunk wall time，也不能作为新 speedup。候选 forward 含 SharedForward preparation，但不涵盖全部 tree packing/lease/controller。原 prompt-KV 缓存已经在10/13实测变慢（短/中/长 +17.33%/+23.79%/+6.48%），因此不能仅由 cross 占比高推导缓存必定更快；下一步应单独验证 guard、克隆/物化和投影融合的成本。

首版使用两个隔离 CUDA primitive 替换 self/FFN modulation 与 gated residual 四处逐元素表达式，固定 CFG14/Q16/D3072，并保留 FP32 逐步舍入及 BF16 cast 顺序；norm、attention、GEMM、RoPE 都沿用原实现。原 tile 代码与历史 HW 未修改。这是最小原型，尚未完成 MegaTransformer 整体融合。

真实 GPU primitive 数值检查通过：24个捕获布局位置、24个 contiguous 版本、16个极端舍入/抵消/符号零等合成案例，以及10个边界/default-off拒绝检查；输入未改写、RNG不变。使用共享 GPU，仅检查数值，没有 micro timing 或新加速比。

- [唯一 kernel 归因审查](../../../review/OMISSIONS.md)
- [Profile 及 primitive 独立结果审计](../../../review/OMISSIONS.md)
- [CUDA 融合源码](../../../review/OMISSIONS.md)

## 完整 chunk 接入与正式性能的状态

共享六 chunk v1 在第一次完整推理之前因私有模块也叫 `adapter`，与冻结原 prefix adapter 导入名冲突而失败。失败没有数值结果，进程已清理，600 个未闭合预约按保守值计费。v2 将新模块改为私有 `block_fusion`，保留原适配器，真实 Torch CPU 导入及独立封存审查通过后启动。

v2 六个完整 chunk **PASS**：短/中/长实际 verifier KV 长度288/1152/2160，每档BASE_CURRENT/FUSED_CURRENT各一个。三对保存输出、接受路由、RNG、50次提交更新与canonical hash一致；GPU现场全30层有效KV数值逐字节检查通过。独立CPU审计比较480个张量、6个numpy数组、2442项标量metadata及235个冻结source SHA；融合eligible block共1830次、native共7320次，非候选前向按预期回原实现。方法恢复现场断言及adapter closed状态通过。

保存输出未包含KV数值，因此KV逐字节一致性由冻结GPU worker现场检查支持，不能声称已离线重载复验KV。该组在共享GPU运行，0个正式计时样本，不能据其wall time报告加速比；也没有完整LIBERO闭环实验。

- [完整 chunk 独立审计简报](../../../review/OMISSIONS.md)
- [机器可读完整 chunk 审计](../../../review/OMISSIONS.md)

旧42 chunk正式计时DRAFT同样存在通用导入名冲突，不能直接启动；原DRAFT与失败证据保留。已另建 `megaverifier_chunk_v2` 私有模块名fork，19项CPU检查和真实Torch CPU导入检查通过，独立审查后封存。新plan SHA `41f1189c6803f989434c886fea181ddf704dd7b6d62a33b55e30ca29333ed00f`，42 chunks、36正式样本、900秒/3000target上限，绑定微核及六chunk正确性和最新carry。未写GPU授权、未启动或排队；需独占窗口执行。

- [正式计时封存入口](../../../review/OMISSIONS.md)
- [正式入口独立代码审查](../../../review/OMISSIONS.md)

此前已验证的现用投机完整 policy chunk 加速比短/中/长分别 **1.602×/1.544×/1.500×**，该结果不含 reset。它来自此前的独占测试，不能乘上未测量的融合收益。

## 下一步与工作量

1. 算法线：保留 A_off；优先补充不同 init/任务上的 candidate-induced same-query teacher，并冻结新训练和留出协议。重新训练候选需要通过原 root/分任务/多 seed 门，再做配对闭环成功率、环境步数/惩罚步数与 trace 评价。历史混合改善只是开发证据，不启动没有质量依据的长时间全量训练。
2. 系统线：使用已封存的私有模块名正式 harness，在约10–20分钟独占窗口先测首版四处融合的完整 chunk 净收益；不加速则保留负结果。根据 profile 再对 cross K/V 重复投影、guard/物化成本提出具体消融，与已失败的原 prompt-cache 区分实现。
3. 可并行：teacher 数据与训练、CPU trace/代码分析、方法段落在会话逐段讨论。正式计时独占；质量及数值工作可共享 GPU。reset 继续后置，原 tile 研究独立保留。

最终累计GPU **31065.842105945572/36000秒**，target intents **147346/174494**，剩余约 **82.24分钟/27148 intents**。本轮新增GPU生命周期 **515.353703662752秒（8.59分钟）**，包含两次真实失败及全部清理；新增target **2796**，其中失败未闭合预约2100、成功精确闭合696。预算不会因修复重置。正式42chunk入口已继承最终carry，其900秒上限在剩余预算内。新的闭环采集仍需冻结规模，不能保证剩余额度覆盖完整全量训练及全基准质量证据。

本次shared六chunkGPU生命周期236.58280725218356秒、精确闭合282 intents。所有自有GPU root/worker均退出，GPU cleanup只留下获准共享的lhw PID2492517，未向其发送信号；本轮两个CPU训练任务及其独立审计均已结束。失败profile marker和完整chunk import记录完整保留，未退回预约费用。controlled micro在GPU争用门失败、0秒/0forward，也保留记录。

本轮原估60–90分钟，因真实导入失败与审计延长，最终墙钟耗时见 [HW-CLOSEOUT](../../../review/OMISSIONS.md)。只新增研究记录及导航，不覆盖历史记录。无论文初稿文件；未删除或压缩历史材料；未修改现用preset、原server/model、tile或state/cycle/rules/config。下一阶段仍有正式性能及闭环质量工作，未将整个研究项目标为完成。

最终保留审计 **PASS**（2026-10-14 22:42 +08:00）：217条旧Knowledge记录、47份原tile证据、13个保护路径原SHA一致；13个导航文件剔除本次明确新增片段后，原字节完全一致。新增1条研究记录，索引与本轮报告链接有效；所有冻结plan/source复核通过。执行manifest20:49:14至审计闭合22:42:04共 **112.84分钟**，加最初上下文准备，本轮约1小时55分钟。
