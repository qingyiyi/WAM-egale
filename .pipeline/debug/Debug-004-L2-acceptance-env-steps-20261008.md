# Debug-004：局部 L2、verifier acceptance 与环境步数的关系

日期：2026-10-08（Asia/Shanghai）  
范围：**仅 action branch**；线上协议仍为 `draft K=2 → real target/verifier → accept/rollback`；video branch 延期。  
状态：**离线关系分析完成；没有启动训练、LIBERO rollout、smoke 或新的 GPU 服务。**

## 1. 用户问题与判断标准

当前不应把“draft 与原始 target 的 L2 更小”直接等同于“任务更好”。本轮把指标拆成三层：

```text
local prediction error
    → verifier acceptance / accepted prefix / rollback
    → 实际 action trajectory
    → task success 与总 environment steps
```

真正的晋级目标改为：

1. task success 不低于 target-only；
2. 配对 episode 的总环境步数不高于 target-only，且不出现严重长尾；
3. verifier/tree 预算不恶化到抵消收益；
4. 最后才判断端到端 wall-clock。

原始方向/latent L2 继续保留，但只作为 local-fidelity diagnostic，不作为唯一 gate。

## 2. 数据与分析边界

数据来自：

- `/rjs/cxz/论文探索/投机采样/results/analysis/p0-d-compare-candidate-shadow-20261008-r1/pooled_candidate_shadow_result.json`
- `/rjs/cxz/论文探索/投机采样/results/analysis/p0-d-compare-candidate-shadow-20261008-r1/worker-task8/candidate_shadow_chunks.jsonl`
- `/rjs/cxz/论文探索/投机采样/results/analysis/p0-d-compare-candidate-shadow-20261008-r1/worker-task9/candidate_shadow_chunks.jsonl`

覆盖：

- 75 个完整 rollout chunks；
- 1,800 个 paired anchors；
- 4 个 draft variant，共 7,200 条 variant-anchor 观测；
- action-only、K=2、真实 candidate-state target forward；
- 不使用 ZOH gate；
- acceptance threshold 为 `0.007839680526450805`，语义是历史真实 verifier replay 的**绝对 successor-L2** q90 calibration；
- shadow replay 不执行真实 policy rollout，因此不含 task success、真实环境步数或 E2E 时间。

本轮同时读取正式 matched episode：

- target-only：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-libero-libero_10-task0-init0-3-targetonly-gpuvae-egl-par2-20261006/client/episodes.jsonl`
- K2：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-libero-libero_10-task0-init0-3-k2b2-global-gpuvae-egl-par2-20261007/client/episodes.jsonl`

## 3. 结果一：L2 与 acceptance 有关系，但这是 verifier-local 关系

### 3.1 绝对 successor-L2 是当前 gate 真正使用的量

在 pooled 数据上，`first successor-L2` 与 acceptance/prefix 的 Spearman 相关为：

| 误差指标 | first accept | both accept | accepted prefix |
|---|---:|---:|---:|
| first successor-L2 | -0.4665 | -0.5233 | -0.5239 |
| second successor-L2 | -0.4653 | -0.5247 | -0.5251 |
| candidate-state relative-L2（诊断） | -0.4653 | -0.5202 | -0.5210 |

按 variant 分层，`first successor-L2 → both acceptance` 的 Spearman 约为 `-0.503` 到 `-0.551`。这说明误差较大时，接受两步的概率确实下降；但这只证明**当前 verifier 的局部判定关系**，不能推出环境步数一定下降。

当前 threshold 是绝对 successor-L2，而不是 normalized relative-L2。因此后续如果要优化 draft，优先对齐：

```text
post-scheduler / post-action-projection 的 active-channel absolute successor error
```

不能只看未经 scheduler 缩放的 direction L2，也不能只优化 relative-L2。

### 3.2 acceptance 存在明显的 threshold cliff

把所有 variant-anchor 按 candidate/first absolute successor-L2 分箱后：

| absolute successor-L2 区间 | first accept | both accept | mean prefix |
|---|---:|---:|---:|
| ≤ 0.003293 | 100.00% | 100.00% | 2.0000 |
| 0.003293–0.006378 | 100.00% | 95.37% | 1.9537 |
| > 0.006378 | 21.25% | 4.72% | 0.2597 |

因此“把已经小于 threshold 的平均 L2 再降低一点”大概率不会减少环境步数；真正有价值的是把**高误差尾部、尤其是 threshold 附近和 above-threshold 的样本拉回到可接受区间**。

## 4. 结果二：early denoising 是主要误差/acceptance 瓶颈

| denoising phase | phase4 MLP both | phase4 Transformer both | PhaseA MLP both | PhaseA Transformer both |
|---|---:|---:|---:|---:|
| early | 69.83% | 66.50% | 71.83% | **72.83%** |
| middle | 99.83% | 99.83% | 100.00% | 100.00% |
| late | 99.17% | 99.17% | 99.17% | 99.17% |

PhaseA Transformer 相对 phase4 Transformer 在 early 阶段的：

- first successor-L2：`0.00928092 → 0.00876163`，约下降 5.59%；
- second successor-L2：`0.01030490 → 0.00912608`，约下降 11.44%；
- both acceptance：`66.50% → 72.83%`，提高 6.33 个百分点。

但 pooled 到所有阶段后，PhaseA Transformer 相对 phase4 MLP 的 both acceptance 只从 `89.61%` 到 `90.67%`，accepted prefix 只从 `1.81944` 到 `1.83111`。原因是 middle/late 已经接近饱和，平均指标会掩盖 early tail 的差异。

**含义：** 如果继续降低误差，应该做 phase/timestep-aware 的 hard-example 优化，优先处理 early denoising 和 threshold 附近样本，而不是均匀压低所有样本的平均 L2。

## 5. 结果三：目前没有证据证明 local L2 会降低总环境步数

正式 matched episode 的现有结果为：

| init | target env steps | K2 env steps | K2 − target |
|---:|---:|---:|---:|
| 0 | 269 | 441 | **+172** |
| 1 | 272 | 270 | -2 |
| 2 | 265 | 256 | -9 |
| 3 | 300 | 290 | -10 |

两边 success 都是 `4/4`。汇总：

- target-only mean：`276.5` steps；
- K2 mean：`314.25` steps；
- K2 − target：`+37.75` steps；
- 总差：`+151` steps；
- 配对差中位数：`-5.5` steps；
- K2 的 E2E 仍只比 target-only 快 `1.263368s`，约 `0.34%`，应视为基本持平。

这里最重要的现象是：K2 相对 target-only 的 `Δ inference chunks` 与 `Δ env steps` 在这 4 个配对上的相关约为 `0.9992`。init0 的 `+172` 步伴随 `+10` 个 inference chunks；其余三个 init 的 chunks 数没有增加，环境步数也没有增加。

这支持当前的首要解释：

```text
轨迹是否变长 / 是否需要更多 policy chunks
    比单个 anchor 的 L2 小幅变化更直接地决定总环境步数。
```

但是这 4 个 episode 的 episodes.jsonl 只保存 episode 汇总，没有把 shadow anchor、online accepted depth、rollback、fallback 与环境 timestep 一一关联。因此现在**不能**声称“PhaseA Transformer 的局部 L2 改善会减少 env steps”，也不能声称 init0 的长轨迹一定由某个具体 L2 样本造成。

## 6. 决策：先找关系，再做有目标的误差优化

### 不建议直接做的事情

- 不再以“整体平均 L2 更小”作为 draft 晋级条件；
- 不再以是否超过 ZOH 作为唯一判断；
- 不在没有 trajectory/acceptance 账本的情况下继续跑两天级别的训练或大规模 LIBERO eval；
- 不把 shadow acceptance proxy 直接写成 task success 或环境步数收益。

### 建议保留的误差优化方向

在完成 online instrumentation 后，如果确认高误差 tail 与长轨迹相关，再考虑：

1. **early-phase / timestep weighted loss**：提高 early denoising 以及 threshold 附近样本的权重；
2. **post-scheduler active-action loss**：直接监督 scheduler 更新后的 active action successor，而非只监督 raw direction；
3. **hard-negative / near-threshold mining**：重点训练 absolute successor-L2 接近或超过 threshold 的样本；
4. **task-progress-aware auxiliary signal**：记录每个 policy chunk 的环境进度、accepted depth、rollback/fallback 和下一次 chunk 是否增加；
5. **confidence / abstention**：draft 不能确定时主动少提议或回退，避免一次错误候选导致长尾 episode；
6. 继续遵守真实 target verifier 合同，draft 不能替代 target。

### 当前新的硬 gate

候选 draft 只有同时满足以下条件才进入下一轮：

- success rate 不低于 matched target-only；
- aggregate env steps 不高于 target-only；
- mean env steps 不高于 target-only；
- 不允许单个 episode 出现类似 `+172` 的 catastrophic long-tail；
- accepted depth/rollback/fallback 与 verifier target-row budget 可解释；
- 最后才比较 E2E wall-clock。

## 7. 下一步计划与成本

### Step 1：完整 horizon instrumentation pair（需要源码改动确认）

仅加入诊断计数和阶段时间，不改变 K=2 的 accept/rollback 语义，记录：

- 每 chunk 的 draft/tree/verifier forward 数；
- logical/CFG candidate rows；
- ordinary target、terminal target、fallback 数；
- accepted depth、rollback ratio、selector/tree-build/pack 时间；
- reset/cache 子阶段；
- episode timestep、chunk 边界、每 chunk 环境步数；
- online candidate error 的均值、p90、above-threshold count。

### Step 2：用已有 4 个 matched init 重跑完整 episode

不做 smoke、one-chunk、dry-run 或人为截断；保持 EGL、GPU-VAE、双 worker/disjoint shards、隔离端口、resume，算法条件在单 A100 上串行。优先跑 target-only 与 PhaseA Transformer K2，必要时加 Phase4 MLP 对照。

这一步的目标不是重新训练，而是建立：

```text
per-chunk error/acceptance
→ accepted depth/rollback/fallback
→ chunks / env steps
→ E2E
```

按已有 4-init 运行量粗估，两个条件完整重跑约需 **45–60 分钟**，实际以当前服务启动和 episode 轨迹为准；不会启动本轮任务，直到源码改动和评测执行分别得到明确授权。

### Step 3：再决定是否训练

- 如果 high-error tail 与长轨迹显著相关：先做小规模离线 task-aware loss ablation，再进行短列表完整闭环验证；
- 如果 acceptance 与 env steps 关系弱：停止继续压 L2，转向 verifier budget、轨迹控制和 confidence/abstention；
- 如果 env steps 持平/减少但 wall-clock 仍不降：优化 verifier rows/tree build/cache，而不是继续训练 draft。

## 8. 当前运行态

本轮只做 CPU/offline 分析：

- 未启动训练；
- 未启动 LIBERO rollout；
- 未启动 smoke/截断实验；
- 没有新增 GPU 服务；
- 当前 GPU 仍按上一轮核验为无本项目 compute app、显存已释放。

## 9. 结论

用户提出的方向是正确的：**目标不是让 draft 在 L2 意义上尽可能像原始 target，而是让它在真实 verifier 合同下减少无效回退和长尾轨迹，同时保持任务成功。**

现阶段最合理的顺序不是“继续盲目减少平均 L2”，而是：

```text
先用完整 episode instrumentation 找到
local error → acceptance/rollback → trajectory length → env steps
的关系；
再针对 high-error early tail / near-threshold samples 做 task-aware error optimization；
如果关系弱，则停止以 L2 为中心，转向 verifier budget 与 trajectory control。
```
