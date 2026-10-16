# Phase 2A：轻量离线信号审计

**时间：** 2026-10-09（Asia/Shanghai）  
**范围：** action-only、历史 K=2/B=2/global campaign 的离线信号盘点；视频分支不在本轮范围。  
**执行边界：** 仅 CPU、只读既有 CSV/JSON；未启动 GPU、训练、WAM/LIBERO 服务、LIBERO eval/rollout，也未运行 smoke、one-chunk 或截断实验。

## 1. 目标与判定标准

本阶段不是训练预测器，也不是证明投机采样的因果效果，而是先回答：已有数据中是否存在可以安全用于“长尾预测”或“安全门控”的可追溯信号。

任务级主结果仍定义为：

- episode success；
- 完整 episode 的环境步数分布及长尾；
- 完整 episode E2E 时间。

verifier-local cost/L2 只保留为解释接受/拒绝的局部诊断特征，不把平均 L2 当作任务级目标、长尾标签的替代物或安全门控阈值。

## 2. 数据来源与完整性

来源 campaign：

```text
p0-libero-libero_10-verifier-k2b2-global-1init-20260930-r2
```

来源分析目录：

```text
投机采样/results/analysis/p0-phase01-causal-tables-20261009/
```

本阶段生成目录：

```text
投机采样/results/analysis/p0-phase2a-signal-audit-20261009/
```

数据规模与完整性：

| 层级 | 数量 | 解释 |
|---|---:|---|
| episode | 10 | 当前唯一可视为独立 task/init 观察单位 |
| action chunk | 171 | 嵌套在 10 个 episode 中，仅作描述性诊断 |
| verifier decision | 4,600 | 嵌套在 chunk/episode 中，仅作 verifier 路由诊断 |
| success | 10/10 | 没有 success variation |

Phase 0–1 的历史表完整性为 **PASS**：episode、chunk、decision 的 join 和 env-step reconciliation 均通过；历史 fallback 按 K=2 的 `accepted_depth < 2` 规则推导，并与相应 target scheduler update 逐项核对，`1,378/1,378` 匹配。需要保留的历史兼容限制是：旧 trace 没有 raw fallback 字段、`episode_attempt_id`、原生 chunk event 或逐 chunk reward/done。

本阶段分析产物的生成时间为 `2026-10-09 11:57:45 +08:00`，其 provenance、schema、源文件 SHA-256 和边界均记录在同目录的 `provenance.json` 与 `schema.json` 中。

## 3. episode-level 结果：accepted depth/fallback 不能解释当前长尾

episode-level 是本阶段的主分析单位。环境步数统计为：

- 均值 `263.7`，中位数 `260.5`；
- P90（10 个样本的线性插值）`319.9`；
- 范围 `166–364`，总体标准差 `48.94`；
- E2E 均值 `482.84 s`，范围 `350.43–603.84 s`。

预先固定的描述性长尾分组为：

- `p90_or_above`：`env_steps >= 319.9`；
- `ge_300`：`env_steps >= 300`；
- `le_p10`：`env_steps <= 223.6`。

其中唯一达到样本 P90 的 episode 是 task 8：`364` env steps、`23` chunks、mean accepted depth `1.4984`、fallback rate `0.3000`。这些 verifier routing 数值处于普通范围，并没有与其他 episode 形成明显分离。最短 episode 是 task 5：`166` env steps、`11` chunks、mean accepted depth `1.5222`、fallback rate `0.3038`；它同样没有显示出一套可用于预测短尾的独立 routing 模式。

跨 episode 的范围也很窄：

- mean accepted depth：`1.4729–1.5223`；
- fallback rate：`0.2807–0.3145`。

因此，当前数据不支持把 accepted depth 或 fallback rate 作为长尾预测器，更不能根据这 10 个样本制定线上安全门控。

## 4. episode-level 关联：只说明轨迹长度记账，不是可用预测器

主要描述性关联如下：

| 信号 | 结果变量 | Pearson | Spearman | 解释边界 |
|---|---|---:|---:|---|
| mean accepted depth | env steps | -0.1095 | 0.0673 | 无稳定关系 |
| fallback rate | env steps | -0.3963 | -0.5915 | 方向不稳定、样本极小且 task-confounded，不能解释为 fallback 越多越快 |
| infer chunks | env steps | 0.9962 | 0.9752 | 主要是轨迹长度/固定 chunk 记账 |
| verifier decisions | env steps | 0.9956 | 0.9358 | 主要是随轨迹变长而增加的计算量记账 |
| target updates | env steps | 0.9773 | 0.8930 | 主要是轨迹长度记账 |

infer chunks、verifier decisions 和 target updates 与 env steps 高度同步，是因为当前协议的 action chunk 长度和 verifier 树结构基本固定。它们在 episode 结束后当然能解释总成本，但不能在 episode 过程中作为一个独立、提前可用的 task-progress predictor。

特别是 fallback rate 与 env steps 的表面负相关不能作因果解释：只有 10 个不同 task、没有 matched target-only 对照、所有 episode 都成功，且投机路由本身会改变后续状态。该相关性只能标记为不稳定的观察现象。

## 5. chunk-level 结果不能替代 episode-level 证据

171 个 chunk 并不是 171 个独立 task 样本，而是 10 个 episode 的重复测量。当前历史表还存在三个重要限制：

1. chunk env-step delta 是根据累计 `env_steps_before` 和 episode 最终计数器重建的，不是历史原生 `wam-p0-chunk-event/v1`；其中 `152/171` 个 chunk 的 delta 为固定 `16`。
2. 历史数据没有原生 per-chunk reward/done、可信连续 task distance 或 progress 字段。
3. episode success/env_steps 在 chunk/decision 表中是重复的 endpoint label，不能归因给单个 chunk。

因此 chunk-level 相关性只能帮助检查字段形状和数据覆盖，不能用来训练长尾模型，也不能宣称某个 chunk 信号改变了最终任务步数。

## 6. L2 的结果与严格边界

按 accepted prefix depth 分层，verifier-local cost 出现预期区分：

| accepted depth | decision count | first-edge mean | first-edge P90 |
|---:|---:|---:|---:|
| 0 | 907 | 0.01787 | 0.04043 |
| 1 | 471 | 0.00897 | 0.01449 |
| 2 | 3,222 | 0.00482 | 0.00730 |

这说明 local cost 能解释当前 verifier 的 accept/reject 分层，验证了它的 **verifier-local 语义**。它并不说明 L2 能预测 success、env steps、长尾或 E2E，也不说明继续压低平均 L2 会让闭环轨迹更短。

本阶段保留的定义是：

```text
L2-norm((draft_direction - target_direction) * transition_scale)
```

它是在同一候选状态和 transition 上比较 draft 与 target 方向的局部边代价。后续可以将其作为安全门控的一个解释特征，但必须先用真实配对闭环数据验证 outcome 风险；在此之前不把平均 L2 作为 gate 或优化目标。

## 7. Phase 2A 决定

### 不进入的方向

当前不做以下工作：

- 不用这 10 个 episode 训练 long-tail predictor；
- 不根据 accepted depth、fallback 或平均 L2 直接设计安全阈值；
- 不宣称 accepted depth/fallback 能减少环境步数；
- 不把旧历史 campaign 与后来的 EGL+parallel campaign 拼成伪配对数据；
- 不启动两三天规模的训练或全量线上 eval。

原因是：独立样本只有 10 个、全部成功、没有 task/init matched target-only control、没有原生 chunk reward/done，也没有可信 task-distance/continuous-progress 字段。

### 进入的方向

下一步只做一次小规模、完整 episode 的真实闭环诊断采集，目的是补齐因果链和候选风险信号，而不是立即训练：

1. 选少量 matched task/init，target-only 与 K=2 采用相同初始状态和完整 horizon；
2. 只采 action branch，保持 `draft → target verifier → accept/fallback → target committed action`，draft 不替代 target；
3. 正式采集强制 EGL + parallel eval，启动前验证 GPU、worker 数、server/master ports、trace 开关和 provenance；条件间在单 GPU 上串行，避免资源竞争；
4. 原生记录每个 chunk 的 reward、done、simulator timestep、env-step delta，并保留 `episode_attempt_id + rollout_chunk` join key；
5. 如果环境确实提供可靠的 task-progress/goal-distance，记录真实字段；若没有则保持 null，不从任务名、图像或 L2 猜测；
6. 预注册保守安全门控：风险升高或信号未校准时回退 target，先看 false-accept/false-risk，再看是否值得线上启用；
7. 评价顺序为 success → env-step mean/median/P90/P95/max → E2E；L2 只作解释性特征。

该下一步仍应是“轻量配对诊断采集”，不是 smoke、one-chunk 或截断 rollout；每个 episode 必须完整结束，并保留已有结果、不覆盖旧产物。

## 8. 验证与产物

已完成：

- `python -m py_compile 投机采样/results/analysis/p0-phase2a-signal-audit-20261009/audit_phase2a_signals.py`；
- `python audit_phase2a_signals.py` 已生成全部 Phase 2A 表格；
- 源表 integrity PASS；
- Phase 2A 输出含 `README.md`、`summary.json`、`provenance.json`、`schema.json`、episode/chunk/association/tail/local-cost CSV。

本轮没有修改原始历史表，也没有修改受保护的：

```text
.pipeline/state.yaml
.pipeline/cycle.yaml
.pipeline/rules.yaml
```
