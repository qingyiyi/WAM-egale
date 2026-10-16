# Debug-011：Phase 2B matched closed-loop 收尾

- 时间：`2026-10-10 15:55（Asia/Shanghai）`
- 状态：**完成（CPU-only 后处理）**
- 范围：WAM action 分支、`libero_10/task0`；不涉及 video 分支
- 本轮未启动 GPU、WAM/LIBERO 服务、eval、rollout 或训练；未停止任何外部服务；未改写原始 runtime 数据。

## 1. 要回答的问题

比较 Target-only 与线上 K=2/B=2/global speculative verifier 在同一任务、同一 init 上，是否同时减少环境步数并改善端到端时间；同时确认现有 Phase 2B 原始 chunk-event 的完整性，避免把缺失或错误 provenance 当成加速证据。

## 2. 实验合同与来源

- 正式 campaign：action-only、`libero_10/task0`、同 checkpoint/seed 条件、Target-only 对比 `K=2/B=2/global`。
- 使用完整可用 rollout budget（LIBERO horizon 配置为 800）；没有 smoke、one-chunk 或 max-policy-chunk 截断。成功时允许按任务终止条件提前结束，所以 episode manifest 中 `horizon_reached=false` 不等于人为截断。
- 两臂均开启 EGL；live audit 检查到 NVIDIA EGL library 已加载、`MUJOCO_GL=egl`、`PYOPENGL_PLATFORM=egl`、EGL device 0。Transformer 与 VAE 均在 `cuda:0`，text encoder 在 CPU。
- 每个算法条件由两个独立 worker 并行执行，worker0/worker1 分别负责 init4/init5；两种算法条件按 campaign 串行，而不是同时争用单张 A100。
- 不训练 draft model；trace/profile 关闭。因 trace off，本轮不具备 verifier decision-level 时间拆分或可重新估计的 accepted-depth/L2 数据。
- 正式输入共四个 campaign：
  - Target-only init0–3：`p0-libero-libero_10-task0-init0-3-targetonly-gpuvae-egl-par2-20261006`
  - K2 init0–3：`p0-libero-libero_10-task0-init0-3-k2b2-global-gpuvae-egl-par2-20261007`
  - Target-only init4–5：`p0-phase2b-task0-init4-5-targetonly-gpuvae-egl-par2-20261009`
  - K2 init4–5：`p0-phase2b-task0-init4-5-k2b2-global-gpuvae-egl-par2-20261009`

## 3. 当前 matched init4–5 结果

两个 campaign 都是 2/2 completed、2/2 success、0 error、0 artificial truncation。配对 key 精确为 `libero_10/0/4`、`libero_10/0/5`。

| 指标 | Target-only | K=2/B=2/global | K2−Target | 相对变化 |
|---|---:|---:|---:|---:|
| E2E elapsed (s) | 366.521 | 378.859 | 12.338 | 3.37% |
| Environment steps | 284.5 | 287.5 | 3.0 | 1.05% |
| Inference chunks | 18.50 | 19.00 | 0.50 | 2.70% |
| Reset request (s) | 128.937 | 124.403 | -4.534 | -3.52% |
| Policy request (s) | 207.486 | 224.728 | 17.242 | 8.31% |
| Cache request (s) | 9.853 | 10.460 | 0.606 | 6.15% |
| Model request total (s) | 346.276 | 359.590 | 13.314 | 3.84% |

逐 init：

| Init | Target E2E (s) | K2 E2E (s) | Delta (s) | Target steps | K2 steps | Target/K2 chunks |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 371.102 | 390.437 | 19.335 | 299 | 302 | 19/20 |
| 5 | 361.940 | 367.280 | 5.340 | 270 | 273 | 18/18 |


## 4. 六个 matched init 的描述性合并

合并既有 init0–3 与本次 init4–5，得到双方各 6 个完整成功 episode。此合并只作描述性方向判断，不是统计显著性结论，也不代表跨 task 的泛化结果。

| 指标 | Target-only | K2/B2/global | K2−Target | 相对变化 |
|---|---:|---:|---:|---:|
| E2E elapsed (s) | 369.085 | 372.355 | 3.270 | 0.89% |
| Environment steps | 279.2 | 305.3 | 26.2 | 9.37% |
| Inference chunks | 18.17 | 20.00 | 1.83 | 10.09% |
| Reset request (s) | 133.310 | 126.974 | -6.336 | -4.75% |
| Policy request (s) | 205.776 | 214.057 | 8.281 | 4.02% |
| Cache request (s) | 9.895 | 10.713 | 0.818 | 8.27% |
| Model request total (s) | 348.981 | 351.744 | 2.763 | 0.79% |

两臂均为 6/6 success、6/6 completed。当前 6 个 init 中，K2 E2E 平均慢 `+0.89%`，environment steps 多 `+9.37%`，chunks 多 `+10.09%`。不把小样本下的时间波动解释为显著差异。

## 5. 原始 chunk event 与 provenance 审计

当前 init4–5 两个 campaign 均完成 native event-level audit：

- Target-only：2 个 worker event 文件、37 条 chunk event；
- K2：2 个 worker event 文件、38 条 chunk event；
- 检查通过：必需字段存在；event ID 唯一；campaign/episode/attempt/worker provenance 一致；chunk index 从 0 连续；env-step 与 simulator-timestep 的 before/after/delta 自洽、chunk 间连续；每 episode 的事件数与 manifest `infer_chunks` 一致；最终 env steps 与 manifest 相符；最后 chunk `done_after_chunk=true` 且有非零 reward；manifest join 精确。
- worker provenance 修正为 `worker0`、`worker1`（旧 helper 曾用 `path.parent.parent.name` 错误地产生 `workers`）。两名 worker 各自对应 init4/init5，launch assignments 不重叠；live EGL/device audits 与两个 parallel merge manifest 均完整。
- init0–3 旧 campaign 没有独立的 native `chunk_events.jsonl`，所以只进行了 episode-level、配对 key 与 campaign provenance 审计；没有把旧 episode 结果冒充为 chunk-level event audit。

## 6. 结果解释：哪里慢了，证据到哪一步

1. **当前 pair 可直接量化的部分：** reset request 平均减少 `4.534 s`，但 policy request 增加 `17.242 s`，cache 增加 `0.606 s`，model request 总量增加 `13.314 s`；E2E 最终增加 `12.338 s`。轨迹还有平均 `+0.50` 个 inference chunks 与 `+3.0` env steps。
2. **为何 speculative 理论上未转成加速：** draft 计算没有替代大模型 target verifier；在线路径仍需 target verification，并引入候选构造、selector/fallback 等控制流。Phase 0–1/旧 trace 与源码审计支持这条机制链。但本轮 trace 关闭，`policy_infer_wall_sec` 是聚合数，无法把多出的 `17.242 s` 进一步定量分配到 verifier、tree、selector 或 fallback。因此后几项是受既有审计支持的机制解释，不是本轮分别测出的时间份额。
3. **reset/cache 不是本次 E2E 变慢的主要正向增量项：** reset 在当前 pair 反而更低；cache 虽增加约 `0.606 s`，明显小于 policy request 增量。主要已观测成本信号是 aggregate policy request 上升和轨迹略长。
4. 所有 episode 都成功，所以本数据不能证明 K2 质量更差；它只表明目前没有达到“步数持平或更少且总时间更快”的目标。当前点估计方向对 speculative 线上扩展不利。

## 7. 对 L2、gate 与 predictor 的结论

- L2 仍仅作为 verifier-local 的偏移/accept-reject 解释特征，不作为 task-level success、environment steps、长尾或 E2E 的替代指标，也不作为唯一 gate 目标。
- 当前 campaign trace off，无法重估 verifier-local L2 与 accepted-depth 的关系；原生 chunk events 只有 sparse reward/done/timestep/env-step，没有连续 goal-distance/progress。
- Phase 2A 仅有 10 个独立 episode 的历史信号审计也未支持安全门控或 long-tail predictor。现阶段**不训练 draft、不训练 predictor、不启用 safety gate**。

## 8. 阶段结论与下一步

### 已完成

- Phase 0–1：因果链/schema/instrumentation 审计；历史数据关联覆盖 10 episode、171 chunks、4,600 verifier decisions；L2 保留为 verifier-local explanatory feature。
- Phase 2A：轻量历史信号审计；accepted depth/fallback 无稳定长尾解释力，目标距离字段缺失，样本不足以建 gate/predictor。
- Phase 2B：完成 full-budget/no-artificial-truncation 的 action-only matched closed-loop 配对收尾，并完成 init4–5 原生 event audit。Target-only 与 K2 均 2/2 success，但 K2 没有端到端加速。

### 尚未完成 / 建议顺序

1. 默认恢复 Target-only 为当前速度/质量基线；不启动大规模 eval、长时间训练或 video 分支。
2. 仅做代码级 verifier budget/control-flow 静态优化：减少 candidate rows、复用 root/target 已计算输出、提高一次 verifier forward 对有效 scheduler steps 的覆盖率；先明确候选优化确实减少 target work，而不是只移动开销。
3. 若代码路径和单元/合同验证显示有真实 verifier 工作量减少，再做**最多一个极小 matched full-episode pair**；仍须 action-only、EGL、GPU-VAE、并行 worker、两个条件串行、完整可用 horizon、无 smoke/截断，并检查资源归属。
4. 若新的 matched pair 仍不能稳定实现 environment steps 不增且 E2E 下降，则封存 speculative rollout 路线；保留负结果、瓶颈机制和局部实现证据用于论文，不声称已证明 speculative acceleration。

## 9. Runtime 与受保护状态完整性

- Runtime baseline：`runtime_hash_before.json` 于 `2026-10-10T01:04:24+08:00` 捕获，四 campaign 共 807 files、376,826,106 bytes。
- CPU-only finalizer 之后：仍为 807 files、376,826,106 bytes；文件路径、大小、SHA-256 全部相等，`runtime_unchanged=true`。
- 本轮只生成分析目录的 summary/provenance/paired CSV/README/runtime hash，以及本报告和 Hypo-Workflow 记录。
- `.pipeline/state.yaml` 与 `.pipeline/rules.yaml` 未修改；`.pipeline/cycle.yaml` 原本不存在，本轮没有创建。

## 10. 主要产物

- 分析 README：`投机采样/results/analysis/p0-phase2b-matched-closed-loop-20261009/README.md`
- Summary：`投机采样/results/analysis/p0-phase2b-matched-closed-loop-20261009/summary.json`
- Provenance：`投机采样/results/analysis/p0-phase2b-matched-closed-loop-20261009/provenance.json`
- 当前配对表：`投机采样/results/analysis/p0-phase2b-matched-closed-loop-20261009/paired_episode_metrics.csv`
- 六 init 配对表：`投机采样/results/analysis/p0-phase2b-matched-closed-loop-20261009/paired_episode_metrics_all_six.csv`
- Runtime 哈希核验：同目录下 `runtime_hash_before.json`、`runtime_hash_after.json`、`runtime_hash_comparison.json`
