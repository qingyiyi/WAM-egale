# Debug-003：K=2 投机采样端到端瓶颈与 GPU 运行态收尾

日期：2026-10-08（Asia/Shanghai）  
范围：**仅 action branch**；video branch 继续延期。  
状态：已完成证据分析与只读运行态核验；本轮没有启动训练、评测、rollout、smoke、one-chunk 或服务重启。

## 1. 结论先行

当前 K=2 并不是“draft 小模型替代大模型”，而是：

```text
draft 提议 K=2 的候选 scheduler transitions
    → 真实 target 在候选树上做 verifier forward
    → selector 决定 accept / rollback
    → 拒绝分支仍由 target fallback 修正
    → chunk 末尾仍有 terminal target forward
```

所以，单 chunk 中 target action update 从 50 降到 2，并不等于 target transformer 只执行 1 次。当前实现每个 K=2 tree 仍会执行一次 batched target verifier；稳态 profile 中每个 policy chunk 平均约有 **26 次 verifier target forward**，每次约打包 **7 个逻辑 candidate rows**，CFG 打开时实际 transformer batch 约为 **14 rows**。此外还要支付 draft/tree build、candidate pack、score、selector、route execute、fallback 和 terminal forward。

最新四个完整 `libero_10/task0/init0–3` 配对 episode 的端到端结果：

- Target-only：`370.366941 s`
- K=2+B=2+global：`369.103573 s`
- 差值：`-1.263368 s`，即 `-0.341113%`，四舍五入为 **快 0.34%**

这应判定为**基本持平**，而不是稳定加速。当前最主要的两个问题是：

1. **verifier budget 仍然很大**：target 计算没有被压缩到“一次 verifier 覆盖一个完整 chunk”，而是被拆成很多候选树的 target verifier。
2. **闭环轨迹变长**：K2 平均 `20.5` 个 inference chunks、`314.25` 个环境步；Target-only 为 `18.0` 个 chunks、`276.5` 个环境步。单次 policy 便宜并未转化为整条 episode 更短。

因此建议在继续大规模训练或正式多任务评测前，先解决并量化 verifier/轨迹长度问题。

## 2. draft 是否真的在线使用

是。已有 K2 live trace 审计记录：

- draft scheduler updates：`6,915`
- target fallback/correction updates：`1,635`
- action scheduler updates：`8,550`
- K2/B2 每棵 tree：`7` 个逻辑候选 rows、`1` 次 target verifier forward

因此不是“只加载 checkpoint 没有使用”。但 draft 的职责只是生成候选状态，不能替代 target 的质量判定。candidate verifier 使用 `update_cache=0`，不会把候选分支写入持久 KV；这只解决 cache 污染，不会消除候选 target transformer 的计算。

## 3. 单 chunk 的成本账本

来自已有稳态 phase profile；单位为秒/每个 policy chunk。子阶段是嵌套项，不能把它们再次相加到 `action_total`。

| phase | Target-only | K=2+B=2 | 解释 |
|---|---:|---:|---|
| video total | 1.8363 | 1.8284 | video branch 未做 speculative，基本不变 |
| video target forward | 1.8075 | 1.7979 | 没有被 draft 消除 |
| action target forward | 4.2700 | 0.2047 | 逻辑 target scheduler update 从 50 降到少量 direct target forward |
| action tree build | — | 0.2214 | draft/tree expansion、scheduler 等 |
| action draft model | — | 0.0697 | 嵌套在 tree build 内 |
| verifier target forward | — | 2.1782 | 新增主要成本；每 tree 一次 batched target verifier |
| action total | 4.4360 | 2.8211 | action 局部确实变快 |
| policy total wall | 6.2772 | 4.6559 | 单次 policy 调用约少 1.62s |

因此“单 chunk 变快”是真实的；但收益不是免费跳过 48 次 target forward，而是用多候选 target verifier 和 draft/tree bookkeeping 换取较少的串行 scheduler updates。

## 4. 为什么完整 episode 几乎不变

### 4.1 轨迹长度抵消了单次 policy 收益

正式四-init 配对均值：

| 指标 | Target-only | K=2+B=2+global | K2−Target |
|---|---:|---:|---:|
| E2E | 370.367s | 369.104s | -1.263s |
| reset | 135.496s | 128.260s | -7.237s |
| policy | 204.921s | 208.721s | +3.800s |
| cache | 9.916s | 10.840s | +0.924s |
| other | 20.034s | 21.283s | +1.249s |
| env steps | 276.5 | 314.25 | +37.75 |
| inference chunks | 18.0 | 20.5 | +2.5 |

按这组数据粗略计算：

- Target-only 每个 policy call 平均约 `204.921 / 18 = 11.384s`；
- K2 每个 policy call 平均约 `208.721 / 20.5 = 10.182s`；
- 单次约省 `1.20s`，但 K2 多了约 `2.5` 次 policy call，按 K2 单次价格就是约 `25.45s` 的额外 policy 工作；
- reset 少约 `7.24s`，cache/other 合计多约 `2.17s`，最后只剩 `1.26s` 的 E2E 差异。

这是闭环控制轨迹的混杂，不是说 K2 的局部优化没有效果，而是当前 draft 产生的 accepted/rollback 路径没有保证和 target-only 在相同的完成步数上结束。`init0` 中 K2 比基线慢约 `91.5s`（18→28 chunks），而 `init1`、`init3` 又分别快约 `40.2s`、`57.7s`，说明四个 init 的轨迹方差已经大到足以覆盖单次推理收益。

### 4.2 verifier 不是“每个 chunk 一次验证”

当前 K2 的实际结构更接近：

```text
一个 50-step action chunk
  = 多棵 K=2 draft tree
  = 每棵 tree 构建 7 个 candidate state
  = 每棵 tree 1 次真实 target verifier forward
  + reject 时 target fallback
  + chunk terminal target forward
```

已有 profile 的一个稳态 chunk 为 `26` 棵 tree，因此是 `26` 次 verifier forward，而不是一次 verifier 覆盖整个 50-step chunk。K=3 的成本审计也验证了同一机制：更深的 tree 会增加 tree build 和 verifier 时间，接受深度更高并不自动更快。

### 4.3 video branch 仍然由大模型完成

当前只 speculative action branch。每次 policy 调用仍要执行约 `1.8s` 的 video target forward；这部分不会因 action draft 而消失。因此即使 action branch 局部减少约 `1.6s`，完整 policy 调用也只能获得有限收益，更不可能直接按 action 的比例缩短整条 episode。

### 4.4 reset/cache 是大项，但不是本轮 K2 持平的首要原因

- reset 是每个 episode 的绝对大项（约 `128–135s`），但最新配对中 K2 反而少 `7.24s`；其内部 T5/text、KV 初始化、streaming VAE reset、同步/cleanup 的精确比例尚未插桩，不能武断归因。
- cache 约占 E2E 的 `2.68%`（Target-only）和 `2.94%`（K2），K2 只多 `0.924s`。它不是当前 E2E 持平的主因。
- 真实跨 chunk observation history 仍必须保留；可以丢弃普通 diffusion 中间态和只读 verifier candidate 分支，但不能把真实历史 KV 误删成“只留最后一次”。

## 5. 是否现在就应该先解决

**应该。** 当前继续训练更多 MLP/Transformer 或直接扩大多任务线上评测，信息增益都低于先把系统账本拆清楚。建议顺序如下：

### Gate 1：源码级轻量插桩（不做 smoke）

仅记录完整 policy chunk/episode 的计数和阶段时间，不改变 accept/rollback 语义：

- draft forward 数、draft tree 数；
- verifier forward 数、logical/CFG candidate rows；
- ordinary target forward、terminal target forward、fallback 数；
- accepted depth、rollback 比例、selector 时间；
- tree build、candidate pack/score/route execute；
- reset 内 text/T5、KV 初始化/清理、streaming VAE、同步/cleanup；
- cache 内 VAE encode、video/action KV forward、同步/cleanup。

### Gate 2：完整 horizon 的诊断 pair

仍然只做完整 episode，不做 smoke、one-chunk、dry-run 或截断；保持 GPU-VAE、EGL、双独立 worker/disjoint shards、端口隔离、`--resume`。诊断 profile-on 数据与 profile-off 主 latency 表分开保存。

### Gate 3：先改 verifier budget，再决定继续训练/大规模 eval

优先验证以下方向：

1. 只验证高不确定性 candidate，减少无效 candidate rows；
2. 复用 root/已计算 target 输出，避免重复 target forward；
3. 设计一次更高效的 batched verifier，明确按 target rows 和 transformer FLOPs 记账；
4. 让一个 verifier 覆盖更多有效 scheduler steps，而不是每棵小树都独立验证；
5. 保留真实 target accept/rollback 作为质量门槛，不能用 draft 直接替代 target。

在 verifier budget 没有明显下降、且轨迹长度没有恶化前，**暂停 Phase-A Transformer/MLP 的大规模 LIBERO 线上评测和继续训练**是更合理的。

## 6. 当前 GPU 运行态

本轮在 **2026-10-08 19:14:10（Asia/Shanghai）** 只读核验：

- `nvidia-smi --query-compute-apps`：无 compute app；
- 无 `wan_va_server`、eval client、rollout/train 进程；
- 29302、29303、29456、29461、29496、29501 均无监听；
- GPU 0 显存 `0 / 40960 MiB`；
- 瞬时 GPU utilization `23%`，但没有 compute app 和显存占用，不是本项目模型服务。

结论：**当前没有需要暂停/释放的本项目 GPU 服务，GPU 已经处于释放状态；本轮没有杀进程，也没有重启服务。** 16:45 的历史记录曾观察到两个旧 server，但 19:14 的新核验已确认它们已经消失；本节状态优先于旧快照。

## 7. 本轮决策与边界

- 不把当前结果描述为稳定端到端加速；
- 不把 draft 描述成替代 target；
- 不把 cache/reset 的未知内部比例写成已确认原因；
- 不启动 video branch；
- 不启动新训练、线上 eval 或 smoke；
- 下一步先做完整 horizon 的 instrumentation pair，再决定 verifier 源码优化；
- `.pipeline/state.yaml`、`.pipeline/cycle.yaml`、`.pipeline/rules.yaml` 未修改。

## 8. 证据索引

- 正式 E2E：`/rjs/cxz/论文探索/.pipeline/debug/Phase123-report-20261007.md`
- 单 chunk/K2/K3 profile：`/rjs/cxz/论文探索/.pipeline/reports/P0-depth3-cost-analysis-target-K2-K3.md`
- K2/K3 完整 episode：`/rjs/cxz/论文探索/.pipeline/reports/P0-K2B2-bonus-vs-K3B2-LIBERO-task0-init0.md`
- 当前 K2 trace 汇总：`/rjs/cxz/论文探索/.pipeline/PROGRESS.md` 中 K2 counters 条目
- server implementation：`/rjs/cxz/论文探索/投机采样/wam-p0-worktree/wan_va/wan_va_server.py`

## 11. 2026-10-08 19:36:56（Asia/Shanghai）：用户追问后的运行态复核

本次仍为只读检查，没有启动、停止或重启任何任务：

- `nvidia-smi`：GPU 0 为 `NVIDIA A100-SXM4-40GB`，显存 `0/40960 MiB`；
- compute-app 查询为空，`nvidia-smi pmon` 没有 GPU 进程；
- 未发现 `wan_va_server`、eval、rollout 或 train client；
- 29302、29303、29456、29461、29496、29501 均无监听；
- 未发现 `/dev/nvidia*` 的占用者。

因此当前没有可暂停的本项目 GPU 服务，也没有执行 kill。一次性 `utilization.gpu=25%` 的采样没有对应 compute app/显存占用，不能据此判断有遗留评测服务；后续正式任务启动前仍按 EGL、GPU-VAE、双 worker、隔离端口和完整 horizon 重新做 preflight。
