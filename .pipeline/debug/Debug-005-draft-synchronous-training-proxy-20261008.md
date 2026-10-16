# Debug-005：同步训练 Draft Model 的离线代理实验

日期：2026-10-08（Asia/Shanghai）  
范围：**仅 action branch**；video branch 暂不处理。  
状态：**低成本离线 proxy 实验完成；未启动 LIBERO、verifier、线上 rollout 或新的 GPU 服务。**

## 1. 用户问题与假设

用户提出：如果在 target 大模型训练过程中同步训练 draft model，使 draft 持续跟随 target 的当前策略，而不是只拟合最终 target checkpoint，是否可以获得更好的效果？

动机是 target 在训练过程中不断变化。若 draft 只用最终 target label 训练，可能存在策略分布滞后；同步训练可能让 draft 学到更接近 target 当前轨迹的动态，尤其可能改善 draft 自回归 rollout 的长程漂移。

本轮只做一个低成本、可快速判别方向的代理实验，先回答：

1. 同步训练是否值得继续研究；
2. 它的收益是否主要体现在 recursive/closed-loop，而不是单步 teacher-forced 误差；
3. 是否有理由进入真实 target intermediate checkpoint 的离线实验。

## 2. 实验边界：这不是真实 target co-training

本实验不是 LingBot-VA/WAM 大 target 的真实 checkpoint co-training，也没有在真实 target 训练进程中插入 draft optimizer。

使用的是离线 surrogate-target proxy：

```text
固定最终 target label
        ↓
surrogate target 持续拟合最终 label
        ↓ detach
同步 draft 跟随 surrogate 当前输出
```

每个 batch 的行为：

1. surrogate target 用最终 target label 更新；
2. 用更新前的 surrogate prediction 作为 draft 的 teacher signal；
3. teacher prediction 使用 `detach`；
4. draft 梯度不会回传到 surrogate；
5. direct 与 synchronous draft 使用相同的 draft 更新步数预算。

因此该实验只能提供“同步跟随可能改善闭环稳定性”的 proxy evidence，不能直接推出线上 acceptance、task success、environment steps 或端到端加速。

## 3. 数据与配置

- 分支：action-only；video branch 延后。
- 数据文件：`/rjs/cxz/论文探索/投机采样/results/draft_full_phase4-20261007-f32`
- 总行数：80,017。
- 特征维度：480。
- 固定 split：train 67,277；validation 10,290；test 2,450。
- seed：20261008。
- epochs：3。
- batch size：256。
- learning rate：2e-4。
- weight decay：1e-4。
- 总耗时：约 490.39 s。

### 对照组

**Direct baseline**

- draft：`ContextMLP(d_model=128)`；
- 直接拟合最终 target residual/label。

**Synchronous proxy**

- draft：`ContextMLP(d_model=128)`；
- surrogate target：`ContextTransformer(d_model=128, layers=2, heads=4, ff=512)`；
- surrogate 每 batch 持续拟合最终 target label；
- draft 每 batch 跟随 surrogate 当前 prediction。

### 结果文件与复现脚本

- 结果：`/rjs/cxz/论文探索/投机采样/results/draft_sync_proxy-20261008/proxy_result.json`
- 运行脚本：`/tmp/sync_draft_proxy_20261008.py`
- 日志：`/tmp/sync_draft_proxy_20261008.log`

## 4. Teacher-forced test 结果

| 指标 | Direct | Synchronous proxy | 同步相对 Direct |
|---|---:|---:|---:|
| direction MSE | 0.00278309 | 0.00282747 | 约差 1.59% |
| direction relative L2 | 0.0803665 | 0.0811240 | 略差 |
| direction cosine | 0.993102 | 0.993073 | 基本持平 |
| one-step state relative L2 | 0.0137358 | 0.0137325 | 基本持平/略好 |

validation 的最佳 direction MSE 为：

- Direct：0.00297618；
- Synchronous proxy：0.00302668；
- 同步代理约差 1.70%。

**结论：** 同步训练没有改善平均单步拟合误差，反而有轻微退化；但退化幅度小，cosine 与 one-step state 指标基本持平。

## 5. Recursive test 结果

正值表示同步 proxy 相对 Direct 的 endpoint state relative-L2 改善。

| Horizon | Direct endpoint state rel-L2 | Sync endpoint state rel-L2 | 相对改善 |
|---:|---:|---:|---:|
| h=1 | 0.0125068 | 0.0124861 | +0.17% |
| h=2 | 0.0243690 | 0.0242818 | +0.36% |
| h=4 | 0.0468360 | 0.0463402 | +1.06% |
| h=8 | 0.0904772 | 0.0870888 | +3.75% |
| h=16 | 0.2030717 | 0.1750335 | **+13.81%** |

recursive direction relative-L2 的变化也支持同一趋势：

- h=1：同步约差 0.82%；
- h=2：同步约差 0.98%；
- h=4：同步改善约 0.74%；
- h=8：同步改善约 10.27%；
- h=16：同步改善约 33.34%。

surrogate target 自身在 h=16 的 endpoint state relative-L2 为 0.1712600；同步 draft 为 0.1750335，已经接近当前 surrogate 的闭环行为。

## 6. 现象解释

### 6.1 为什么单步略差

同步 draft 的 teacher 不是固定的最终 target，而是 surrogate 在训练过程中的当前输出。这个 teacher 仍在移动，并且早期训练阶段包含优化噪声，因此 draft 收到的是一个非平稳、略带噪声的监督信号。与直接拟合最终 label 相比，它不一定能把最终时刻的平均单步误差压到最低。

另外，本实验的同步代理有额外的 surrogate 训练和 teacher forward；虽然 direct 与 sync 的 draft update budget 相同，但总训练计算量并不等价。因此这不是“相同计算量下同步一定更优”的证明。

### 6.2 为什么长 horizon 反而改善

recursive rollout 中，draft 的输出会进入下一步输入，误差会被反馈放大。同步 draft 学到的是 surrogate 当前策略的局部动态，而不只是最终标签的静态回归，可能因此减少策略分布不一致带来的长程漂移。

从结果看，h=1/h=2 基本持平，h=8/h=16 才出现明显差异；这更符合“改善闭环稳定性，而不是改善单步精度”的解释。同步 draft 在 h=16 已接近 surrogate target，说明它更像是在跟随一个持续变化的 teacher，而不是只追求最终标签上的最小平均误差。

以上是基于 proxy 结果的机制解释，不应当当作真实 target 训练中的因果证明。

## 7. 当前决策

本轮结果支持**继续研究同步训练**，但不支持立即把它接入真实线上训练或 LIBERO 正式评测：

1. 保留同步训练作为候选方向；
2. 不替换当前正式协议：

   ```text
   draft K=2 → 真实 target verifier → accept / rollback
   ```

3. draft 仍然只能提出候选，不能替代 target/verifier；
4. 暂不启动大规模真实 target co-training、长时间 LIBERO eval 或新的 GPU 服务；
5. 先使用真实 target intermediate checkpoints 做更接近真实问题的离线对照。

## 8. 解释边界

本实验**没有测量或证明**：

- 真实 online verifier acceptance 提升；
- accepted prefix 增加；
- task success 提升；
- environment steps 不增加或减少；
- E2E wall-clock 加速；
- 真实 LingBot-VA target checkpoint refresh 下的收益；
- video branch 的收益。

尤其要注意：目前真实 E2E 的主要问题仍是 verifier/tree/fallback/trajectory budget，而不是单纯的 draft 单次 forward。因此即使同步训练改善了 h=16 的离线 drift，也必须通过真实 target verifier 和完整环境轨迹 gate 才能证明有系统价值。

本实验还只有单 seed，没有方差估计；h=8/h=16 的改善需要后续多 seed 和真实 checkpoint 实验确认。

## 9. 后续计划

### Phase 1：归档本轮 proxy

已完成本报告、Knowledge Ledger、PROGRESS 记录及索引重建。

### Phase 2：准备真实 target checkpoint provenance

保存每个 checkpoint 的：

- model state；
- optimizer state；
- scheduler state；
- global step；
- random state；
- 数据迭代位置；
- 配置与 provenance。

### Phase 3：真实 checkpoint 的低成本离线对照

在不启动 LIBERO 长时间 rollout 的前提下比较：

1. **Direct final-target**：draft 只拟合最终 target；
2. **Stale-target**：draft 拟合较早 target checkpoint；
3. **Synchronized intermediate-target**：draft 随多个 target intermediate checkpoint 更新。

优先使用已有完整 action 数据和真实 target checkpoint 输出，避免再次构造小 surrogate 代理。

### Phase 4：离线晋级指标

除平均 L2 外，重点统计：

- teacher-forced direction error；
- recursive h1/h2/h4/h8/h16；
- post-scheduler active-channel absolute successor-L2；
- threshold 附近样本与 high-error tail；
- early denoising 阶段；
- acceptance proxy；
- accepted prefix proxy；
- draft inference cost。

核心问题是：

```text
同步训练是否减少超过 verifier threshold 的高误差样本？
```

### Phase 5：只有离线 gate 通过后才进行线上验证

仍保持完整正式协议：

- action-only；
- K=2 draft → real target verifier → accept/rollback；
- EGL；
- GPU VAE；
- 双 worker、disjoint task shards、端口隔离；
- `--resume`；
- 完整 horizon；
- 无 smoke、无 one-chunk、无 dry-run、无人工截断；
- video branch 延后。

线上晋级条件：

1. task success 不低于 target-only；
2. aggregate/mean environment steps 不增加；
3. 不出现严重 long-tail；
4. verifier/tree budget 不抵消收益；
5. 最后才判断 E2E wall-clock。

## 10. 复核结论

> 可以先实验，而且低成本离线同步 proxy 已经完成。它没有改善单步平均误差，但把 h=16 recursive endpoint state relative-L2 从 0.2031 降到 0.1750，改善约 13.8%。这说明同步训练可能改善 draft 的长程闭环稳定性；但还不能证明真实 target verifier 下会提高 acceptance、减少 environment steps 或加速 E2E。下一步应优先用真实 target intermediate checkpoints 做 stale-target 与 synchronized-target 对照，而不是立即启动大规模线上 co-training。
