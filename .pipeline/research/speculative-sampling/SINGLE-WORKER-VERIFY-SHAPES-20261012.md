# 投机采样：普通 action / batch14 verifier 的输入与 KV 形状

日期：2026-10-12（Asia/Shanghai）。范围：投机采样，非 TILE 算子研究。

用户要求一边后台修私有包装并预检/GPU验证，一边由只读 subagent 解释形状，以便共同讨论更有效的共享方案。本文为源码/config证据；不能替代本轮 GPU 正确性或性能结果。

## 形状

以下以同等长历史有效长度 L=2144 举例，每层 K、V 各自采用 BSHD 布局。模型30层，hidden=3072，24 heads×128；原始action通道数30与层数30是不同概念，当前动作 token 数为4×4×1=16。

| 张量 | 普通 action | 7候选 verifier |
|---|---|---|
| CFG前原始action | [1,30,4,4,1] | [7,30,4,4,1] |
| CFG后模型输入 | [2,30,4,4,1] | [14,30,4,4,1] |
| 当前hidden | [2,16,3072] | [14,16,3072] |
| 当前Q、K、V，各自 | [2,16,24,128] | [14,16,24,128] |
| 持久有效历史K、V，各自 | [2,2144,24,128] | [2,2144,24,128] |
| 展开的候选历史K、V，各自 | 不做候选展开 | [14,2144,24,128] |
| attention完整K、V，各自 | [2,2160,24,128] | [14,2160,24,128] |

普通列表示同等历史状态下读入当前16 tokens的数学尺寸，实际原始缓存临时写入/rollback/eviction需按有效mask判断。不能把所有普通forward的历史长度固定成2144。

custom_sdpa将BSHD转成BHSD，verify的Q为[14,24,16,128]，完整K/V为[14,24,2160,128]。历史hidden不重复经过30层；各层只计算当前16 tokens的Q/K/V并读自己的历史cache。

历史rowmap为[0,0,0,0,0,0,0,1,1,1,1,1,1,1]。两份history对应conditional/unconditional CFG；不应假定全局只有batch1。候选当前KV有区别，不允许跨候选attention。

## pool、prompt与现有优化边界

- 有效历史长度不同于pool容量。按LIBERO配置推导，P=(30//2)×(128 video tokens/chunk +16 action tokens/chunk)=2160，pool每个K/V为[2,P,24,128]。本文未加载原始大fixture确认其运行时pool shape。
- 用T表示实际prompt长度：普通text_emb为[2,T,4096]，verifier为[14,T,4096]；投影后最后一维3072，每层text cross-attention K/V为[B,T,24,128]。正常reset显式指定T=512，实际恢复fixture的T尚未读取。本历史self-attention KV与text KV是两个不同对象。
- 当前C是将14行展开历史保存在受管lease中，跨29个变化query复用；HC再复用动态检查。每次full-KV cat仍保留。**目前不是直接读取两份prefix的共享attention kernel，prompt/text投影共享也没有实现。**
- BF16形状算术：两行history K+V每层50.25MiB；十四行每层351.75MiB，30层约10.305GiB。两行30层约1.472GiB。每层full-cat K+V为354.375MiB，其中历史约99.26%。这些是张量存储大小，不是测得HBM traffic或速度占比。

## 下一步可共同讨论的接口

```text
Q / K_current / V_current: [14,16,24,128]
K_prefix / V_prefix:      [2,2144,24,128]
rowmap:                  [14]
output:                  [14,16,24,128]
```

目标是第b行只读取prefix[rowmap[b]]及current[b]，不物化history14/fullKV14。

必须保持一个prefix+current共同softmax；不能两个独立归一化输出直接相加，不新增causal mask，不跨候选attention。应正确处理有效slot次序/非连续pool、历史commit/rollback/restore/观测替换、模型或rowmap变化的失效。实现后仍须验证数值等价、显存及完整verifier净收益，不能仅由内存算术宣称加速。

当前先完成既有C/HC full-verifier GPU对照。真正prefix共享kernel、prompt共享及公共attention改造尚未实施；用户的设计建议作为下一步讨论输入，不自动重构、训练或扩大GPU预算。

## 源码证据

- [模型配置](../../../models/lingbot-va-posttrain-libero-long-modelscope-20260922/transformer/config.json)
- [LIBERO动作/CFG配置](../../../投机采样/wam-p0-worktree/wan_va/configs/va_libero_cfg.py)
- [SDPA布局转换、历史KV展开、动作embedding](../../../投机采样/wam-p0-worktree/wan_va/modules/model.py)，另见633–677、1060–1116行。
- [候选CFG打包与rowmap](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)
- [C/HC复用与保留full-cat](../../../投机采样/experiments/single_worker_chc_gpu_v4/adapter.py)

证据：只读subagent汇总，经主agent轻量源码复核；不是独立GPU审计。subagent未编辑文件、运行模型、初始化CUDA或加载巨大fixture。
