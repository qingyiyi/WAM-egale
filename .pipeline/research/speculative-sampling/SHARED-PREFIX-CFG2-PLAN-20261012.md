# 投机采样：CFG 保留两路、七候选共享历史 KV

## 当前进展与旧任务结果

- 用户本轮明确要求 **CFG 暂不合并，只共享同一 CFG 的七个候选**，实施任务私有、默认关闭包装并测试 GPU 性能；不训练、不改公共模型、不做 mega Transformer。
- 旧 job `p0-single-worker-chc-v4-20261012-152028` 因外部 GPU PID 1739088 进入而停止，并非数值或接口失败。CPU、真实接口准入和捕获已通过；C/HC 各 29 次真实输入、30 层输出逐位一致，前十类失效测试通过。第十一类中断，未完成总验证关、没有干净 OFF 性能结果。两次单序列非主测时间 **不得用作加速比**。
- 本次旧 job 已消费 183 次 target 预算意图，179 次 host 返回，4 次被测拒绝/异常；写前日志证明没有未决调用。GPU 加载至清理 471.296531 秒。前一个中断 job 消费 0 次、35.994813 秒。累计消费 **183 次、507.291344 秒**。旧任务和失败原始证据不覆盖、不复用。
- 现有 fixture CPU 检查：历史池确实 `[2,2160,24,128]`，视频 replay 前有效长度 2016。仅第 0 层两路 CFG 的 K/V 相同，第 1–29 层不同；正/负 prompt `[1,512,4096]` 也不同。本次保留 CFG2 有实际依据。verifier 捕获后的有效历史长度为 2144，不能混淆两个阶段。

## 实现 S 的边界

```text
Q/currentK/currentV    [14,16,24,128]
history poolK/poolV    [ 2,2160,24,128]  原池零拷贝引用
valid_slots           [2144]           按原 mask.nonzero 顺序
rowmap                [14]             0×7，1×7
output                [14,16,24,128]
```

- S 只替换候选 **self-attention 历史读**，保留投影、归一化、RoPE、FFN、cross-attention、prompt、原 batch14 与所有模型参数。
- 不生成 history14、不生成 fullKV14，也不拷贝历史到新 prefix2；直接从两路原池读取有效位置。原历史已经常驻显存，不把它算成“消除了”的 1.472 GiB。增量长期保留只有有效槽位索引，30 层约 0.491 MiB。
- 对历史和本候选的当前 16 个 token 做 **一个联合 softmax**；不是独立计算两个 softmax 后相加；不读其他候选当前 KV。候选当前 KV 本次 attention 必需，但验证结束不提交缓存。
- 沿用 v4 的真实 loader/source/实例身份、host epoch、张量版本、rowmap 所有权、模型世代、更新/回滚/恢复/驱逐失效检查；历史变化使旧租约失效，新租约重建索引。
- 首版 Triton 每候选每 head 一个 CTA。**共享历史存储不等于七 CTA 片上自动共享同一 K tile**；后者属于后续 kernel 设计，不把它混成 TILE 项目或 mega Transformer 的进展。

## 顺序、数值与预算

1. CPU：独立 float64 联合 attention oracle、空/短/长/非连续历史、候选及 CFG 隔离、默认关闭、真实生产 loader 全 30 层、失效及零拷贝检查。CPU 只能证明数学和接口，不能证明 GPU kernel。
2. 新 job：绑定本版 source SHA 和 CPU 关；空闲资源确认后单 worker 一次启动；0 episode/EGL/完整 chunk/环境步/训练。
3. GPU：先独立 kernel 数值/候选隔离，再恢复真实 fixture、捕获 29 变化 query（52 次非终止 target），验证全 30 层完整输出与十二类失效。
4. **不降低旧 C/HC 逐位门**。S 在线 softmax 使用不同浮点规约顺序，数学等价不意味着逐位一致；严格检查并保存原始误差。未获用户允许诊断分支时，S 逐位门失败就停止，不把误差小当成通过，不投用真实策略。
5. 通过后才能做完整 verifier 干净 OFF：R/S 六个预定交错配对块（冻结 HC 仅做数值/显存参考，不拿未配对时间作性能），全部样本保留；初始化/索引构建、staging、host guard、持久化计数、释放均收费；MAD/中位数 ≤7%。≥5% 且所有配对同向仅为速度 pilot，不证明成功率、环境步数或完整 policy 加速。

新任务最多 **715 次 target 前向意图**，累计 898/900；另有有界 kernel micro（不计作 target 前向，单独记录、不混入完整 verifier 指标）。实际分配 585 次，其余 130 次不作补跑许可。GPU 总限时仍累计 45 分钟：理论剩余约 **36 分 33 秒**，新 job 安全限取整收紧至 **36 分钟**，包括加载、编译、验证、性能和清理；15 秒预留清理。资源忙时只读等候，禁止干预其他用户，不声称软件监控可以独占硬件。失败停止，不补局、不自动重跑。

## 下一步

完成 S CPU 实现与新 job 入口回归，记录 HW；之后按上述预算测试 GPU 数值/失效，再测完整 verifier 性能。若 S 未逐位一致，先汇报误差与需审批的诊断分支，不继续声称“做成功”。最终再单独规划完整单 worker policy/闭环，环境步数与小模型训练问题仍留待后续。

源码目录：`/rjs/cxz/论文探索/投机采样/experiments/shared_prefix_gpu_v1`。
旧 job：`/rjs/cxz/论文探索/投机采样/results/runtime/p0-single-worker-chc-v4-20261012-152028`。
CFG CPU 证据：`/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/evidence/shared-prefix-cfg-cpu-20261012/CPU-RESULT.json`。

## CPU记录勘误与预算收紧

首轮实际22项（CPU-tests-001.json），早期Knowledge摘要误写24项，保留原记录并在后续HW显式更正；加入冻结HC与旧C模式拒绝两项后的新轮为24项，不能把新轮数量写回首轮。新GPU主要性能改为R/S六个交错配对块，HC仅数值/显存参考；分配585次、硬限715，累计分配768/900。

CPU入口预检在GPU前发现原草案对stage_caps的加总误写685（实际585）；按程序实际求和修正assert、摘要和测试。没有创建GPU job/启动GPU，不消耗target预算；保留create首次失败终端证据，不将CPU拒绝当GPU失败。硬限715及累计900/45分钟不放宽。
