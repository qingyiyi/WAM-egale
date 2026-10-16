# 投机采样：CFG2 保留、七候选历史 KV 共享的真实 GPU 数值结果

日期：2026-10-12，Asia/Shanghai。GPU 于 16:43:57 结束；17:01:44 完成已有输出的 CPU 复算，之后只读独立审查完成。此报告属于 **投机采样**，不属于旧 TILE/M04 算子里程碑。

## 1. 当前结论

**S 的真实共享实现已运行，但没有通过正式逐位一致门；不能宣称优化成功或报告正式性能收益。** 新 job 因数值关失败停止，不是被其他用户打断，也不是 loader/CPU 启动失败。GPU 已清理，无后台重跑/等待队列，无训练。

本轮保持 CFG 两份历史。S 直接读取原始 `K/V_pool[2,2160,24,128]`，以 `valid_slots[2144]` 选择有效历史、`rowmap[14]` 选择 CFG 分支；Q 和当前 K/V 仍为 `[14,16,24,128]`。不构造 batch14 历史，不构造完整 `K/V[14,2160,24,128]`，当前候选只读自己的 16 个 token，历史和当前 token 做一次联合 softmax。prompt 不共享，公共模型源码、权重、batch14 不变。S 只替换默认关闭、任务专用实例包装内的候选历史 attention。

这仍是 **历史存储复用**，不是七个 CTA 已实现片上 tile 复用；不能承诺七倍速度或 shared memory 降低七倍。原来的 C/HC 扩展历史缓存仍是独立对照，不称作 S。

## 2. 各关结果及边界

| 关卡 | 实际结果 | 边界 |
|---|---|---|
| 新 v3 job CPU 预检 | 25 项 prefix + 17 项入口/监督器，共 42/42；同实际生产 loader/worker probe 58 次意图、58 次返回，日志 CLOSED | CPU 数学/接口不代表真实 BF16 GPU 一致性 |
| GPU 独立 kernel 诊断 | 6 组长度/stride/幅度样例、FP64 计算后转 BF16 oracle 和候选隔离，15 次直接 attention 意图；诊断关 PASS | 不是逐位 PASS，未执行 warmup/计时；不等于真实模型隔离或策略通过 |
| 真实恢复/捕获 | PASS，52 次 target 意图；30 层、历史 2144、29 个不同 verifier 输入；在 noise=0 终端动作 forward 之前停止 | 0 EGL、0 完整 policy chunk、0 episode、0 环境步 |
| R 资源探针 | 1 次 target，PASS | 非成本主测 |
| 冻结 HC | 29/29 输出逐位一致；全层 history/metadata/RNG 检查通过 | 此段不是平衡重复的干净 OFF 主测，不拿 wall 时间算加速比 |
| S 全层固定输入重放 | 29/29 输出有限，但 **0/29 逐位一致**；全部输出及第一条 query 的逐层归因已保存 | 正式 FAIL；S 末尾 history/RNG 检查尚未到达 |
| S GPU 十二类失效、真实模型隔离、warmup、六组 OFF | 均未到达 | 不能拿 CPU/synthetic 隔离替代；没有正式速度和显存成本对照 |
| 清理 | PASS，本任务 PID/端口均清空；未信号其他任务 | 本次没有自动补跑 |

GPU job：[p0-shared-prefix-v3-20261012-163453](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/plan.json)。[原始失败](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/FAILED.json)、[清理](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/OWNERSHIP-CLEANUP.json)。本轮执行源是 `shared_prefix_gpu_v3`，不是初稿 v1。

## 3. 为什么结果不一致：事实与推断分开

### 3.1 首次差异在 attention 内，之后有层间传播

对第一条真实 query，第 0 层 S/R 的 **Q 和 currentK 逐位相同**，attention 输出却已不同：相对 RMS **0.07470%**，最大绝对差 **0.015625**。这排除了“所有差异都只来自后续层 Q/K 先变了”的解释。

进一步把每一层保存的 **同一份原始 R Q/currentK/V** 单独送进新 kernel，30 层均不逐位一致：attention 局部相对 RMS **0.06864%–0.10589%**，中位数 **0.08931%**；最大绝对差范围 **0.0078125–0.03125**。这把新 attention 的局部差异与后续层不同输入的传播分开了。

第 1 层开始实际 S 的 Q/currentK 已不同；第 29 层实际 S/R attention 的相对 RMS 为 **2.55534%**。这支持一次 30 层 forward 内的误差传播和部分放大，**不是每层都单调变大，也不能定量分摊全部最终误差**。

[真实逐层原始结果](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/numerical/ATTENTION-ATTRIBUTION.json)。注意：该逐层归因仅覆盖第一条 query；JSON 未保存 S/R currentV 的比较，也未独立比较归因时的完整 prefix pool/slots。真实 S 的末尾状态检查因严格异常未执行，仍不能完全排除其他实现/状态问题。

### 3.2 最有力的解释是数值执行路径变了，但不能宣称“没有 bug”

新 kernel 使用 64-token 分块、在线 `exp2` softmax、FP32 累加，并在 `P×V` 前执行 `prob.to(vval.dtype)`，这里为 BF16。改变分块归约、缩放/指数计算、概率舍入或融合顺序，即使联合 attention 数学式相同，也不保证与原 SDPA 的字节结果相同。

synthetic 长历史 `L=2144` 中，S 与原 SDPA 的相对 RMS 为 **0.29692%**；它们与“FP64 计算、最后转 BF16”的独立联合-softmax oracle 的相对 RMS 分别为 **0.243791% / 0.243696%**，非常接近。再加上无历史 `L=0` 也有极少量差异，支持 **kernel 数值路径改变，而非历史共享本身必然算错** 的解释。

但是：上述 oracle 是 synthetic 数据，尚不是全部真实 29 query 的高精度基准；没有测出原 SDPA 的实际选中 CUDA backend/分块细节。真实输入相关的映射、尾块、数值实现、prefix 状态问题仍须逐项排查。**“主要是舍入/归约变化”置信度中高；“唯一原因就是舍入且绝无 bug”证据不足。**

[kernel 源码](../../../投机采样/experiments/shared_prefix_gpu_v3/prefix_kernel.py)、[原始独立 GPU 诊断](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/results/kernel.json)、[只读独立复核](../../../review/OMISSIONS.md)。子代理未写文件、运行 GPU/模型或训练，由主代理派发、复核并回收，无需用户手动控制。

### 3.3 最终 Transformer 输出的差异不能直接忽略

对保存的 R/S BF16 输出 `[14,16,30]` 在 CPU 独立复算，与原始 GPU 导出的误差统计吻合：

| 指标 | 最小 | 中位数 | 最大 |
|---|---:|---:|---:|
| 相对 RMS | 0.39543% | 1.19123% | **4.51987%** |
| RMS 绝对差 | 0.002204 | 0.007912 | 0.023007 |
| 最大元素绝对差 | 0.015625 | 0.046875 | **0.22265625** |
| 不同元素个数（总 6720） | 5228 | 5921 | 6509 |

**17/29** 超过源码中既有的 1% 诊断 corruption ceiling。实际停止原因仍是逐位门，本次从未启用诊断容差；即使只授权诊断性能，当前版本也不会因此自然通过既有 1% 上限。该上限不是正式策略验收容差，不自动放宽。

另外按生产 server 的 CFG 输出顺序及 LIBERO 使用通道做已有数据的 CPU 切片：CFG0（conditional 前 7 行）、通道 0–6 的方向输出相对 RMS **0.31472%–3.56286%**，中位数 **0.88727%**，13/29 超过 1%。因此差异并不只集中在弃用的 CFG1 或未使用的通道。此切片仍是 **去噪方向**，不是最后执行的机器人动作；没有 scheduler 推进、selector 重打分、终端 KV 更新或真实环境实验，不能据此宣称任务准确率相同。

**特别注意：29 组是 original 路径捕获的不同固定输入，逐个独立 replay；上一条 S 输出没有喂给下一条输入。** 所以这不是沿 S 去噪轨迹累计 29 次误差，也不是跨 chunk/环境步误差累积。这里只看到了单次 forward 内的层间传播及不同输入的误差差异。

[CPU 复算及 CFG/通道切片](../../../review/OMISSIONS.md)、[复算程序](../../../review/OMISSIONS.md)、[生产 CFG 合并和通道配置](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)、[LIBERO 配置](../../../投机采样/wam-p0-worktree/wan_va/configs/va_libero_cfg.py)。

## 4. 预算与证据保全

- 本次 GPU：**153.259 秒（2 分 33 秒）**，111 次 target 预算意图、111 次已确认 Python host 返回，0 失败调用、0 pending；写前日志 CLOSED。计数不是通用 CUDA 完成证明；已保存输出和同步支持实际数值数据分析。
- 另有 **45 次直接 attention 意图**：独立诊断 15 + 同 R QKV 逐层归因 30。它们不是 target forward，持久化在独立日志，耗时计入 GPU 限时。主账本 `extra_micro_calls=0` 不能解释成“没有直接 attention 诊断”。
- 前次已消耗保留为 183 意图 / 507.291 秒，不退还负例或失败成本。累计 **294/900 意图、660.551/2700 秒（11 分 0.55 秒）**。
- 算术余额：606 次 target / 2039.449 秒（33 分 59 秒）。**余额不代表授权重跑**；新 GPU job/预算与任何验收标准变化均另确认。
- 0 新完整 episode/EGL/chunk/env steps，0 训练/RL/大小模型联训。单 worker，无 peer，无 compute-mode 改动。
- CPU 分析只读取旧数据，`CUDA_initialized=false`，未加载/运行模型，未启动新 GPU。校验 plan 冻结的 1367 份输入 + 14 份执行源码，共 1381 个 SHA；失败 runtime 全部文件另行封存哈希，保护 state/rules 未修改，旧 FAIL 不改写。

[恢复计数](../../../投机采样/results/runtime/p0-shared-prefix-v3-20261012-163453/RECOVERED-COUNTER.json)、[45 次直接诊断日志](../../../review/OMISSIONS.md)、[只读封存](../../../review/OMISSIONS.md)。

## 5. CPU 包装版本的记录勘误（旧记录不覆盖）

1. 首轮 prefix CPU 实际 **22/22**，不是早期 HW 摘要的“24项”；第二轮 24、第三轮 25。最终新 v3 job 是 **25+17=42**，另有 58 次同 worker probe。
2. v1 CPU 入口测试的 2 个失败来自私有 argparse 测试 harness 漏掉 argv 第一项剥离；0 GPU，并非共享 kernel/模型数值失败。修复采用新 v2，不覆盖失败原件。
3. v2 40/40 CPU 通过、0 GPU；只读审查发现未授权的 micro 性能计时分支和最终 strict 汇总缺口，主动不启动 GPU。v3 关闭未授权 micro 计时、将所有 S/OFF/结束检查 strict 状态汇总，42/42 后才运行此次 GPU。新 v3 与 v2 的 kernel 数学实现相同。
4. 初稿阶段预算误算 685 已留档纠正，最终阶段合计 **585**，本轮硬限 715；本次实际只消耗 111。

[CPU 最终记录](../../../review/OMISSIONS.md)、[v2 暂缓记录](../../../review/OMISSIONS.md)。

## 6. 下一步（当前未执行）

**不重构投机研究方案，也不开始 mega Transformer/训练。先处理 prefix attention 的数值一致性，再做成本主测。**

建议下一步先只 CPU 制定最小数值归因/修复方案：

1. 检查原 SDPA 可选 backend 与本地实现，列出 scale 位置、指数、tile 归约、P 转 BF16、融合顺序的对照；实际 GPU 选中 backend 仍需后续有界验证，不凭名称猜测。
2. 后续最小 GPU 诊断应先保存并比较实际 prefix/slot/currentV、同 QKV 局部误差和状态不扰动，再扩大到真实变化 query；不能只看 synthetic 相似就批准整模型。
3. 保持正式逐位门。如希望改为数值非劣路线，需用户先明确同意并制定独立标准：实际使用方向/动作、selector/accept/rollback trace、之后真实任务成功率和环境步数；不能把既有 1% 诊断上限或“看起来很小”直接当正式标准。
4. 数值及状态关通过后，才重新确认预算并跑六组平衡 R/S 完整 30 层、29 变化输入 OFF 成本与峰值显存，冷租约/索引/搬运/guard/日志/释放均计费。最终仍需闭环成功率、环境步数、E2E，不用局部 kernel 速度代替。

当前只完成已有结果分析与 HW 记录；没有实施上述新数值修复或新 GPU 实验。
