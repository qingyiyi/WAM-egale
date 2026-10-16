# Debug 012 投机采样第 1 步合同与 leaf 复用核查

时间：2026-10-10（Asia/Shanghai）
范围：action-only 投机采样；静态审查、既有 CPU 合同测试与 CPU 合成例子。
状态：本步静态与 CPU 核查完成；真实 WAM 数值一致性、批量成本和轨迹变长根因仍未验证。
执行边界：没有修改模型源码、启动训练、GPU、服务或 LIBERO rollout。

## 1. 症状与本步目标

历史单 chunk 加速未转化为稳定 E2E 加速；最新六-init 描述性池中 K2 环境步数 +9.37%、E2E +0.89%。本步先核对 scheduler/accept/fallback/cache/环境执行合同，明确 leaf 输出是否被消费，不把训练误差先验认定为根因。

用户同意逐步推进，并要求每一步输出本步状态、结果、未解决项和下一步。后续记录仅属于投机采样路线，不与 tile/macro-tile 算子研究混用。

## 2. 假设与验证

| 排序 | 假设 | 核查结果及边界 |
|---|---|---|
| 1 | 普通 K2 计算 leaf，却未复用其 target 输出 | 确认。七行 root/两 internal/四 leaf 全部执行；评分只用前三行。普通模式提交最多两步，不消费选中 leaf 输出。bonus 模式才消费额外真实 target update。 |
| 2 | 近似验收被误当作 target-only 轨迹等价 | 确认两者不等价。完整接受提交 draft leaf；额外 target step 是在该 leaf 上算，不会逆向纠正此前接受的 draft 更新。CPU 合成例子对此构成反例，不是 LIBERO 根因实证。 |
| 3 | 候选分支污染持久 KV，或 scheduler 构树时偷偷推进内部索引 | 静态路径及 CPU scheduler 合成检查不支持。候选 target update_cache=0，专用候选 attention 只读选取历史；scheduler 通过 timestep 查 sigma，不维护逐调用递增索引。未运行真实模型缓存比较。 |
| 4 | fallback/bonus 复用了错误状态、时间或无效 leaf | 既有 helper 合同通过；prefix 0/1 置空 prefetch，prefix 2 才返回选中 leaf 输出。bonus 只在两步全接受且剩余更新数至少三步时启用。全 server/WAM 集成数值一致性未验证。 |
| 5 | 两算法环境执行 stride、成功检查或配对配置不同 | 当前 init4–5 的共享 client/source hash、checkpoint、seed、init、budget 与实际 event 不支持该假设。两臂 action_shape 均 [7,4,4]；首 chunk 执行 12 步，中间 16 步，终点按 done 每步退出。不能据此排除所有未观察到的控制实现问题。 |

## 3. leaf 复用的精确语义

- 验证行顺序为 root、两个 first state、四个 leaf；选中 leaf index = 3 + 2*parent + child。
- prefix 0：root target fallback，提交一个 scheduler update；leaf 不可复用。
- prefix 1：提交一个 draft update，再在其 first state 用 target 修正第二步；修正后状态不是原 draft leaf，prefetch 无效。
- prefix 2 普通 K2：提交两个 draft updates，保留 draft leaf 为新状态；leaf target 输出没有被用于额外提交。
- prefix 2 K2+bonus：在同一个选中 leaf 和对应第三步 timestep 上消费已有 target 方向，额外提交一个 target scheduler update，总计三步。下一轮状态及 previous direction 随之更新。
- bonus 覆盖的是去噪 scheduler step，不是直接多执行一个机器人 env step。
- 当前 launcher 使用 verifier_k2_b2，而非 verifier_k2_b2_bonus。当前源码与最新 K2 launch_config 中五个源码 hash 均相符；target-only 的 server/config/client/live-audit/launcher 五个 hash 也相符。

因此，删除四个 leaf 并非唯一候选；优先保留“七行一次验证并消费 leaf”与“三行验证后再算 target”的比较问题。没有本轮真实 target timing，不能判定哪种更快，更不能声称环境步数会因此减少。

## 4. scheduler 与缓存合同

K2 tree 和 fallback 都调用实际 FlowMatchScheduler.step 加相同有效通道/前缀投影。该 scheduler 按传入 timestep 选择 sigma；重复候选调用不会改变 sigmas/timesteps 或隐藏步索引。CPU 同输入重复调用结果一致。

候选 forward 使用 update_cache=0 和 candidate_cache_rows；模型专用候选 attention 读取匹配的历史行，不把候选写入持久 KV。完成全部 action scheduler updates 后仍有一次 terminal target update_cache=1；真实执行后的 observation/action cache 更新仍由 client compute_kv_cache 完成。

静态条件正确不等于已经用真实 WAM 权重验证所有 batch、dtype、CFG 和 cache 数值路径。这个限制保持未决，不把 32 个 helper 测试写成完整 WAM 集成验证。

## 5. 环境执行与 campaign 合同

当前 target-only/K2 的 checkpoint 标识和路径一致。四个 worker metadata 均使用 seed 20260922、horizon 800、无 max_policy_chunks 截断、5 个 stabilization no-op；双方 worker0/init4、worker1/init5 对齐。

共享 client 对每个执行动作调用一次 env.step 并累计一次 env_steps，每一步检查 done；首 chunk 跳过固定条件帧，之后使用全部四帧。与去噪接受深度/bonus 没有直接的环境步数计数绑定。

37 个 target 和 38 个 K2 原生 chunk event 的 action_shape 均 [7,4,4]。非终点序列为首 chunk 12 步、后续 16 步，终点允许提前完成。这支持“多走步不是简单换了执行 stride”的结论，不证明任务轨迹变长的因果机制。

这些 event 只含 shape、step/timestep、reward 汇总及 done，不含动作向量、夹爪切换、物体位姿或任务阶段。它们不足以单独定位哪里先发生返工；第 2 步需先盘点其他已有产物。

## 6. 本轮 CPU 验证

使用已有 libero-va Python，torch 2.9.0+cpu；CUDA_VISIBLE_DEVICES 为空，禁用 bytecode 与 pytest cache，不安装依赖。

命令：

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 /rjs/cxz/miniconda3/envs/libero-va/bin/python -m pytest -q -p no:cacheprovider 投机采样/tests/test_p0_batched_verify.py 投机采样/tests/test_p0_phase_b_k2_beam2_verifier_replay.py 投机采样/tests/test_p0_chunk_event_contract.py 投机采样/wam-p0-worktree/tests/test_p0_k3_b2.py

结果：32 passed in 2.08s。

额外无文件写入的 CPU 合成检查：
- 实际 scheduler 同输入重复调用相等，sigmas/timesteps 不变。
- 选中 leaf 的 target 方向与索引对应；bonus 等于从该 leaf 直接执行同 timestep 的 target scheduler step。
- prefix 1 不消费 leaf。
- target 恒零、draft 方向轻微非零且通过阈值时，接受后的 leaf 约为 -1.6078473e-05，bonus 后仍非零；target-only 对照为零。这只证明验收不保证轨迹等价，不证明此类误差造成 LIBERO 的额外环境步数。

## 7. 结论和下一步

本步发现的是已确认的计算消费差异与近似协议边界，没有定位到环境步数变长的真实根因，也没有修改/修复源码。

下一步：第 2 步先盘点已有 action、视频、observation、simulator state 和任务谓词产物，判断能否定位最先出现的停滞、返工或关键阶段延迟。若现有数据不足，先给出最小诊断字段及完整 episode 配对采集方案，不直接进入训练。

后续顺序：
1. 本步静态/CPU 核查完成；真实 leaf 净收益仍待独立测试。
2. 任务进度诊断与数据可用性盘点。
3. 冻结 target 的最小训练路线对照设计。
4. success/env steps/长尾/E2E 验证。
5. 再判断真实大小模型联训是否值得开展。

源码变更、服务操作和新采集需遵守 hybrid 边界与项目 EGL+parallel、完整 horizon 合同；没有把用户同意逐步研究解释成任意模型修改或服务操作授权。

## 8. 证据入口

- [K2 server 接受和 bonus 路径](../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)
- [K2 score/route/helper](../../投机采样/scripts/p0_batched_verify.py)
- [bonus helper](../../投机采样/wam-p0-worktree/wan_va/utils/p0_k3_b2.py)
- [实际 scheduler](../../投机采样/wam-p0-worktree/wan_va/utils/scheduler.py)
- [候选只读 KV 路径](../../投机采样/wam-p0-worktree/wan_va/modules/model.py)
- [client 环境执行与 cache handoff](../../投机采样/scripts/p0_libero_multichunk_client.py)
- [最新 matched 闭环结果](Debug-011-Phase2b-matched-closed-loop-20261009.md)
- [投机采样持续研究入口](../research/speculative-sampling/RESEARCH.md)
