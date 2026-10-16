# 完整 episode 投机 E2E 与轨迹长度基线

## v3 有限输出磁盘方案

v2 CPU21项通过；GPU启动前磁盘检查时只有约39.66GiB，低于原先40GiB启动值，
因此没有初始化GPU、没有新前向和episode；该失败与全部CPU证据冻结保留。
本轮仅D/S共4个episode，无新GPU snapshot/teacher/full trace。新增job输出硬限1GiB；
启动需要16GiB，运行保持12GiB余量并每5秒核对输出体积。16GiB > 12GiB+1GiB，
为这个有限任务留足余量；当前可用约39.66GiB。无需删除或覆盖旧证据。
模型、客户端、计时、共享计算、预算携带和v2启动身份修正均不变，CPU重新绑定新plan。
以下旧方案作为历史设计保留，由本节覆盖磁盘启动值，v2节覆盖任务数量。

## v2 补齐与编排修正

v1 的真实完整 chunk GPU probe 已逐位通过，T 两完整任务均成功（init3=293、init4=300步）。
进入 D 时，新客户端 /proc cmdline 瞬时不稳定触发身份校验；此前创建后未登记的客户端遗漏于清理。
只按精确 PID/start_ticks/UID/PGID/完整 argv/任务环境恢复清理该客户端，没有外来进程信号；
v1 FAILED.json 保留，额外清理间隔全部计入 GPU 时间。累计17551 intents（两个未闭合D预留各4000已全额收费）、
5705.285101825371 GPU秒。RECONCILED-CLEANUP.json 记录 GPU 显存和进程清空。

v2 只补 D/S 各两完整 episode，共4次新 EGL/4 model workers，沿用原累计35000前向/18000秒硬限。
每 episode4000，stage5400秒；不新跑probe，复用被冻结的逐位 PASS。
v1 T完整任务和计时按唯一(init,arm)复用，不删除、覆盖或重复计数。
online_shared/runtime/client/server及三个S0 wrapper全都逐字节同v1；只修编排、范围与引用。
新进程在Popen后立即进入清理清单，等待最多2秒让cmdline稳定；失败清理刚创建且身份未变的PID。
这只是元数据观测等待，不是模型、EGL、episode或任务重试。

以下v1设计作为原始口径保留；本节覆盖新数量、累计携带和复用范围。

用户已确认：E2E 指完整 LIBERO episode，trace 长度指任务环境步数。
先保留有收益的 self-attention S0；停止扩展 prompt/cross-KV 和行证明优化。
此前数值与性能证据保留，不把 verifier 序列的加速推断成完整任务加速。

## 执行清单

- [ ] CPU：实际生产 loader、在线 lease、普通前向失效、跨 chunk 清理、预算与客户端检查。
- [ ] GPU：固定真实短 KV 快照下原投机 D 与共享投机 S 完整 chunk 逐位比较。
- [ ] E2E：T 原 target、D 原 K2B2 投机、S 同 draft + S0；每臂初态 3/4，各一个完整 episode。
- [ ] 汇总：逐初态和总量速度比、成功状态、环境步数、接受深度、完整输出与轨迹一致性。
- [ ] HW：保留旧证据，写结果和下一步轨迹长度控制建议。

每臂两台独立服务和两个 EGL 客户端并行；三臂分批，等待该批服务都就绪后启动客户端。
client elapsed_sec 包含环境 reset/稳定、prompt、所有 draft/验证/回退、动作提交、环境步进、KV 更新；
环境构造和模型服务 bootstrap 不包含在原客户端 elapsed_sec 中，另外记录完整外层 wall。
成功或完整 800 horizon 才结束，不设 policy chunk 或轨迹截断。
主要速度比 T/S；D/S 表示共享给既有投机额外带来的净收益。两初态只是 pilot，不宣称全任务稳定收益。

所有臂 profiler、teacher、完整 server trace、逐步重型 state/RGB 诊断关闭。
共同 RAM 计数 hook 及原有异步输出保存保留；计数无逐前向 fsync，无 CUDA event。
轻量客户端仅把实际动作存 RAM，初末状态在任务计时结束后保存。
比较同初态初始 simulator/controller/RNG 与所有 D/S 输出、命令、最终状态。
任务长度目标为成功不下降且 S/T 环境步数 <= 1.05；先测原有策略，后续控制实验另立方案。

## 授权与资源

沿用用户“中途你自己决定就行，我只要结果”“GPU 消耗上限暂时拓展到 5 小时”的授权。
本轮完整任务由用户明确要求，不改任何旧零 episode 方案；所有代码在本实验目录，默认关闭。
累计先前 6641 次 target intent、4994.309834191576 GPU 秒，失败尝试已计入。
本轮 GPU stage 最多 5400 秒；总累计仍 <= 18000 秒。每完整 episode 最多 4000 target forwards，
精度 probe 最多 400，6 episodes、6 EGL 初始化、7 model workers，无隐式重试。
为覆盖完整 horizon，代理在用户自主授权内设置累计前向有限硬限 35000；这不是用户指定的数值。
每 worker 启动前持久化整份预留；缺少干净终结计数则按整份预留收费，不退款不隐藏失败。

启动前：冻结代码、目标/draft、旧实验/HW 证据与受保护 state/rules/cycle；CPU gate；
互斥 GPU 锁；NVML 连续 15 次空闲，compute/graphics 都无外来进程；验证 EGL、2 workers、端口、trace/provenance。
启动至少 40 GiB 磁盘可用，运行低于 12 GiB 即停；只清理精确身份归属本实验的进程组。
原始 attention/scheduler/阈值/CFG/cache 不改；无训练、teacher、RL、安装、下载、已有服务重启。
