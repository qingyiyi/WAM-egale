# 投机采样：单 worker 的 C/HC 历史 KV 复用与完整 verifier 验证方案

2026-10-12，Asia/Shanghai。状态：**CPU 静态方案完成；新包装未实现、真实接口 CPU 回归未执行、GPU 未授权/未启动。** 本文属于投机采样，不属于 TILE 算子路线，也不修改保护状态、公共模型或旧实验合同。

## 1. 本步结论与范围

- 下一阶段仅比较 **R / C / HC 三臂、一个模型 worker、一个活动请求、无驻留 peer**。H 单独的元数据外提已有局部对照，但不占用本轮完整 verifier 的主测预算。
- 保持冻结 target、A_off、原 `cache_mode=original`、7 树节点 × 2 CFG 的 batch14、BF16、原 attention 后端、scheduler、selector、阈值和 bonus 不变。
- 先用展开历史缓存检验“第 0 层收益能否保留到全部 30 层及真实变化的查询”。不是直接实现真正两路共享 prefix、新 attention kernel、mega Transformer、文本 KV 缓存或缩树。
- 本次只做 CPU 阅读、算术/AST/证据核查和 HW 记录。不得从本文或旧过夜授权自动启动 GPU、资源等待队列或训练。
- 最终目标仍为冻结 target 下成功率不劣、环境步数相近、完整策略/E2E 更快；本轮 verifier pilot 不证明这些任务指标。

## 2. C/HC 具体做了什么：现有实现与数学含义

现有实验代码在 [attention_control.py](../../../review/OMISSIONS.md) 的 `Sequence`，46–102 行；不是公共模型里的 `_c_attention`，也不是切换 contiguous/paged/lifecycle cache 的旧算子 C0/C1。本路线仍保留 original 槽布局。

原候选分支见 [model.py](../../../投机采样/wam-p0-worktree/wan_va/modules/model.py) 624–660 行：

```text
两路持久历史 pool
 → mask.nonzero 得到有序有效槽
 → pool[:, valid] 提取两路历史
 → index_select(rows) 物化成 14 行历史
 → cat(每个候选自己的当前 K/V)
 → 原批量 attention
```

行映射是 `[0]*7 + [1]*7`：每一路条件历史各复制 7 份。每棵树只有一次 batch14 target 前向，不是 14 次前向。Q 和当前 K/V 属于各自候选，不能混成同一序列。

| 臂 | 每个只读序列的历史处理 | 每次调用仍做什么 |
|---|---|---|
| R | 每次有效槽提取、gather、行检查、14 行展开 | 当前 Q/K/V、cat、原 attention、后续输出投影 |
| C | 序列开始只 gather/展开一次，保存展开 K/V | 保留动态 nonzero/行检查；当前 Q/K/V、cat、原 attention 不变 |
| HC | 和 C 一样展开一次，同时把有效槽索引/行检查外提一次 | 轻量只读租约键检查；当前 Q/K/V、cat、原 attention 不变 |

旧 micro 代码的 C 每次 `valid_indices()` 返回值用于保留工作对照，**并没有依此重新校验旧缓存是否仍有效**；HC 也没有生产级失效机制。旧 `Sequence` 的前提是整段输入/历史不变，仅测第 0 层 29 次相同 QKV，不能直接拿来跨 chunk 使用。

原理是：历史已经经过该层投影与位置编码；在持久历史不变的连续验证中，重复 gather 和 2→14 行展开没有新增数学信息，因而可以复用其结果。当前候选仍重新进行 Q/K/V 投影、RoPE、attention 和其他 Transformer 计算。C/HC 节省的是重复准备/物化，不是减少候选数、替代大模型或复用旧 direction。

旧真实 GPU 局部值为 R/C/HC = 86.900/43.726/39.781 ms，包含 setup，C/HC 减少 49.7%/54.2%。这里只支持进一步实验，不可乘 30、不可推完整推理翻倍。旧整轮在双需求重叠门 FAIL、ON 未执行，全部原样保留。见 [部分结果](../../../投机采样/results/runtime/p0-kv-sync-attribution-20261012-014447/CPU-PARTIAL-ANALYSIS.md)。

## 3. 真实历史生命周期：哪些改变只是状态，哪些会改 KV

| 路径 | 源码 | 对历史的影响与新包装要求 |
|---|---|---|
| batch verifier | server 1907–1942；model 624–660 | `candidate_cache_rows` + `update_cache=0`，直接读取历史并 cat 当前 KV，不写原 pool；可在合法只读租约内命中 |
| 接受/拒绝、route execute | server 1215–1237、1320–1377 | 改 `actions/previous_direction/step_index` 与 scheduler 状态；当前路径不直接写持久 KV，不因“接受深度变化”本身失效 |
| 普通 anchor/尾部 target | server 1126–1139；model 669–685 | 即使 update=0，也先写临时槽再 `restore_cache`；必须在前向前失效，不能当成只读 candidate |
| action/video terminal | server 1379–1392、2310 | update=1 真正提交预测 KV；写前失效 |
| 新观测替换 | server 2640–2670 | `clear_pred_cache` 后 video/action update=2；全部失效，不跨观测盲目复用 |
| 分配/淘汰/写入/回滚 | model 547–594 | 可改有效槽、值、id、is_pred；任何变更均失效 |
| reset、清空、重建 | server 2111–2145；model 338–434、881–897 | 更换缓存和请求身份，全部失效 |
| 快照恢复 | fixed_load_cost/snapshot.py 55–63、94–98 | 对同一地址进行 copy；恢复相同数值也必须增加代次，不能把地址相同当命中依据 |

**关键风险：** original 的 `allocate_slots` 在临时 update=0 前向中也可能先淘汰历史；`restore_cache` 只将新写槽的 mask 设为 False，不自动恢复被淘汰的旧槽。因此“前向后长度一样”“最后 mask 一样”“只看 id 最大值”都不足以证明历史不变。失效覆盖所有写路径，比事后按长度猜测安全。

## 4. 拟议复用单元、键与失效规则（尚未实现）

### 4.1 只读租约，不建永久无界缓存

缓存单位是 **当前 worker / 请求 / cache_name / layer 的一份历史展开 K/V**。同一时刻只运行一个臂，每层最多一份展开缓存。请求/臂结束、异常或失效时释放 Python 引用；不得同时保留 C 与 HC 的全层缓存，不依赖每步 `empty_cache()`。

租约状态为 `EMPTY → LIVE → INVALIDATED → EMPTY`。每个序列从 EMPTY 开始，首 candidate 按层惰性构建；初始化、键检查、释放开销均计入主序列耗时。重建只能发生在确认新的合法只读上下文之后，不在已失效的 LIVE 租约里悄悄忽略写入。

### 4.2 键

```text
worker/model_generation, request_generation, lease_generation,
cache_name, layer_id, cache/allocation_generation,
history_mutation_epoch, slot_order_generation,
CFG_semantics_and_row_map_generation, batch_geometry,
device, dtype, heads, head_dim, frozen_model/backend_identity
```

- `history_mutation_epoch` 是任务包装维护的单调 host 代次，写前失效、异常也失效；不仅统计新增 token。
- 覆盖 `allocate_slots/update_cache/restore_cache/clear_pred_cache/clear_cache/init_kv_cache`、快照 restore，以及所有非 candidate target 前向。先用全局代次保守失效全 30 层，暂不追求按层细粒度保留。
- 普通 B2 target、video、terminal、观测提交均走原函数；包装只在允许的 original/read-only/batch14 action candidate 路径替换历史准备部分。
- rows 必须由包装按 host CFG 契约生成并保持不可变；不能仅缓存 CUDA tensor 的地址。row map 改值、顺序、形状或 CFG 语义均开启新代次。
- 历史槽顺序保留原 `mask.nonzero()` 次序。池原地改值、同长度换槽、恢复到同地址、淘汰后又回到原长度都失效。
- Q、当前 K/V、当前 timestep、当前 query grid 的变化本身不改变历史，因此**不把当前 Q/timestep 放进历史键**；它们仍重新计算。历史位置/条件、模型权重或 backend 变更则不得沿用旧缓存。
- 不在 OFF 每步 hash/复制整段 GPU history，避免诊断同步侵入测速；正确性关在计时外核验所有 30 层有效值、mask/id/is_pred、rows、RNG。无法证明所有写入口都受控时拒绝打开租约，不自称 HC 已安全。

### 4.3 应失效的十二类测试

1. 预测 terminal 提交；2. 观测 update=2；3. clear_pred_cache；4. clear/reset/recreate；5. snapshot restore 到原地址；6. 同 mask/同长度的有效 K/V 原地改值；7. 同长度不同有效槽/次序；8. 满容量 allocate/eviction；9. 普通 update=0 的写入＋rollback；10. CFG row map/语义变更；11. 请求或 cache_name 切换；12. device/dtype/backend/geometry/model generation 键失配。

在第 12 类中，合法同设备输入的键代次变更应重建；实际不支持的 dtype/device/backend/形状应在 GPU 调用前拒绝，不能换 backend 继续测。CPU 还需覆盖非法 rows、空历史、异常中断、默认关闭和每层缓存释放。真实写入后的比较对象是**同一新历史下的 R**，不是强行要求新历史等于变更前历史。

### 4.4 实现边界

未来可实现任务专用、默认关闭的实例级 adapter：只替换 `attn1.forward` 中 624–660 的候选历史准备，保留 608–623 的投影/norm/RoPE、原 `attn_op`、687–690 的输出处理及其余公共代码。**只包装 `attn_op` 太晚，原 gather/index_select/cat 已经完成，无法消除其成本。** 用固定公共源码 SHA/AST 核对复制边界；不是修改公共类或 monkeypatch 所有模型。`attn2`/文本条件、FFN、scheduler 不改变。

本次不创建这个 adapter、不修改旧封存的 `Sequence/capture/worker`。路径重定位、job 所属断言、spawn 后状态更新等必要可靠性修复也留待下次 CPU 实现，双 worker 结束屏障不再列入主线。

## 5. 单 worker 显存预算

### 5.1 静态载荷

几何来自真实旧 capture：BF16，history=2144，current=16，24 heads × 128，30 层。K+V 字节公式为 `2 × rows × history_tokens × 24 × 128 × 2`。

| 项目 | 每层 | 30 层 | 说明 |
|---|---:|---:|---|
| 两路有效历史 | 50.25 MiB | 1.4722 GiB | 原 pool 的有效部分；不是新增缓存，不重复计入 |
| C/HC 的 14 行展开历史 | 351.75 MiB | **10.3052 GiB** | 新增保留载荷；C 与 HC 串行，不能加倍 |
| 14 行 history＋current 的 cat 输出 | 354.375 MiB | 不按 30 层常驻相乘 | 顺序处理的临时输出；仍在 R/C/HC 中存在 |
| 构建时两行有效历史临时 gather | 50.25 MiB | 不按 30 层常驻相乘 | 一层构建完就释放，不保存第二套逐层两行副本 |

这些是 payload，不是实测 HBM 流量，也不能把所有行直接相加当峰值。模型权重、原容量池、VAE、CUDA context、激活、SDPA workspace、allocator reserved/碎片、输入暂存都要另计。

### 5.2 历史显存参考与未来准入

旧 ownership 记录显示该 A100 总量 **40 GiB**，driver reserved 约 0.5065 GiB；旧双模型每 PID 观测最大约 **14.4375 GiB**。这是离散采样、不同实验生命周期的参考，**不是新单 worker 的峰值**，不取双模型总量直接当单模型预算。

新验证暂定：

- 先在**无 peer 的 R**上实测原模型峰值、NVML allocated/free/reserved、PyTorch allocated/reserved；不可把 allocated 与 reserved 重复相加。
- C/HC 相对 R 的额外峰值上限 **12 GiB**：覆盖 10.3052 GiB 展开载荷及临时准备余量，但是否足够仍待 GPU 验证。
- R 资源探针后、开 C/HC 前 NVML free 必须至少 **20 GiB = 12 GiB 额外额度＋8 GiB 安全余量**；进入任一臂后 free 不得低于 **8 GiB**。
- 每序列检查峰值与结束时 live 引用释放；allocator 可保留 reserved，不能据此把引用泄漏忽略掉。
- golden 输出/历史参照存 CPU，不驻留第二个 target、不保留 30 层 GPU golden pool，不同臂不同时展开。CPU 可用内存至少 10 GiB、磁盘至少 20 GiB；诊断数据最多 1 GiB。
- 余量不足、峰值超限、增长泄漏或历史长度/容量不符就停止。不临时缩候选、不混入“部分层缓存”凑成绩，也不新增共享-prefix kernel；那是另一份需确认的方案。

目前只完成以上 CPU 算术和历史记录提取，**不能宣布显存准入已通过或一定不会 OOM**。

## 6. 完整 verifier 的拟议 GPU 验证

### 6.1 数据：真实变化的查询，0 新完整 policy chunk

复用 [T-init3-chunk14.pt](../../../review/OMISSIONS.md) 及 JSON 元数据；本次只 stat/核对 JSON hash，没有加载/重新 hash 1.53 GB PT。未来 GPU 工作需实际检查 PT state/request hash。

按已验证的 reset→A_off 预加载→快照恢复顺序，在单 worker 运行原视频 21 前向，再运行原 action 去噪路径，采集该真实 A_off 状态上的 29 棵树的**原始 candidate 模型输入与参考输出**。这些输入来自实际 draft/原 scheduler/route，随 timestep/候选状态变化，不重复同一 QKV、不造随机查询替代。

在 action terminal `update_cache=1` **进入 Transformer 之前**截断；不返回完整 policy chunk、不执行环境动作，不增加 episode/EGL。既有计数为 video21＋普通 action2＋candidate29＝52 个 target 前向；若实际路径/数量不符、历史在 candidate 间不一致、无法按此边界停止，就失败停止，不延长捕获或补跑完整 chunk。

这是一份诊断 replay 输入/输出，不是 teacher 训练数据，不训练。采集完后回到核验过的 post-video/anchor 有效历史；每个序列 29 个真实变化输入，全 30 层正常 forward。history 内容/有效槽各层稳定后才开缓存租约；原 input pack 到 device 的准备、lease/keys/setup/释放都计时，hash 与 golden 比较在计时外。

### 6.2 关卡与通过标准

| 关 | 通过条件 | 状态/失败行为 |
|---|---|---|
| 当前 CPU 方案核查 | 原源码/几何/slot 生命周期/静态内存/顺序/预算一致；模型、保护和历史结果未变 | 本步静态核验；不替代包装回归 |
| 下一步真实接口 CPU 回归 | 默认关闭走原路径；只替换 candidate 历史片段；失效十二类和异常；真实预加载/捕获停止接口、计数、job 路径隔离、spawn 状态、timeout/owned cleanup | 未实现、NOT_RUN；不过就不申请 GPU 启动 |
| 单 worker 恢复与采集 | state/request/RNG 恢复；一个模型无 peer；只读候选输入有真实变化；29 树/2144 history/batch14/30 层计数符合；截断前无 action terminal | 未授权、NOT_RUN；不重试、不增 fixture |
| 全 verifier 数值与隔离 | C/HC 对全部 29 个输入的最终 output 与 R **逐 bit 相等**；30 层有效 KV/metadata/RNG 不变；相同 dtype/stride/backend；改变一个候选的当前输入不改变其余 13 行 | 不能放宽数值容差/换 attention，失败停止 |
| GPU 失效行为 | 十二类中 C/HC 实际 drop/rebuild 或拒绝；变更后与新历史下 R 逐 bit 一致，不能仅看命中计数；每臂隔离恢复参考历史 | 旧 expanded tensor 不得命中；不支持的接口 fail closed |
| 显存 | R 探针 free≥20 GiB；额外 peak≤12 GiB；free≥8 GiB；序列结束 live 缓存清理、无增长泄漏 | 不满足即停止，不换方案偷跑 |
| OFF 有效性 | 三臂各 6 次、全部预注册顺序/原始值保留；MAD/median≤7%；setup/lease/input staging/释放计费；无逐层 profile/hash/tensor→host 数值检查混入 timer | 无效/不确定就记录停止，不补重复 |
| 性能晋级 | 长单 C 或 HC 完整 verifier 序列中位数比 R 至少低 **5%**，且 6 个配对顺序块均更快；同时正确性、资源有效 | 只标“完整 verifier 局部 pilot”；不证明全 chunk/E2E/质量 |
| ON/收尾 | 仅 R/HC 各一独立序列记录实际 cache 准备次数和阶段；ON 不混 OFF、不把 event span 当 kernel 忙时；最后原路径复核/所有 owned 进程退出，源码和旧结果不变 | trace/finish 未完成须 NOT_RUN；保留阶段性与负结果 |

性能变慢或不满足 5% 可是**有效负结果**，不是运行故障；不因慢而删样本重测。单 fixture 六次只是时间重复，不是六个独立 episode；无非劣统计功效/跨任务保证。

### 6.3 预注册主测顺序

```text
R,C,HC     C,HC,R     HC,R,C
R,HC,C     HC,C,R     C,R,HC
```

三臂各 6 次，每个位置各 2 次；每个有序相邻臂对各 2 次。每次是完整 30 层 × 29 真实变化 candidate，不是 29 次相同第 0 层 QKV。每次从 cold lease 开始，禁止给 C/HC 免费使用上一轮展开缓存。各臂使用同一份 CPU 有效历史快照和同一恢复/hash 流程，边界外不额外调用 target 或隐性增加 warmup；这些恢复/hash 开销另报并进入任务安全计时，不冒充部署 E2E。

### 6.4 GPU 预算：提案，未授权

| 用途 | target 前向上限 |
|---|---:|
| 原 partial replay 采集 52＋原 candidate 资源探针 1 | 53 |
| C/HC 全 29 输入逐 bit 比较 | 58 |
| R/C/HC 各一次 29 输入 warmup | 87 |
| 正式 OFF：3 臂 × 6 次 × 29 | 522 |
| 十二类失效 GPU 探针：每类最多 9 个含原 mutator 前向/前后参照；臂间串行恢复 | 108 |
| 候选隔离：3 臂 × 原/扰动各 1 | 6 |
| 独立 ON：R/HC 各 29 | 58 |
| finish：三臂各 1 只读前向/原状态复核 | 3 |
| **已分配总额** | **895** |

- 硬上限 **900 个完整 target Transformer 前向**，包含 video、普通/terminal 失效探针、candidate；未分配的 5 个不是失败重试或补测授权。额外 micro attention 调用 **0**；模型内部 attention 不另冒充 micro。
- **0 完整 episode、0 EGL 初始化、0 完整 policy chunk、0 执行环境步，最多 1 个模型 worker/1 次 GPU 尝试、1 个既有 fixture。** CPU 监督器不是第二个模型 worker。
- 暂估 GPU 阶段 **15–25 分钟，45 分钟安全硬限**，从唯一模型 worker 第一次启动计时，含加载、恢复、CPU↔GPU staging、诊断、比较和清理；不含未来另批准的排队时间。估计依据旧 ~2.6 分钟 partial/micro 任务、约 0.36 秒旧双驻留原 candidate warmup跨度及有界全层工作，非新独占单 worker 实测，不保证时间。
- 计数在进入 GPU forward 前预留，异常 forward 也算；超时/资源/数值/协议失败就停止，不补局、不重试、不自动转训练或另方案。
- 只使用 owned PID/PGID/私有端口；不干预别人的 GPU 进程。必须重新通过已修复 NVML v2 分配内存/归属 CPU 回归；本次没有调用 NVML/CUDA，没有新资源等待任务。
- 不安装依赖。ON 使用已有兼容工具；缺失或不兼容如实停止/NOT_RUN，不能把带 profile 的值替代 OFF 主测。

## 7. 下一步与授权边界

**先请用户确认仅 CPU 的实现范围：** 新建任务专用、默认关闭的单 worker 全层 C/HC adapter、失效租约、真实变化 query 捕获、计数/资源/路径/状态监督包装及 CPU 回归；公共模型、原 batch/attention 不改。不能复用旧“第 0 层冻结 QKV”包装冒充完整 verifier 实现。

CPU 回归通过后再单独确认本方案的 **1 worker、0 episode/EGL/full chunk、≤900 target、45 分钟** GPU 预算，并创建新 job；当前 CPU 规划授权不是实现或启动授权。不再做双 worker 重叠/结束屏障优化。

若完整 verifier 无净收益或显存不合格，记录结果后先 CPU 分析：缩小缓存生命周期/真正共享 prefix/文本重复投影分别规划，不自动扩大；若通过，再另规划完整 action/chunk 与冻结 target 自然闭环，检查成功率、环境步数、长尾、E2E。训练、RL/KL、大小模型联训和 Fast-WAM 迁移仍未授权。

## 8. 可定位源码与证据

- [当前 C/HC 微对照实现](../../../review/OMISSIONS.md)：46–58 初始化，64–80 元数据/gather/展开，81–95 各臂调用，98–102 次数账本。
- [当前首次真实 verifier 捕获](../../../review/OMISSIONS.md)：19–68；现有代码只捕获第一次并停止，**尚不能捕获全 29 树**。
- [当前旧 GPU worker](../../../review/OMISSIONS.md)：micro 的 timer 先于 `Sequence` setup；完整 candidate 只 replay 原路径，**没有全层 C/HC adapter**。
- [公共 attention 原路径](../../../投机采样/wam-p0-worktree/wan_va/modules/model.py)：338–434、547–690；[server](../../../投机采样/wam-p0-worktree/wan_va/wan_va_server.py)：1126–1139、1379–1392、1907–1942、2111–2145、2640–2670。
- [CPU 静态核验与机器提案](../../../review/OMISSIONS.md)；[PLAN.json](../../../review/OMISSIONS.md)；[历史资源读取](../../../review/OMISSIONS.md)。

## CPU 初稿核查说明

第一版预算行合计误写为 995；CPU 算术核查实际为 895，已将硬上限同步收紧至 900（未分配 5 个不用于重试/补测）。保留 [初稿失败核查](../../../review/OMISSIONS.md) 与原 draft；不是 GPU 运行失败，也没有消耗新 GPU 预算。
