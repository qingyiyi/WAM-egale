# 投机采样：单 worker 全层 C/HC 实现与 CPU 回归结果

2026-10-12，Asia/Shanghai。**本步完成：默认关闭的私有包装＋真实接口 CPU 回归，81/81 通过。GPU 未授权/未启动，未训练。** 本文不是 TILE 算子实验，也不把 CPU 结果写成真实 GPU 性能结果。

## 1. 完成的工作

新增 [任务目录](../../../投机采样/experiments/single_worker_chc/README.md)，不修改公共模型、旧实验或保护状态：

| 文件 | 本步实现 |
|---|---|
| [adapter.py](../../../review/OMISSIONS.md) | 实例级、源 SHA 固定的 30 层 C/HC；机械 AST 替换仅限候选历史准备段；原投影/norm/RoPE/输出处理、cross-attention、FFN、原 SDPA 保留 |
| [capture.py](../../../review/OMISSIONS.md) | 默认关闭的 29 个变化查询捕获；52 个非 terminal 前向后，action terminal 进入模型前停止；首候选前历史保存在 RAM；检查 29 次历史恒定 |
| [contract.py](../../../review/OMISSIONS.md) | 显式任务开关、CPU 默认设备边界、预算失败不退款、Unicode/绝对路径与 symlink 逃逸拒绝；仅元数据/预算原语，无 GPU 启动器 |
| [test_cpu.py](../../../review/OMISSIONS.md) | 导入真实公开 `WanTransformer3DModel`/`WanAttention`/`VA_Server`，小尺寸 CPU 模型数值与生命周期回归 |

**C/HC 的数学含义不变：** 历史仍从两路 pool 展开到 14 行，但每个只读租约/层只展开一次。C 保留每次动态元数据与行检查，HC 将这些外提；每次当前 Q/K/V、cat、原 attention 仍重新计算。没有实现真正两路共享 prefix kernel、mega Transformer、缩树或减少 denoising steps。

公开服务每棵树新建的 CFG 行映射均为 `[0]*7+[1]*7`。包装先检查原值，再在租约内绑定一份可追踪版本的固定映射，避免身份变化阻碍缓存命中；不改变 batch14、候选顺序或条件语义。

## 2. CPU 验证结果与证据

最终 [81 项结果](../../../review/OMISSIONS.md)；[完整日志](../../../review/OMISSIONS.md)。全部通过，无跳过；`CUDA_initialized=false`、禁止 CUDA 初始化的哨兵触发 0 次。

| 类别 | 通过数 | 主要检查 |
|---|---:|---|
| 默认关闭/实例边界 | 18 | 无开关不 patch、源 SHA、原 SDPA 身份、30 层、eval、重复安装/线程/嵌套拒绝、部分安装恢复、非 CPU 输入在公开数学前拒绝 |
| 数值/隔离 | 10 | 完整公开 30 层、29 个变化查询，C/HC 与 R 逐 bit；30 层 BF16；改一候选其余 13 行不变；有效 KV/metadata/RNG 不变；空历史、2144 历史、关闭诊断、固定 rows |
| 失效 | 27 | 真实 update1/update2、预测清除、reset/recreate、同地址 snapshot restore、K/V 原地写、同长度换槽、id 变更、满容量淘汰、update0 rollback、rows/请求/backend/模型代次/stride/dtype 变化、无 version 池拒绝 |
| 异常/清理 | 5 | 单层/中途 payload 超限、candidate/调用者异常、实例方法恢复、缓存引用释放 |
| 服务接口 | 4 | 真实 CFG 输入打包、实际 reset 清全 30 层、reset 后冻结 A_off 实际 CPU 加载与权重一致、公开模块路径 |
| 变化查询捕获 | 10 | 真实 infer/candidate 入口、29 个变化查询与首候选历史、52 调用/terminal 前停止、重复 query/期间历史变化/候选写历史/中间普通前向/提前 terminal/异常拒绝与方法恢复 |
| 预算/路径 | 7 | 900/45 分钟提案仍未授权、失败不退款、deadline、拒绝扩预算、Unicode 重定位/逃逸拒绝、CUDA 不初始化 |
| **合计** | **81** | **CPU 正确性关通过，不是 GPU 性能关** |

每层连续 29 个候选的诊断断言（C 与 HC 均逐 bit 一致）：

| 操作 | C | HC |
|---|---:|---:|
| 历史 gather/14 行展开 | 各 1 次 | 各 1 次 |
| 元数据 nonzero/行范围检查 | 各 30 次：首构建 1＋每次 29 | 各 1 次 |
| cat 与原 SDPA | 各 29 次 | 各 29 次 |

这说明 CPU 包装确实复用历史而非复用旧输出；**操作计数不等于真实 GPU 延迟收益**。诊断默认关闭，OFF 仍须计入合法键检查和构建/释放开销。

### CPU 模型范围

- 完整 30 层使用公开原类，但仅 2 heads × head_dim4、随机 CPU 权重；验证原 forward 及后续 cross-attention/FFN/输出处理，不是实际 24×128 target。
- 2144 历史测试使用真实 attention 接口、小 head geometry、单层；不能当作实际全层显存/GPU 数值验收。
- 捕获测试通过实际 `VA_Server.infer` 和候选打包方法，candidate 使用真实全 30 层模型；video/普通调用账本使用 CPU 驱动夹具。**没有执行实际训练 target 的完整 `_infer` 或实际 GPU 29 树捕获。** 首候选前历史检查/恢复由真实 snapshot 接口验证。
- 冻结小模型 A_off（约 112 KB）确实在 CPU 加载，其 SHA 与原提案相同；训练好的大 target 未加载，1.53 GB 原 fixture 仅 stat。
- 公开 autocast 声明产生“CUDA 不可用，禁用”警告；这是 CPU 导入/执行提示，没有 CUDA 初始化、GPU 任务或显存分配。

## 3. 失效与安全边界

1. 写前/写后 mutation epoch 保守撤销全层租约；普通 update0 也可能写入、淘汰、rollback，绝不视作只读 candidate。
2. 覆盖公开 allocation/update/restore/clear/init、所有非 candidate model/attention forward、模型 `_apply/train/load_state_dict`。同地址恢复也撤销；异常不能重启旧租约。
3. 正常原地写由 tensor version 等检测；键变化或失效后旧租约拒绝继续，退出后方可在新历史开新租约，与新历史的 R 对照。
4. 外部 snapshot 必须走 `external_call`/`mutation`。单 worker、单线程、单活动租约；未知写入口不能开启缓存复用。
5. **`.data`/裸指针写不能只靠 version 发现**；必须有显式 mutation 边界。无 version 的 inference pool/参数直接拒绝，不按地址或长度猜命中。未来真实部署需另核验这些条件。
6. 每层最多一份展开历史，租约结束/异常/关闭清引用；12 GiB 是 payload 上限，不是 allocator/激活峰值。真实 BF16 几何下 30 层新增展开 payload 静态约 10.305 GiB，真实显存门尚未运行。

## 4. 失败记录与完整性

所有原始 CPU 尝试保留，没有覆盖后挑选结果：

- `001`：72 项、5 error；捕获驱动夹具未实现公开 reset/cache 方法。改用实际模型子类提供原接口，不修改公共服务。
- `002`：修复后 72/72 通过。
- `003`：增加固定历史段检查后，79 项中 1 个断言失败；提前结束夹具先执行了普通 tail，因此被更早的“不允许中间普通前向”门拒绝。修正夹具使其直接触达提前 terminal；**没有放宽该门**。
- `004`：增加原 SDPA 身份及候选写历史拒绝检查后，最终 81/81 通过。

[源码/范围核查](../../../review/OMISSIONS.md)：192 个冻结文件 SHA 不变，包含公共模型/服务、旧实验源码、旧 GPU 原始结果与旧 CPU 方案；state/rules 不改，缺失的 cycle 不创建。9 个 target 文件只核对 size/mtime，没有全量 SHA 或加载。TILE 路线不改、旧 GPU FAIL 保留。本步为同一代理自审与回归，不宣称独立审计。

## 5. 当前判断与下一步

**阶段性进展：** 已从“第 0 层固定 QKV micro/CPU 静态方案”推进到“默认关闭、全 30 层、变化查询、可撤销租约的真实接口 CPU 正确性实现”。现在值得做真实 GPU 验证，但不能提前认定完整 verifier 更快。

尚未做：真实 target 全层 GPU 数值/显存/耗时、完整 policy/chunk/E2E、成功率与环境步数；没有训练/RL/联训/mega。GPU 监督器/新 job 子进程预检还需在下一步补齐，不能因本次回归通过就跳过。

**下一步须单独确认 GPU 预算：** 一个模型 worker、无 peer，0 完整 episode、0 EGL、0 完整 policy chunk/环境动作；895 个已分配 target 前向、900 硬上限，安全限时 45 分钟（原估计 15–25 分钟，不含排队，非保证）。5 个未分配调用不是重试或补测授权。

预算确认后，新建隔离 job → 私有进程级监督器/真实子进程 CPU 预检 → 独占资源/路径/归属/限时检查 → 恢复与真实 29 输入捕获 → 原 batch/backend 下数值、失效、隔离/显存关 → 计入冷构建/检查/staging/释放的 R/C/HC 全 verifier 成本。失败停、不补跑、不训练。

若达到既定稳定性门及完整 verifier pilot 收益门，再单独规划完整单 worker policy/闭环；若无净收益，先根据完整测量分解 cat/attention、全层缓存显存压力及 host key 开销，不直接扩成 mega 或联训。最终目标仍是成功率不劣、环境步数相近、E2E 更快。

[机器结果汇总](../../../review/OMISSIONS.md)；历史 [冻结 CPU 方案](SINGLE-WORKER-CHC-KV-PLAN-20261012.md) 与其 900/45 分钟提案保留原样。
