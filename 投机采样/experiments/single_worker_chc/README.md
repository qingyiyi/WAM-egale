# 投机采样：默认关闭的单 worker 全层 C/HC 包装

本目录仅属于投机采样（**不是 TILE**）。公共模型、原 batch14、原 attention/backend、scheduler、接受规则均不改。无 GPU 启动器、等待队列、训练或服务重启入口；旧 GPU 实验及其 FAIL 原样保留。

## 文件与原理

- [adapter.py](../../../review/OMISSIONS.md)：实例级、源文件 SHA 固定的 AST 片段替换，覆盖全部 30 层 `attn1`。只替换只读 candidate 的历史准备段，保留原投影、norm、RoPE、输出处理及原 `attn_op`；`attn2`/FFN/公共类不改。退出或部分安装失败时恢复所有实例方法。
- **C**：每层在一个只读租约内只 gather 一次两路有效历史、只展开一次到 14 行；每次仍保留动态 `nonzero` 和行范围检查。
- **HC**：在 C 上把上述动态元数据/行检查外提一次。两臂每次都重算当前 Q/K/V，并保留 `cat(history, current)` 和原批量 attention；并非真正两路共享 prefix kernel。
- [capture.py](../../../review/OMISSIONS.md)：默认关闭；通过真实 `server.infer` 入口最多执行 52 个非 terminal Transformer 前向，捕获 29 个变化的 candidate 输入/输出，在 action terminal **进入 Transformer 前**停止，不能完成 policy chunk。保存首 candidate 前的全层有效历史到 RAM，检查 29 次历史恒定；不能拿视频更新前的旧历史替代。输入不得退化为 29 次同一 query。
- [contract.py](../../../review/OMISSIONS.md)：预算预留（失败不退款）、路径隔离、独立默认关闭开关。900 前向/45 分钟仅为冻结的**未授权 GPU 提案**，不是启动许可；目前仅有 CPU 元数据/预算原语，不是进程级安全监督器。
- [test_cpu.py](../../../review/OMISSIONS.md)：真实公开模型与服务方法的 CPU 回归；小尺寸随机权重不是实际训练好的 target。

## 默认关闭与只读租约

`SingleWorkerCHC(model)` 不修改任何模型实例；必须显式 `enabled=True` 才安装。`P0_SINGLE_WORKER_CHC_ENABLED=1` 只是调用者可以读取的本任务开关，旧任务的 GPU/优化开关不能启用它。`allow_cuda=False` 是另一个独立防线，禁止非 CPU 参数、输入和租约设备；当前没有调用 `allow_cuda=True` 的运行入口。

CPU 用法（从本目录导入；`model` 必须是原公开类的小尺寸 CPU eval 实例，生产默认要求 30 层）：

```python
with SingleWorkerCHC(model, enabled=True) as control:
    with control.lease(request="fixed-state-1", arm="HC") as lease:
        with torch.no_grad():
            for captured_query in queries:
                output = captured_query.replay(model, rows=lease.rows)
```

行映射由租约固定为 `[0]*7 + [1]*7`，值与公开服务每棵树新建的 CFG 张量相同。先验证原行映射的值，再绑定这个固定张量；不改变候选排序、batch 或条件语义。相同值的另一个张量不能在已有 C/HC 租约内冒充固定映射。

键包括请求/租约代次、全局 mutation epoch、cache_name/layer、原 cache 对象及 K/V/mask/id/is_pred 的身份/地址/version/shape/stride/dtype/device、原 backend 身份、模型参数代次。当前 Q/timestep 的变化不造成历史失效。

## 失效与边界

- 包装 `allocate_slots/update_cache/restore_cache/clear_pred_cache/clear_cache/init_kv_cache`、所有非 candidate model/attention forward、模型 `_apply/train/load_state_dict`。任何写入、回滚、淘汰、reset/观测提交、异常，都先撤销并清空全 30 层缓存。
- 直接正常 tensor 原地改值、同长度换槽、同地址恢复、metadata/model/row map 变更用 `_version` 等检测；旧租约拒绝继续，不悄悄重建。写完后需退出旧租约，在新历史下重新开租约，数值应与**新历史的 R**相同。
- 外部快照恢复必须走 `control.external_call(restore_fn, ..., reason="snapshot restore")` 或 `control.mutation(...)`；包括恢复到原地址/相同数值、失败或回滚的恢复。
- 只支持一个模型实例 owner、一个线程、一个活动租约；不能跨请求/观测/ordinary anchor/tail/terminal 保留展开缓存。每层最多一份；异常和关闭释放所有引用。
- **无法由 version 检测 `.data`/外部裸指针写入**；这些路径必须先声明 `mutation`，未知写入口不能启用复用。CPU 覆盖的是实际公开写路径，不是任意外部代码的无条件安全承诺。
- **无 version counter 的 inference tensor 直接拒绝**；不能拿地址/长度猜命中。租约自己的 rows 在外层 inference mode 下也显式创建为有版本的普通 tensor。未来真实 GPU 预检仍需核验实际历史 pool/参数是否可追踪；不支持则停止。
- 12 GiB 是展开缓存 payload 上限，不是实测 GPU 峰值。真实 BF16、2144 历史、24×128、30 层展开 payload 为 10.305 GiB；仍需为 cat/激活/allocator 留余量。

`diagnostic=False` 默认不写热路径计数，但安全键检查仍收费。未来 GPU OFF 主测必须计入冷租约构建、键检查、输入 staging 和释放，不用 CPU 用时推断 GPU 收益。

## CPU 回归命令

选用**新的、尚不存在的**输出文件名，不覆盖失败证据：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH='/rjs/cxz/论文探索/投机采样/experiments/single_worker_chc:/rjs/cxz/论文探索/投机采样/wam-p0-worktree' \
/rjs/cxz/miniconda3/envs/va/bin/python \
'/rjs/cxz/论文探索/投机采样/experiments/single_worker_chc/test_cpu.py' \
--out /absolute/new-evidence-path/CPU-tests-new.json
```

CPU 包括完整公开 30 层模型的 29 个变化查询、C/HC 逐 bit 对照、BF16、13 个其他候选隔离、历史/RNG不变、写入/rollback/eviction/reset/restore 失效、异常/显存 payload 预算/清理。2144 历史测试使用小 head geometry，**不是实际 24×128 GPU 模型**。捕获测试的 video/普通前向 ledger 使用 CPU 驱动夹具，candidate 使用真实全 30 层模型；没有执行实际训练 target 的完整 `_infer`/GPU捕获。

CPU通过后：单独确认 GPU 预算，再新建 job、补齐私有监督器与真实子进程 CPU 预检，之后才考虑一次单 worker GPU 验证。没有预算确认，不启动、不排队、不训练。
