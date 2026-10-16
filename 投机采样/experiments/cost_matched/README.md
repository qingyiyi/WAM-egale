# 同负载 T/A_off 成本诊断包装

本目录仅属于**投机采样 action-only**，不是 tile/算子路线。2026-10-11 用户只授权实现与 **CPU** 验证；**未授权或启动新的 GPU、真实 LIBERO episode、模型服务、teacher 采集或训练**。

## 文件与边界

- `runtime.py`：双 worker loopback 请求 cohort；client 请求元数据、屏障/transport 账本；工作计数、区间重叠、匹配和对账。
- `server_adapter.py`：本任务进程内包装现有 `VA_Server.infer`、profile finish、transformer forward 和各 block 的 callable `attn1.attn_op`。原函数照常调用，返回值不改，不改公共源码或权重。
- `entry_guard.py`、`client_entry.py`、`server_entry.py`：默认拒绝执行；先检查独立 GPU 授权、原 cache/torch attention、种子、无 teacher/trace 等，再导入真实环境/模型。
- `campaign.py`：CPU prepare-only；未授权计划 `gpu_run_approved=false`。保留独立候选 8 局方案 T→A_off→A_off→T，各双 EGL worker、首局 init3/4。继承的 20 局训练入口不调用；模式、冻结 A_off、完整 horizon 与预算受限。**当前不运行 GPU 入口。**
- `analyze.py`：离线 CPU JSONL 对账。显式物理阶段标签缺失时不强配；输出 unmatched 和完整 single-demand 尾部。拒绝覆写已存在汇总。
- `tests/test_cost_matched.py`：模拟请求、实际 CPU 进程/TCP、fake-env 正式 client，以及从当前公开源码 AST 提取的实际 self-attention/infer/profile 方法合同；不是载入 WAM 权重或运行真实 LIBERO。

## 请求与成本合同

1. 两个活跃 worker 按 reset→policy0→cache0→policy1… 同类别/序号/帧 chunk 组成 cohort；错配、重复、超时或异常共同停止，不把丢失 peer 偷换成低负载。
2. **正常完成一个完整 episode 后**，另一 worker 继续完整 episode/horizon，单需求尾部保留；不添加假请求，不截断、不重试补局。
3. 分开记录 client 屏障等待、transport 区间、server 原函数区间。队列同步≠实际 server 区间重叠，区间重叠≠GPU kernel 同时执行；保存相同主机 boot 时钟范围。
4. 每次 transformer forward 保留分支/cache 更新/实际 CFG batch、输入 shape；从有效 KV 被选择之后的 attention 调用读 Q/K/V shape，不以分配的 cache pool 容量冒充 KV 长度。只读 shape/dtype，无新增 tensor 取值、CUDA 同步、噪声生成或额外模型调用。
5. **已有 phase profile 会 CUDA 同步**；本包装还有 Python/记录开销。所有 timing 是诊断，不是 clean E2E；server 区间之外的 sidecar 构造/写入仍在 transport 内，不把残差叫纯环境时间。
6. T 的空 counters 不是零工作：读原 phase.calls，并与实际 forward 对账。历史 profile 无实际 forward sidecar 时 actual calls/rows 为 null；reset/cache 内部 phase 缺失保持 null，边界与实际 forward 仍记录。
7. 保持 ordinary K2（非 bonus）：七逻辑候选/十四 CFG 行、terminal 单列、50 action/20 video scheduler updates 不变。拒绝边仍可能使用 verifier 已计算的 target 方向提交一步，**scheduler fallback 不等于再调用一次 transformer**；实际调用与 committed 更新分开对账。
8. 只按公共 provenance、worker/task/init、frame、明确物理阶段、每行 Q/K/V shape、planned demand 与 observed overlap 类别分层。batch 倍数和调用次数是成本结果，不是匹配键。记录 overlap 数值/比例；相同类别并不保证同等 GPU 争用。缺字段/没有对侧样本→unmatched，不强配不同物理阶段的同序号 chunk。
9. 输出请求分布不是独立 episode 样本；观测/隐藏历史并非完全相同。rows×calls 不是 FLOPs；不从 14/2 直接推断耗时倍数或宣称成功率非劣。

## 允许的 CPU 用法

在项目根目录，使用已有环境，不安装新依赖：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/rjs/cxz/miniconda3/envs/libero-va/bin/python -m pytest -q \
'/rjs/cxz/论文探索/投机采样/experiments/cost_matched/tests/test_cost_matched.py' \
'/rjs/cxz/论文探索/投机采样/tests/test_p0_minimal_frozen.py'
```

可在**新的**临时目录准备未授权 JSON 合同：

```bash
/rjs/cxz/miniconda3/envs/libero-va/bin/python \
'/rjs/cxz/论文探索/投机采样/experiments/cost_matched/campaign.py' \
--prepare-only --job-dir /tmp/spec-cost-prepared-new
```

这不会启动服务、GPU 或真实 episode，不能作为执行许可。真实成本原始账本取得后，另用 `analyze.py --job-dir … --phase-labels … --output …` 离线汇总；没有阶段标签不会自动生成匹配。

## 下一关（尚未执行）

先另确认 GPU 数值/完整轨迹不扰动验证的预算：公共 seed/首局/init/native/fingerprint 合同、target 与 A_off 配置和资源/EGL 门冻结，先检查 profile+包装是否改变各自完整输出与轨迹。不能用本次 CPU PASS 代替该关。

关卡通过后，才按另批准的最多 8 完整 episode 做上述诊断、阶段标注和工作量对账；不通过就停止定位，不自动追加。自然调度、profile-off、无屏障的 E2E 是后续独立关卡。训练/RL/KL/大小模型联训均未开始。
