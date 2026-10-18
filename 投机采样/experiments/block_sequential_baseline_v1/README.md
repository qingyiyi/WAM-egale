# 完整 block 顺序基线（CPU阶段）

科研base固定为 `ca5f37e0482c2e38e3738e3d8e71e3b185990a23`，通过Git对象读取原始类。当前main只是公开镜像，不能替代固定source判断。原tile、生产模型文件和已有运行队列不修改。

`SequentialBlock` 是 **B0_HYBRID_SEQUENTIAL**：显式self-attention、cross-attention、FFN、norm、时间调制、残差与缓存顺序；参数复用原block，GEMM/norm/激活仍为Torch，attention后端可单独比较。它不是完整自有CUDA block，尚无device GEMM或persistent scheduler。

入口分工：

| 文件 | 作用 |
|---|---|
| `reference.py` | 从固定Git对象编译未经改写的原类AST，使用真实依赖，只加载原checkpoint一个block到CPU |
| `block_baseline.py` | 显式顺序B0；R1只复用已有aligned SDPA，导入native前单独要求GPU授权 |
| `fixtures.py` | 解码已有六个真实action输入，恢复独立可变cache，记录保存后的布局；不采集数据 |
| `validate_block.py` | CPU R0/R0自复现、完整B0自传输出、合法cache控制、输入/RNG/权重引用检查 |
| `debug_taps.py` / `inspect_block.py` | 只观察原模块边界、首个可观察分叉与Linear几何；不把gold喂回B0 |
| `engines.py` | 独立静态上下文投影诊断、既有guarded candidate内核的fixture适配，不改生产安装器 |
| `benchmark_block.py` | 有界A/A、ABBA/BAAB计时器和待运行矩阵；无GPU授权默认只输出PENDING表 |
| `runtime.py` / `cpu_job.py` | CPU两核/CUDA拒绝、真实PID出生身份、超时与退出收据 |
| `gpu_matrix.py` / `gpu_job.py` | 独立GPU矩阵与专用worker上限/身份清理；已准备但本轮未执行 |
| `audit_sources.py` / `audit_results.py` | 固定源码/旧CSV来源审计和描述性噪声复算，不重签旧证书 |

已有compact保存格式调用过`.cpu().contiguous()`，因此原生产CUDA stride、storage offset、alias与原CUDA RNG无法从它恢复。当前证明范围是保存后的逻辑CPU契约；不能把这些CPU结果改写成原GPU输入/输出逐位证明。原GPU gold与CPU reference的差异独立保留。

受限CPU运行示例（在项目根目录；输入/权重原件不进入GitHub）：

```bash
export PYTHONPATH="$PWD/投机采样"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export WAM_FROZEN_REPOSITORY="$PWD/review_exports/WAM-egale-libero-fullscale-20261017"
/rjs/cxz/miniconda3/envs/va/bin/python -B -m experiments.block_sequential_baseline_v1.test_contract

/rjs/cxz/miniconda3/envs/va/bin/python -B -m experiments.block_sequential_baseline_v1.validate_block \
  --repository "$WAM_FROZEN_REPOSITORY" \
  --checkpoint "$PWD/models/lingbot-va-posttrain-libero-long-modelscope-20260922" \
  --fixtures "$PWD/投机采样/results/runtime/p0-operator-torch-alignment-v2-20261018-r3/process1" \
  --output /path/to/new_registered_cpu_attempt --legal-write-controls
```

正式实际执行先写新attempt登记，再通过`cpu_job --attempt ... -- command`保存上限、日志、PID出生与退出收据；以上示例只描述CPU入口，不能重写历史attempt。CPU affinity默认4、5，最多两核。

只准备GPU矩阵，不执行CUDA：

```bash
/rjs/cxz/miniconda3/envs/va/bin/python -B -m experiments.block_sequential_baseline_v1.benchmark_block \
  --output /path/to/new_registered_prepare_attempt
```

GPU数值/性能本轮 **NOT_RUN_NO_GPU_AUTHORIZATION**。`--execute`在缺有效新授权时拒绝。完整GPU矩阵driver已绑定，但只有CPU控制/负例证据，没有GPU执行证书；需新审批文件、独占窗口、原件与已有native二进制，不会自动编译/安装或租卡。原block的动态nonzero/tensor-to-Python缓存分支不能直接当作合法整block Graph；旧SDPA Graph-core时间另外列。

后续获批后，使用新的独立attempt，启动`gpu_job --authorization NEW_AUTH.json --attempt NEW_DIR --repository ... --checkpoint ... --fixtures ...`。必须在新GPU进程中保留明确的物理GPU0映射（默认可见或`CUDA_VISIBLE_DEVICES=0`），不能继承CPU命令的空可见列表。该命令本轮NOT_RUN，不能以本轮CPU登记文件冒充授权。

授权文件须绑定本轮source patch hash与IMPLEMENTATION-MANIFEST、指定本机A100-SXM4的expected_gpu_uuid，六固定输入身份、accuracy/performance两个阶段、最多5000次block尝试、新存储≤1GiB、GPU生命周期≤7200秒及绝对清理截止、用户授权证据与独占窗口。worker每臂独立cache/模块、共享同一只读Parameter；先严格比较历史GPU gold/R0自复现/所有候选的输出、有效KV/metadata/输入/RNG，再A/A与ABBA/BAAB。检测外来GPU进程或空载计数异常会在CUDA前拒绝，记录资源阻塞；不杀他人、不自动重试。全部scope仍为规范化fixture block，不能写成机器人E2E。

本阶段验收和源证据见 [HW Experiment](../../../.pipeline/experiments/wam-block-baseline-20261019/PLAN.md)。四项机器人质量/正式性能权限保持false；不启动新episode、训练或第二环境。上游许可证保留于`LICENSE.upstream.txt`，B0源码保留原作者声明。
