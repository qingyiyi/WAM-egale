# 单worker C/HC：生产导入路径 CPU 修复修订

这是投机采样任务的独立私有修订，**不是 TILE**。原目录
`/rjs/cxz/论文探索/投机采样/experiments/single_worker_chc` 及旧 GPU job 的
源码、CPU gate、FAIL 和证据保持原 SHA，不覆盖已执行版本。

## 本次改动

- `adapter.py::SingleWorkerCHC.admit()` 只认可 `modules.model` 和
  `wan_va.modules.model` 两种合法路径，同时验证固定源码路径/SHA、模块导出
  class、原 bound forward、forward 可执行结构/defaults、原 SDPA 可执行结构/
  绑定、eval、原 cache 模式及全部层。**不是只放宽字符串**。
- `admit()` 不安装包装、不要求 KV 初始化、不运行 target/attention，不做
  CUDA 初始化。`install()` 在同一准入检查通过后仍只替换原有候选 history
  片段。QKV/RoPE/cross-attention/FFN/SDPA/batch14 不变。
- `worker.py::Runtime` 在替换计数入口和写 `ready` **之前**执行全层准入；
  `capture()` 在读取 fixture、reset、preload 和 infer/52 次原始前向之前再次
  检查。捕获仍使用原 attention，优化仍在捕获/恢复后安装。
- `production_fixture.py` 将小几何随机 CPU 模型保存为临时 checkpoint，使用
  **未 mock 的真实服务器 `modules.utils.load_transformer/from_pretrained`**
  及 `_configure_model` 加载。不是实际训练 target/GPU验收。
- `test_loader_cpu.py` 覆盖合法别名、生产 loader、FP32/BF16 全30层/29变化
  query 的 R/C/HC，及非法路径、源码、class/forward/backend、最后一层、训练/
  cache 模式和提前拒绝的负例。`test_cpu.py` 保留原81项回归。

## CPU 验证

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 \
  /rjs/cxz/miniconda3/envs/va/bin/python \
  /rjs/cxz/论文探索/投机采样/experiments/single_worker_chc_loader_fix/run_cpu.py
```

每次 CPU 尝试独立留档于
`/rjs/cxz/论文探索/.pipeline/research/speculative-sampling/evidence/single-worker-chc-loader-fix-cpu-20261012/CPU-NNN`；
失败不会覆盖，基线冻结所有前置源码和旧 job 证据。

**本修订没有 GPU 监督器、排队器或 create_job。CLI 非 `--cpu-probe` 路径和
Runtime 非 CPU_probe 路径均明确拒绝；旧 GPU 标志和旧批准 plan 不能启用。**
CPU通过之后必须另确认新预算/新job，并对新版本补新job的预检；不重启旧job。
没有性能/E2E/成功率/环境步数的新结论；不训练/RL/联训/mega。
