# 旧视频派生缓存清理与重建

2026-10-15按用户已授权的“确认无用后直接删除”执行。当前主线是action投机，旧phase4 video离线研究不再使用这些派生缓存。只删除8个明确文件，原始数据和历史结果保留。

完整CPU溯源：1633份原始trace、31027行、4字段、FP16/FP32两精度，共6100156416个标量逐值一致，CPU单核277.71秒，0GPU/0环境/0新target，不进行独立SHA作业。来源是`投机采样/traces/p0-libero-targetonly-alltasks-20260924-formal1/`；当前动作训练和新DEV不读被删缓存。

已删两目录`dataset/video/`中的`state.mmap`、`previous.mmap`、`target.mmap`、`target_next.mmap`：

- `投机采样/results/draft_full_phase4-20261007/`
- `投机采样/results/draft_full_phase4-20261007-f32/`

共18300469248字节（17.04GiB）。保留1633原trace、全部action缓存、视频streams/branch metadata/normalization/sigma/splits/context、checkpoint和结果。清单、原inode/mtime/大小、全部源文件inventory及数值证明在`投机采样/results/runtime/p0-cleanup-derived-video-v1-20261015-r1/`。

如要复做旧video离线实验，先显式恢复（不会自动运行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/cleanup_derived_video_v1/restore.py
```

需要至少17.04GiB加5GiB余量，逐原trace重建原精度和行序。历史inode/mtime不伪造，依赖原stat的旧冻结检查会要求显式迁移说明；本次没有重跑旧视频训练或声称其历史结论重新验证。原始`p0_phase4_full_draft_compare.py`仍保留，当前版本使用FP32，重建FP16应使用此专用恢复程序。

本轮另外已清理5.834GiB旧不完整模型下载shard，见独立job `p0-cleanup-obsolete-downloads-20261015-r1`；两批累计22.88GiB。全部当前模型、计时fixture、teacher query、失败证据、tile代码/HW均不属于本批。
