# 原reset与最终T5缓存的完整任务等价验收

用户授权本机2→3→4的第2步。仅新隔离入口，原WAM/AutoDL运行job/preset/tile不改。不做smoke、不做独立SHA检查，直接8条完整LIBERO任务，两个task、T和A_off各原编码/缓存配对。固定800 horizon，正常成功才提前结束。

同一arm常驻一个原模型；每次reset完整清理原KV/streaming VAE，并恢复同一READY后的CPU/GPU RNG，保持配对初态。按原库真正构建EGL环境。原编码每条会与预计算bank在GPU逐位比对；重置后的空cache/规范化/文本/RNG、所有实际动作、初终模拟器/控制器/RNG、最终有效KV与VAE状态均需相同。忽略未写入torch.empty cache区及不影响计算的输出目录名。

仅CPU prepare。run明确GPU执行，等待至少22GiB空闲，允许共享且不报正式性能；一个模型/一个EGL是共享40G资源限制，进程身份与清理复用已验收launcher。CPU模型7线程、client2线程；4小时硬期限、单episode40分钟。失败保留不复用为完整成功结果。

```
python 投机采样/experiments/reset_online_v1/run.py --prepare --job <fresh-project-runtime-job>
python 投机采样/experiments/reset_online_v1/run.py --job <same-job>
```

RESULT需PASS_COMPLETE_RESET_BANK_EQUIVALENCE后，才用于下一独立质量job。当前AutoDL不做热改；新的正式性能仍等本机独占。
