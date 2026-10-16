# PF2 v4：生产构造入口兼容修复

PF3 于实际 fixed_worker→VA_Server.__init__ 入口失败：bonus 构造器只允许 `greedy/global`，supervisor 提前传入 `prefix_then_cost`，未进入模型重载或任何数值 chunk。失败16.584884544834495秒、0完整episode/0数值chunk、未闭合固定960预留全额保守计费；冻结 PF3 与失败证据不改。

最小修复有两处：supervisor.environment 只对构造环境设置合法global；实际运行时 selector 仍为原 SELECTORS，episode server 在构造后照旧安装prefix。fixed.main 显式 global 构造，进入回调后恢复环境并恢复prefix，再运行原数值run_trials；构造或回调失败均保存WORKER-FAILED，finally恢复环境。server_entry文件不改。

fixed_worker整文件SHA改变；`run_trials`、`normalize_routes`、`combine_counters`的源函数体和AST SHA与PF3一致。其它18项核心文件逐字节一致。复用原通过的12个完整CPU数值chunk，数值chunk重跑0。新的CPU入口合同实际调用未改的生产VA_Server.__init__：先复现prefix拒绝，再由实际fixed.main跑到第一个load_vae重载边界的CPU sentinel；无模型权重重载或CUDA。另检查实际main回调prefix、原shutdown/sync顺序和supervisor调用fixed/server实际脚本及三臂构造环境。

依赖分开保留：已完整关闭的resume job `/var/tmp/cxz-wam-quality-20261014/p0-trace-quality-resume-v1-20261014-100500` 提供16 qualified sources；随后失败的 `/var/tmp/cxz-wam-quality-20261014/p0-periodic-refresh-quality-v3-20261014-111500` 提供最新实际预算carry。两者和旧保守960均逐源检查；最新carry=23208.129861099645秒/95527 intents，新有限aggregate forward cap=144487，累计GPU仍硬限36000秒，阶段3600秒/48960 intents。旧plan/counter/source/model/阈值/质量门不修改。

原三臂T/OLD/PF2、task0/1×init13/14、12完整episode、两EGL、固定12GPUchunk/cap960、每episode cap4000及原pilot质量门不改。共享GPU的foreign/任意利用率授权保持，只清理自有身份，容量31.25GiB（双模型+双EGL已观察峰值加余量）、运行512MiB。输出在显式任务专用/var/tmp父目录，磁盘初始13GiB/运行12GiB/输出≤1GiB不变；performance_deferred=true，无新速度结论。

CPU命令：`CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/periodic_refresh_quality_v4/entry_cpu_check.py --out .pipeline/research/speculative-sampling/evidence/thread-continuation-20261014/periodic-v4-entry-fix-cpu/CPU-ENTRY-CONTRACT.json`。

GPU由主代理在CPU合同及prepare/preflight通过后启动；CPU入口合同不代表新数值GPU通过或PF2任务质量结果。
