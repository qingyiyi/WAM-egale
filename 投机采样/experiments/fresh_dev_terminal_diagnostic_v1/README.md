# Fresh30 完整终态独立诊断

2026-10-15实际终态：已在`results/runtime/p0-fresh-dev-terminal-diagnostic-v1-20261015-r2/`生成完整独立诊断。原作业30完整、物理/RNG全配对，只有task8初始腕相机不同；原FAILED未改。首次独立诊断r1因本工具reference/candidate哈希标签写反被一致性检查拒绝，修正标签后新r2通过；原拒绝保留。再次复算必须使用不存在的新输出目录，不能覆盖上述证据。

本工具只在原fresh30 r2全部30局完成、所有arm/client/final清理证明完整、原launcher及作业进程不再活跃后执行。准备阶段不读取任何局的结果，不连接SSH或GPU。原30作业、原160份冻结来源、新AutoDL v2均不改。

允许的输入只有两种：原严格RESULT/REPORT PASS；或唯一FAILED明确来自完整采集后的原`exact_pairing`、且独立证明差异仅为初始相机像素。运行/预算/STOP/其他失败、缺文件、未清理、物理状态或RNG不等均拒绝产生完整统计报告。原FAILED和全部资格门保持原样。

逐局复用原只读episode解析与所有saved输出覆盖/finite验收，独立核对记录cost下的τ/最长prefix/bonus计数。物理配对从排除`initial_obs_hashes`的原meta、原始sim/controller初态数组、完整RESET、READY和实际target CPU/CUDA RNG tensor独立重算。原certificate中名称含`sim_controller_python_numpy_rng`的bool实际上包含obs hash，本工具只核验其原语义，绝不用它代替物理配对。

对T–MLP、T–TF、MLP–TF分别记录每task物理/RNG配对资格、再要求两相机逐位的资格。每camera计算不同元素数、不同像素数、max/mean绝对差和变化bbox，bbox采用`x0,y0,x1_exclusive,y1_exclusive`，无差异为null。容差始终为0，没有放宽门。

独立输出仅小JSON/Markdown：三臂实际SR/步骤P50/P90/max、失败固定800步罚分、target计数和每chunk计数、记录规则违规数、各严格配对子集和逐camera差异。完整采集的观察统计和精确复现分别标注；不能把原FAILED改叫质量PASS，也不能据此放行AutoDL/bench或更换候选。

root确认真实终态及进程清理后，才执行：

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B \
投机采样/experiments/fresh_dev_terminal_diagnostic_v1/audit.py \
--job 投机采样/results/runtime/p0-target-trace-draft-fresh-dev-v1-20261015-r2 \
--out 投机采样/results/runtime/p0-fresh-dev-terminal-diagnostic-v1-20261015-recompute \
--execute-terminal-audit
```

未通过终态/事实校验时不写原作业，不输出完整质量汇总。不能用本工具重试原作业或重建原RESULT。
