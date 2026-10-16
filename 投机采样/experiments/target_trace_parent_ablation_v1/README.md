# TF23 同seed真实父子完整DEV消融

只新增原TF seed20260923、2000步parent的Long10/init24完整10局。对照是r2真实T10与同seed child2500的10局，绝不使用旧40的TF22。parent从实际child23 metadata.parent_checkpoint及SHA绑定；校验原2000/2500 schema、原RESULT、spec及逐位相同norm。调用冻结stage1单checkpoint loader，角色明确为explicit parent ablation，既不读stage1 selected，也不写新selected文件。

root必须先等r2完整30终态及所有进程清理，再运行独立 `fresh_dev_terminal_diagnostic_v1`。它可承认原严格PASS，或仅camera像素失败的原FAILED；本计划需要其真实完整DIAGNOSTIC.json，不能制作原r2 PASS。原失败和全部task失败保持原样。此消融不自动决定候选或放行AutoDL/bench。

仅切换小模型loader，其余目标权重、target seed20260922、原bank/原800horizon、τ0.007839680526450805、K2/B2 bonus/prefix_then_cost与完整client/capture/coverage/cleanup复用原真实实现。每次parent启动完整运行20237真实CAL同GPU batch1/BF16接口，对同权重/norm的原models.Predictor.decode_batch逐位、finite与RNG不变；CPU保存预测来自同一个parent23目录，跨CPU/GPU差仅诊断。target初始CPU/CUDA RNG在loader/CAL前捕获，并与r2 T和child实际保存RNG逐位对齐，setup及每次reset恢复原状态。

一个persistent target、一个EGL client、CPU2；22GiB初始headroom，50分钟总预算，300秒清理预留，单局4000/总25000 target上限，2GiB输出和5GiB磁盘floor。一次attempt，无自动重跑；不训练、不额外采集teacher、不smoke、不调阈值或seed。共享GPU时间不作正式性能。

终态审计新增parent10加引用T/child20的全部原saved输出、计数和接受规则。物理初态/环境RNG、完整RESET及target实际RNG必须严格匹配；两相机原npz逐camera报告像素/元素数、幅度及bbox，并分别列像素逐位子集。像素不等的局不能宣称所有输入逐位相同的强因果。这是DEV同seed训练阶段消融，不是独立TEST。

root确认r2诊断和独立审阅通过后，CPU准备新作业：

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B \
投机采样/experiments/target_trace_parent_ablation_v1/run.py \
--job 投机采样/results/runtime/p0-target-trace-parent-ablation-v1-20261015-r1 \
--reference-diagnostic 投机采样/results/runtime/p0-fresh-dev-terminal-diagnostic-v1-20261015-r2/DIAGNOSTIC.json --prepare
```

唯一GPU启动由root显式执行：

```bash
env OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
/rjs/cxz/miniconda3/envs/va/bin/python -B \
投机采样/experiments/target_trace_parent_ablation_v1/run.py \
--job 投机采样/results/runtime/p0-target-trace-parent-ablation-v1-20261015-r1 --execute-reviewed
```

不接受未完成r2、budget/runtime/缺文件失败或物理/RNG配对失效。准备/导入不会启动GPU。全部10结束、引用30分析/报告及自有清理完成后，才写本新作业RESULT；原r2不写任何文件。
