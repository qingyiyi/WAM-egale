# 固定TF-on的96完整DEV审计

只读CPU报告入口。它不启动GPU、训练、LIBERO、采集、重试、120局或独立TEST，也不修改campaign、旧FAILED、retention、阈值和四权限。

固定来源commit：`b0dfac3a02f65de33a1217246b83003615166640`。输入必须为`tf_effect_campaign_v1`登记的`collect=false`96局：fresh task0–9 × init36/37，T/TF_CONT/TF_ON/MLP_ON各20；已知风险8组仅T/TF_ON各8。两组分开统计，TF_ON是主候选，MLP_ON仅对照。

在campaign、CPU训练和自身源码最终冻结后，使用CPU环境分别prepare、run；最多900秒从prepare开始累计，等待也计预算，并取整轮global12h与最新用户更早截止的最小值。10/17睡前用户要求09:00汇报，未来DEV已前瞻预留08:40 GPU清理、08:55 CPU分析清理；旧预算不改写。两核上限，输出128MiB上限；无子进程。所有路径使用新的runtime命名空间。

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/tf_effect_quality_report_v1/report.py --prepare --job /absolute/campaign --out /absolute/new-report-runtime
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/tf_effect_quality_report_v1/report.py --run --out /absolute/new-report-runtime
```

逐局核验原完整horizon800终态、实际命令映射、初始/最终sim-controller数组与相机、metadata/RNG、reset、ready、全量保存的action/latent、target/acceptance/合法scheduler提交计数、O3、teacher关闭、完整FINAL-SERVER-STATE及自有进程/独立clock清理。观测文件抑制按新入口审计，不能复用期待保留`obs_data_*.pt`的旧coverage helper。

同时核验固定500更新TF-cont复现旧control参数、TF-on同parent、MLP相同预算与新旧真实CAL设备接口证书。实际保存CPU/CUDA RNG张量的shape、dtype和原始byte fingerprint须绑定各自READY/reset，不能只比较各arm文件彼此相同。所有旧源、证据和小权重按既有manifest核验；target整套权重只核查stat，不独立hash。

输出`RESULT.json`、`FACTS.md`、`QUALITY-RESULT.md`、`SYSTEM-RESULT.md`、`CLAIM-EVIDENCE-MATRIX.md`、`STOP-REPORT.md`与CPU终态。未齐全或已有FAILED/STOP则明确PARTIAL/FAILED，不从通过片段选模型，不把中断计成算法失败。像素差异保留并与物理/RNG门分报。

新DEV数值门单列：TF-on成功数不少于T/匹配MLP、双方成功长度比P50≤1.10/P90≤1.20/max≤1.30、风险8无新增T成功TF失败。严格screen还要求全部预登记配对门；旧四权限始终false。DEV不能建立非劣、独立TEST保证或Transformer优于MLP。

完整episode耗时是共享资源下诊断，不能报正式速度。没有逐round时间，`ΣT_round/Σg`为`UNKNOWN`；target路径未直接统计合法更新时，实际g也标UNKNOWN，另列50/chunk固定契约数量。1/3/7引擎矩阵与GPU counters本报告不执行、未测项均为UNKNOWN。

这里只提供入口，尚未prepare或执行新96局审计；源码语法检查不代表实验已完成。
