# 固定DEV96：原作业与授权补齐的独立汇总

本入口只写新报告目录，不改原`tf_effect_quality_report_v1/report.py`、active源码、manifest、原FAILED/PARTIAL或retention。不会启动GPU、模型前向、采样、训练、模拟器、重试或独立TEST。固定科学基线为`b0dfac3a02f65de33a1217246b83003615166640`。

输入为`tf_effect_campaign_continue_v2`的pending-only新job。`p.cases`仍为原96，`continuation_schema=tf-effect-pending-only-continuation/v2`，通过`predecessor_job`、`imported_terminal_manifest`、`pending_keys`、`cumulative_budget`绑定原job和独立授权。新M局终态必须为`COMPLETE_PENDING_EPISODES_ONLY`；原预算FAILED/PARTIAL保留。原完整机器人失败无论关闭审计是否齐全都不能重跑替换。

原/续接真实目录逐case调用原`report.episode(source_job,case,source_READY,source_plan)`，复用原训练、RNG指纹、统计及配对统计助手；跨job初始配对明确读两个来源目录，不搭虚拟单job或复制证据。严格核验冻结模型/源码、相同生产API、BF16动作/FP32 draft、bank、target权重stat、实际初始RNG、物理/controller、相机、命令映射、全量原action/latent、O3/route/phase计数和各自真实CAL/清理收据。

结果分为“机器人完整终态描述”与“必要审计齐全”两层。完整失败但缺COUNTER/READY/RNG等者仍保留成功/失败及实际步数，标`TERMINAL_QUALITY_WITH_AUDIT_GAP`。冻结客户端先写完整`COMMAND-MAPPING.json`再追加`episodes.jsonl`；若预算恰在中间终止，完整命令映射或FINISH收据中的原成败也须保留，不能列入pending替换，缺少耗时/attempt时标UNKNOWN。只有精确原96键、68对及全部审计齐全，才给新DEV数值门/strict判定；否则UNKNOWN/PARTIAL，不能拼通过片段。fresh20与known-risk8分别统计，四个原权限始终false。

中断attempt的RAM forward可能没有落盘。完整局的精确forward与原保守下界/4000上界扣账分报；所有attempt实际总数UNKNOWN，不能把扣账上界当实测。两个job的setup/CAL/中断/cleanup墙钟分别保留，不作正式速度结论；`ΣT_round/Σg`及GPU counters未测即UNKNOWN。

prepare+run累计最多900秒、CPU≤2核、0GPU、报告存储≤128MiB，并取2026-10-17北京时间09:55更早；等待也占预算，不自动重跑。须等两个namespace真实终态及cleanup收据已落盘后登记，再立即执行：

```sh
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/tf_effect_completion_report_v2/report.py --prepare --job /absolute/completion-job --out /absolute/new-report-runtime
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/tf_effect_completion_report_v2/report.py --run --out /absolute/new-report-runtime
```

输出`RESULT.json`及`FACTS.md`、`QUALITY-RESULT.md`、`SYSTEM-RESULT.md`、`CLAIM-EVIDENCE-MATRIX.md`、`STOP-REPORT.md`，另有独立注册/CPU终态或失败。源码创建与语法检查不代表已登记或执行汇总。
