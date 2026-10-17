# 已记录风险对：独立 CPU 组件审计

只读取 `p0-tf-effect-dev96-v1-20261017-r1` 原 namespace 的四个完整 recorded episode：task1/init29、task3/init26，各 T 和 TF_ON。原任务结果及失败全部保留。它不恢复原缺失收据、不生成 quality 证书、不改变原 gate，不执行任何 forward、模拟器、训练、重跑或 GPU 作业。

逐局复用未修改的 `tf_effect_quality_report_v1.report.episode`：核验完整环境命令、物理动作传输、原 CPU postprocessor、全部保存 action/latent、终态、reset、counter、teacher 关闭及 scheduler 接受路径。另核 BF16 原动作形状、真实 CPU/CUDA RNG state 的 shape/dtype/bytes 与本 arm READY 绑定，逐 pair 比较 sim/controller 数组、完整 reset、实际 RNG、两相机像素。

原 root/clock 的 PID＋start_ticks 生命周期、全局 owned 清理和实存 `/proc` 状态独立检查。TF_ON 缺 `arms/TF_ON/OWNED-CLEANUP.json` 会单独报告；全局退出证明不能取代原局部收据。即使组件诊断完整，旧 strict quality gate 仍 BLOCKED，四项原权限始终 false。不推断失败根因。

只提供显式登记和执行，创建新的 runtime 输出目录，输入文件逐项绑定 SHA/大小/mtime；不存在的原局部收据也绑定其缺失状态。原 source、plan、四局、FAILED/PARTIAL 一律只读。prepare、等待、run 合计最多600秒，并受北京时间2026-10-17 08:50更早截止约束；CPU1、无子进程/CUDA，新增输出最多64MiB。失败/timeout 在新目录保留终态，不自动重试。

父 agent 登记后按顺序执行（下面只是 CLI，尚未执行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 12 /rjs/cxz/miniconda3/envs/va/bin/python -B \
  /rjs/cxz/论文探索/投机采样/experiments/tf_effect_recorded_risk_audit_v1/audit.py \
  --prepare \
  --job /rjs/cxz/论文探索/投机采样/results/runtime/p0-tf-effect-dev96-v1-20261017-r1 \
  --out /rjs/cxz/论文探索/投机采样/results/runtime/p0-tf-effect-recorded-risk-audit-v1-20261017-r1

CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
taskset -c 12 /rjs/cxz/miniconda3/envs/va/bin/python -B \
  /rjs/cxz/论文探索/投机采样/experiments/tf_effect_recorded_risk_audit_v1/audit.py \
  --run \
  --out /rjs/cxz/论文探索/投机采样/results/runtime/p0-tf-effect-recorded-risk-audit-v1-20261017-r1
```

完整终态为 `COMPLETE_RECORDED_COMPONENT_DIAGNOSTIC`；未完成组件核验为 `PARTIAL_RECORDED_COMPONENT_DIAGNOSTIC` 或 `FAILED_RECORDED_COMPONENT_DIAGNOSTIC`。产物为 RESULT.json、RECORDED-COMPONENT-DIAGNOSTIC.md、CPU-FINAL.json；失败另外写 FAILED.json。COMPLETE 仅表示本限定 recorded 组件被检查，绝不是 quality PASS、独立 TEST 或非劣效性结论。
