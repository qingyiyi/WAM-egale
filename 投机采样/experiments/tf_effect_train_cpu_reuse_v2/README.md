# CPU核验基础设施续接，未执行

只新增独立入口，不修改v1源码/旧FAILED/STOP/1800秒硬帽、科研门、seed、数据、normalization、loss或生产API。旧作业必须已FAILED且旧PID退出；必须已有完整500更新TF_CONT checkpoint，ON/MLP尚未开始任何更新。`prepare`不允许对仍在运行的旧job执行。

新增1800秒/256MiB，同时受显式master绝对deadline约束；旧实际耗时累计保留，不称旧实验重跑通过。TF_CONT在本轮optimizer更新0，逐tensor核对已完成checkpoint与既有CONT3000全部model/norm，然后恢复至新root的固定v1 checkpoint格式。TF_ON/MLP_ON各首次500更新，训练保持CPU2，原训练AST完整保留。只有CPU逐query `predict_all/reference_cpu` 使用CPU1上下文，离开即恢复CPU2；batch1、FP32/BF16边界不变，**不继承CPU2逐位证据**。

原v1 `prepare/run`仅在内存AST增添前驱/预算/provenance、CONT复用分支和结果附注；逆变换必须逐AST等于原函数，`AST-INVERSE-REVIEW.json`保存证明。原schema、adapter、GPU CAL入口兼容；旧失败目录loader仍然拒绝旧FAILED/STOP，本入口不移除该限制。新root重新签自己的完整CPU1接口结果；生产仍需完整旧20237+新6805同GPU原decode BF16逐位验收。原CAL before采用CPU2历史父预测，after采用CPU1，新报告必须明确这一诊断口径，不从两者差异宣称纯训练收益。

命令（由主任务登记/监督后执行；此处没有启动）：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B \
  投机采样/experiments/tf_effect_train_cpu_reuse_v2/train.py --prepare --job NEW_RECOVERY_JOB \
  --prior-job 投机采样/results/runtime/p0-tf-effect-onpolicy-train-v1-20261016-r1 \
  --onpolicy 投机采样/results/runtime/p0-tf-effect-collect30-pack-cpu-v1-20261016-r1/columns.pt \
  --data-audit 投机采样/results/runtime/p0-tf-effect-collect30-pack-cpu-v1-20261016-r1/DATA-AUDIT.json \
  --trigger 投机采样/results/runtime/p0-tf-effect-closure-v1-20261016-r1/TRAINING-TRIGGER.json \
  --include-mlp-on --master-deadline-epoch MASTER_ABSOLUTE_EPOCH \
  --wall-cap-seconds 1800 --storage-cap-mib 256
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B \
  投机采样/experiments/tf_effect_train_cpu_reuse_v2/train.py --run --job NEW_RECOVERY_JOB
```

外层timeout/CPU监督仍必需。所有prepare/等待/run共用同一绝对deadline；失败不再自动续接。新训练目录仍由 `tf_effect_train_v1.adapter.load_selected` 与 `tf_effect_train_v1.gpu_actual_cal`读取，完整质量DEV最多7小时和用户09:00截止由上层协议约束，不能被此CPU续接扩大。
