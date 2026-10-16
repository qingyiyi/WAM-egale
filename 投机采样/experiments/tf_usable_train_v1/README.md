# 固定主seed23 TF继续训练与显式候选loader

仅新隔离源码，不改变当前部署、旧训练、旧FAILED或算子。默认导入不会训练、采集或初始化CUDA。当前新FIT20/CAL10仍HOLD。

`train.py --prepare --job NEW_JOB` 在root审查后生成独立PLAN；`--train --job NEW_JOB` 才实际CPU训练。固定当前TF child2500 seed20260923（9cb92a...ab07）起点，原stage2配方继续500更新：历史target FIT174/159691行256 + 原same-query14997行256，AdamW重置moments、lr3e-5、wd.01、原双头loss、FP32训练、原FIT归一化逐位保持。重复原seed23/700000/800000的完整500×256采样draw，并与旧draws.pt逐位核对。末checkpoint固定，无早停、CAL挑seed或新阈值。900秒CPU墙钟帽，2核，128MiB产物帽；导出的推理仍用原BF16输入/输出接口。

新schema `tf-usable-draft-continuation/v1`：2500+500=3000，condition=TF_CONT；不篡改原two-stage2500schema。`adapter.load_checkpoint` 验当前child23的原字节、逐位normalizer和原stage1 lineage。主训练终态 `PASS_PRIMARY_TF_CONT_500_COMPLETE`，`complete_runs=1`；完整CAL20237对保存后loader、训练predictor与同CPU原models.Predictor.decode_batch(batch1/BF16)全部逐位，保存完整cal_predictions.pt。训练成功只授予完整闭环DEV候选资格，不是成功率PASS。

统一质量接口：

```
from tf_usable_train_v1.adapter import load_candidate
candidate=load_candidate(checkpoint, architecture, device, expected_channels, expected_sha256=None)
```

显式child2500 TF/MLP均强制实际seed20260923，调用冻结two-stage loader；当前MLP使用原seed23，不能把selected seed22称为同seed。cont3000仅TF。`load_selected(training_dir,architecture,*,device,expected_channels)` 只收完整无FAILED/STOP的主TF_CONT结果，不制作假selected。

GPU完整真实CAL接口（仅root明确运行）：

```
actual_checkpoint_cal_gpu_parity(checkpoint, architecture, *, data_path, device,
    expected_channels=tuple(range(7)), expected_sha256=None, out_path=None)
```

它复用旧helper完整20237原CAL20、同device原decode_batch(batch1)数学，严格逐位/finite/CPU-CUDA RNG门；跨CPU-GPU保存预测差只诊断。`actual_cal_gpu_parity(training_dir,...)` 是已完成TF_CONT目录包装。没有ramp/zero/smoke，也无target前向。

本机仅准备示例（需root审后执行）：

```
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/tf_usable_train_v1/train.py --prepare --job 投机采样/results/runtime/p0-tf-usable-train-v1-20261016-r1
```

远端不得改.pt内部parent绝对路径。需按真实已存在`/rjs/cxz/论文探索` alias建立parent child23 checkpoint/RESULT与stage1 ancestor映射，已有不同字节拒绝；新质量runner另行显式冻结路径映射。collector设计和未来TF-on/MLP-on约束见COLLECTOR-AND-NEXT-DATA.md，目前未实施新数据/训练。
