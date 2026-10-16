# MLP child23 的等更新预算续训

隔离复制晨间 TF_CONT 入口，仅适配 `mlp96`、同 seed20260923 的 MLP child2500、独立 schema/status。原 TF、stage1/2、晨间162源与旧 checkpoint 均不修改。不增加训练功能，不采集新 teacher，不读新 DEV/TEST。

固定500 updates、batch512=历史target FIT174/159691行256 + 原same-query14997行256。AdamW重新初始化，lr3e-5、wd.01，原双头loss、FP32训练及BF16查询/输出不变。继承原FIT normalizer，500×256的两组draw按原seed生成，并与旧stage2真实draw逐位检查；该draw已与实际TF_CONT训练draw完全相同。末更新固定，无CAL选checkpoint/seed。

MLP52928参数；起点是 child23 `655e8647…9165e`，不是selected MLP22。新checkpoint schema为 `tf-usable-mlp-draft-continuation/v1`，condition=`MLP_CONT`，2500+固定500（优化器重置），不冒用TF或旧child schema。新selected schema为 `tf-usable-mlp-selection/v1`。严格loader复核parent/stage1谱系、模型字段、逐位normalizer；失败/STOP拒绝。

CPU2、900秒、128MiB；全20237旧CAL对新loader/训练predictor及同CPU原 `models.Predictor.decode_batch(batch1)` 的BF16路径逐位核验。训练完成不意味着机器人质量或正式性能通过。

准备/训练分别显式执行：

```bash
env CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python -B 投机采样/experiments/tf_usable_mlp_continue_v1/train.py --train --job 投机采样/results/runtime/p0-tf-usable-mlp-continue-v1-20261016-r1
```

上述目录已经CPU prepare；实际训练仅root审定后执行一次。最终新模型供新namespace显式opt-in，旧默认和旧质量门不变。

质量server的独立接线接口：

```python
from tf_usable_mlp_continue_v1.adapter import load_candidate, load_selected
from tf_usable_mlp_continue_v1.gpu_actual_cal import actual_checkpoint_cal_gpu_parity
```

GPU helper 与晨间完整20237真实query的同GPU核验数学相同，仅换新loader；只能稍后由完整任务server setup调用，不单开GPU试跑。原device RNG保持和跨CPU/GPU差异仅诊断的边界不变。
