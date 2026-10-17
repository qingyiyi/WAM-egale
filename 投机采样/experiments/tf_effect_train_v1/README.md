# TF64 唯一一次数据配方对照入口

本目录只新增独立源码；截至创建时未登记/执行采集、训练、模型前向或 GPU 作业。固定科学基线为 `b0dfac3a02f65de33a1217246b83003615166640`。TF64 为主模型；此入口不触碰旧 loader、默认 A_off、阈值、旧 FAILED/retention 或四项 false 权限。

## 固定对照

| 条件 | 初始化 | 每次更新 | 总更新 | 作用 |
|---|---|---|---|---|
| TF_CONT | TF child2500，seed20260923 | 历史 target FIT256 + 旧14997池256 | 500至3000 | 同预算历史数据对照 |
| TF_ON | **同一 TF child2500** | 历史 target FIT256 + 新FIT到达节点256 | 500至3000 | 唯一主候选 |
| MLP_ON（需显式登记） | 同seed MLP child2500 | 同 target/new FIT 抽样 | 500至3000 | 后置等容量架构对照 |

新数据生成器是已存在的 **TF_CONT3000**，训练初始化是 **child2500**，两者分别绑定真实路径/小模型 SHA；它们不是同一个 checkpoint。MLP_ON 使用 TF 诱导分布，不能称 MLP 自身 on-policy 数据。

`train.frozen_functions()` 直接调用 `tf_usable_train_v1.train.math_functions()`，没有复制或改写 draw/decode/loss/metrics。AdamW 重置，lr3e-5、wd.01，固定500末步、旧normalization，主seed23；不按 CAL 选模型或 seed。TF_CONT 额外检查全部权重 tensor 与已完成 CONT3000 逐位相同；checkpoint 元数据不同，不能要求两个新旧文件 SHA 相同。

## 输入接口

`columns.pt` 顶层：

```python
{
  'schema': 'tf-effect-onpolicy-columns/v1',
  'FIT': {'state': FloatTensor[N,112], 'previous': FloatTensor[N,112],
          'target': FloatTensor[N,112], 'scalars': FloatTensor[N,3],
          'delta': FloatTensor[N], 'task': LongTensor[N],
          'episode': LongTensor[N], 'node': LongTensor[N]},
  'CAL': {...},
  'manifest': {...}
}
```

浮点列均 CPU FP32；scalars 为 `[sigma,next_sigma,timestep/1000]`。node0是root，1/2是parent。列只收 root+**实际到达 parent**，不是选中即到达；depth1已到达parent，depth0未到达。反事实另存，不能混入主训练列。若另有 `state_reached` 列，必须逐行 bool true。FIT固定task0–9×init30/31，CAL固定×init32；必须30条完整终态，成功/失败都保留，不能拿DEV26补FIT或CAL。输入不读取TEST40–49。

manifest 必需字段：

```python
{
  'source_commit': 'b0dfac3a02f65de33a1217246b83003615166640',
  'generator_checkpoint': '/absolute/path/to/existing/TF_CONT3000/checkpoint.pt',
  'generator_checkpoint_sha256': '3657388fdce0763093d1b704c88a03b6836954160d43dddb9f75337a054d64ac',
  'generator_condition': 'TF_CONT3000', 'generator_total_updates': 3000,
  'fit_episode_ids': [t*100+i for t in range(10) for i in (30,31)],
  'cal_episode_ids': [t*100+32 for t in range(10)],
  'complete_episodes': 30, 'all_episodes_terminal': True,
  'extra_teacher_forwards': 0, 'actual_o3_preserved': True,
  'row_selection': 'root_and_actually_reached_parent',
  'counterfactual_nodes_in_columns': False,
  'production_mode': 'verifier_k2_b2_bonus',
  'threshold': 0.007839680526450805,
}
```

manifest可增加真实provenance。digest由 `adapter.digest_json(manifest)` 算出（sorted keys/compact JSON/UTF8），不对整套模型或大数据做独立hash。

另需独立采集审计 JSON：`schema=tf-effect-data-audit/v1`，`status=PASS_FULL_30_EPISODE_REACHED_NODE_AUDIT`，`source_commit`、`input_pack`绝对路径、`input_pack_stat=adapter.stat(path)`、`manifest_sha256`；`all_rows_reached/all_episodes_terminal/fit_cal_disjoint/no_development_or_test_rows/collector_equivalence_passed`须为true；`extra_teacher_forwards=0`。其中 collector_equivalence_passed 必须来自真实完整记录重放的 OFF/ON 等价证据，不能只靠数值列形状检查。证书还需 `evidence=[{'path':..., 'sha256':...}]`，每项小审计证据会核验并冻结；本入口不把布尔声明替代原始审查。

训练触发证书：`schema=tf-effect-training-trigger/v1`、`source_commit`、`training_trigger=true`、非空 `mechanism/observed_support/inference_limits`，`allowed_conditions`，`evidence=[{'path':..., 'sha256':...}]`。只有具体机制诊断支持一次新监督机会才可制作正触发；现有的来源缺口或单失败相关性不足以自行填true。

## 执行阶段

主任务先登记并复核输入、授权、GPU/CPU和总预算。以下命令是**用法**，不是已启动作业。单训练CPU进程固定2个已允许核，CUDA隐藏：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/tf_effect_train_v1/train.py --prepare --job NEW_TRAIN_JOB \
  --onpolicy COMPLETE_COLUMNS --data-audit COLLECTION_AUDIT --trigger MECHANISM_TRIGGER \
  --wall-cap-seconds 1800 --storage-cap-mib 256
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/tf_effect_train_v1/train.py --run --job NEW_TRAIN_JOB
```

prepare先登记，再核来源/结构/父模型；**没有模型前向/训练**。run使用prepare开始的同一绝对deadline，等待也计预算，不因重试重置。失败原目录保留；不可就地重试覆盖 STARTED。默认只TF_CONT/TF_ON，额外 `--include-mlp-on` 必须有主登记/触发证书明确涵盖。内层alarm之外仍需外层监督/timeout；本入口不创建worker或service。

训练输出：`PREREGISTRATION.json/PLAN.json/PREPARED.json/STARTED.json/draws.pt/DATA-SOURCE-PROOF.json`，各固定cell的 `checkpoint.pt/RESULT.json/cal_predictions.pt/new_cal_predictions.pt`，最终 `selected.json/FINAL-RESULT.json/status.json`；失败 `FAILED.json`。部分成功cell不自动取得整体完成资格。

GPU CAL需另外登记，不计入CPU训练预算。CPU准备不允许已有CUDA context；之后可在受监督的单GPU进程或已有owned服务器中调用：

```sh
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/tf_effect_train_v1/gpu_actual_cal.py --prepare \
  --job NEW_CAL_JOB --training-dir COMPLETE_TRAIN_JOB --condition TF_ON --wall-cap-seconds 900
PYTHONDONTWRITEBYTECODE=1 /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/tf_effect_train_v1/gpu_actual_cal.py --run --job NEW_CAL_JOB --device cuda:0
```

Python接口：`actual_cal_gpu_parity(prepared_cal_job, device='cuda:0') -> (ActionPredictor, report)`。返回predictor可接原productionAPI；API调用不接管caller的进程监督，caller必须保留外层时钟/cleanup。完整旧20237行+新CAL所有行，每条batch1，用原 `models.Predictor.decode_batch` 同device BF16输出逐位验收，CPU/CUDA RNG必须不变。CPU/GPU接受翻转单列描述，不用CPU结果替代实际GPU部署。GPU报告 `RESULT.json` 必须保留失败并按原预算记账。

GPU CAL的prepare、等待、加载和两套CAL共用prepare时建立的绝对deadline；请在资源已具备时才prepare，超时不能改deadline续跑。

在线加载接口：`adapter.load_selected(training_dir, 'TF_ON', device=..., expected_channels=range(7))` 或显式 `load_checkpoint(..., architecture='transformer64', expected_condition='TF_ON',...)`。新schema只接受上述真实父/500/normalizer/manifest；旧two-stage和旧CONT loader保持原样。完整GPU CAL PASS后仍需完整LIBERO任务；本目录从不签机器人SR、TEST、非劣或正式性能许可。
