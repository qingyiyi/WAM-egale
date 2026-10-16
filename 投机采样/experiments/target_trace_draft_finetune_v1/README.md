# Fixed two-stage draft follow-up

The existing pretraining/adapter/real-CAL source and completed original DEV
remain unchanged. This is a finite development follow-up after observing the
original TF's task3/init24 failure. It is not a new blind test.

`results/runtime/p0-target-trace-draft-finetune-v1-20261015-r1/PLAN.json` was
written before training. Each original architecture/seed checkpoint initializes
its matching run. Stage2 is 500 AdamW updates, LR3e-5, wd.01, batch512 split
256 target-only FIT174 and256 historical same-query FIT. Normalization is the
exact corresponding stage1 normalization. The original two-head DATA loss and
architecture remain unchanged. Both architectures use identical per-seed draws.

The14997 same-query labels are current3906 from A_off and aux11091 from the
historical research TF64 `B_MIXED_HISTORICAL_C` seed22. They are actual-query
teacher labels, but are not newly pretrained TF on-policy collection. No new
teacher execution or simulator is started. CAL20 comprises all20237 recorded
queries from init18/19. Quality init25–29 tensors are never opened.

Run on CPU12/13:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/target_trace_draft_finetune_v1/train.py \
  --job 投机采样/results/runtime/p0-target-trace-draft-finetune-v1-20261015-r1
```

New schema: `target-trace-draft-two-stage/v1`. Metadata identifies the exact
parent checkpoint and runtime SHA, 2000+500=2500 updates, 50/50 mixture,
normalization retained, and no deployment. This cannot be loaded through the
old target-only schema. New opt-in `adapter.load_selected` uses the same
production prediction signature, after six complete runs and fixed CAL median
seed selection. It validates parent lineage, tensor inventory and normalization.

For the explicit full closed-loop worker:

```python
from target_trace_draft_finetune_v1.gpu_actual_cal import actual_cal_gpu_parity
predictor, report = actual_cal_gpu_parity(
    training_dir, "transformer64", device=server.device,
    expected_channels=tuple(range(7)), out_path=output / "FULL-CAL-PARITY.json")
```

That new helper reuses the original full-real-CAL same-device mathematical
verification with only the loader reference resolved to this new adapter. It
does not run on import and has not been executed on GPU during CPU training.
Full FIT improvements and CAL local errors do not establish success rate,
trajectory length, recursive acceptance or speed.

The new loader additionally rejects a failed top-level job and incomplete or
failed seed cells. Its parent-checkpoint lineage currently uses the original
absolute local path. A later AutoDL deployment must either mirror those small
parent checkpoint/result paths or add an explicit checked relocation mapping;
rewriting only PLAN/selected paths is insufficient. This does not affect the
present local full DEV.
