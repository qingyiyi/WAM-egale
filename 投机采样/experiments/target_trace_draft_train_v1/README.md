# Target-trace draft pretraining

This isolated experiment pretrains the original TF64 and capacity-matched MLP96
from the real target-only LIBERO TRAIN194 queries. It does not claim the data
are the new draft's recursive verifier queries. It changes no deployed policy,
original adapter, historical acceptance gate, tile source, or live AutoDL job.

The fixed protocol is FIT174 (init 0–17 excluding task0/1 init6–8), CAL20
(init18/19), 2,000 AdamW updates, batch512, LR1e-4, weight decay .01, and three
seeds 20260922/23/24 per architecture. Both models use identical task/episode
balanced draws and FIT-only normalization. The original two-head DATA objective
is min-head MSE + .1 mean-head MSE + .01 ReLU(.05-normalized head gap).

The final update is always evaluated. CAL tasks are equally weighted. Each
architecture's seed is the middle seed after sorting by descending CAL macro
fraction of single-query errors within the unchanged threshold, ascending macro
scaled error, then ascending seed. CAL local acceptance predicts neither robot
success nor recursive acceptance. Historical six gates are separately recorded
as diagnostics. No offline gate automatically deploys or blocks a full DEV run.

After the independent data audit reports PASS:

```bash
CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
taskset -c 12,13 /rjs/cxz/miniconda3/envs/va/bin/python \
  投机采样/experiments/target_trace_draft_train_v1/train.py \
  --data 投机采样/results/runtime/p0-data-numeric-audit-v1-20261015-r1/historical_train194_columnar.pt \
  --audit 投机采样/results/runtime/p0-data-numeric-audit-v1-20261015-r1/RESULT.json \
  --out 投机采样/results/runtime/p0-target-trace-draft-train-v1-20261015-r1
```

The closed-loop worker calls `actual_cal_gpu_parity(training_dir, architecture,
device=..., expected_channels=..., out_path=...)` from `gpu_actual_cal.py`
before READY, then explicitly injects the returned predictor into
`server._p0_action_draft_b2_predictor`. It compares the complete 20,237 real CAL
queries with the original model's `decode_batch` on the same GPU. CPU/GPU
differences and acceptance flips are diagnostics. It does not call the older
four synthetic ramp/zero helper retained in `adapter.py`.
It must set `action_per_frame` before preloading and provide both
`PREFIX_ACCEPT_ENABLED=1` and `PREFIX_ACCEPT_GPU_APPROVED=1`, as already fixed
after the old r7/r8 failures. The new full DEV is performed by a separate runner.
