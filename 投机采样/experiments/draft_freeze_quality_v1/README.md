# Frozen-draft complete DEV quality entry

`protocol.py` prospectively fixes the `today50` Long10 development set. Its
literal keys identify 50 full, unshortened LIBERO episodes: 22 fresh target
episodes (four repeatability episodes at task 2/8, init 33, plus ten init 35
and eight known-risk cases), 18 TF_CONT episodes (ten init 35 plus the same
eight risks), and ten MLP_CONT init-35 episodes. `run.py` executes them in
arm-major plan order, one persistent original WAM server and one EGL client at
a time. It does not infer task, init, or repetition from the key string.

The target checkpoint, original 512 BF16 text bank, full 800-step horizon,
K2/B2 bonus, O3, `prefix_then_cost`, and threshold remain bound by the plan.
The client reuses the actual legacy `reset -> set_init_state -> five zero
actions` path and its cached first observation. `capture_client` only copies
the two actual initial images and records the unchanged original full episode.
No synchronized redraw, reference-image injection, success-rule change, or
pixel tolerance is part of this entry. This is DEV quality evidence, never a
formal performance measurement or independent TEST.

Before any GPU episode, the CPU `prepare(job)` and `load(job)` bind the
preregistered 50 cases, frozen source/native files, trained TF_CONT and
MLP_CONT checkpoints and lineage, actual CAL data, target weight stats, bank,
and original full-reset equivalence. `T` uses the TF_CONT checkpoint only as
the unused legacy environment dummy while action drafting is disabled. Each
draft arm runs all 20,237 real CAL queries on its actual GPU before its first
episode, using its architecture-specific adapter against the original
`Predictor.decode_batch` interface. The target's CPU/CUDA RNG is captured
before candidate setup and restored before each complete episode.

The registered wall clock began 2026-10-16 12:20:29 Beijing. Work ends by
18:15:29 and cleanup by 18:20:29; the one local supervisor also has a
four-hour limit measured from its real `/proc` birth, including GPU-headroom
waiting, and reserves 300 seconds for cleanup. The earlier bound wins. The
plan caps target forwards at 100,000 total and 4,000 per episode, new output
at 3 GiB, and both server/client at two CPU threads. Shared GPU is allowed
for this algorithm-quality task; latency numbers are not formal speedups.

After independent source and plan review, the explicit CPU preparation is:

```bash
python -B 投机采样/experiments/draft_freeze_quality_v1/run.py \
  --job /rjs/cxz/论文探索/投机采样/results/runtime/p0-draft-freeze-quality-v1-local-r1 \
  --prepare
```

`--prepare` accepts no old site/checkpoint keyword arguments; those frozen
inputs are resolved and verified by the new protocol. It performs no rollout
or GPU CAL. The full supervisor argv is:

```bash
python -B 投机采样/experiments/draft_freeze_quality_v1/run.py \
  --job /rjs/cxz/论文探索/投机采样/results/runtime/p0-draft-freeze-quality-v1-local-r1
```

The root-owned detached launcher must start that exact argv in a new session,
pin the real PID/start ticks in `ROOT-DETACHED-LAUNCH.json`, and immediately
attach the separately reviewed `draft_freeze_clock_v1` CPU clock. The
supervisor writes `RUN-STARTED.json`, then waits up to 60 seconds for a live
`CLOCK-GUARD.json` whose supervisor and guard pins and 4-hour/300-second
deadlines match reality. **No model is launched before this barrier passes.**
An unguarded direct invocation fails closed. The child refuses any existing
failure, STOP, prior RUN, COLLECTION, or RESULT marker and takes the project
GPU locks before launching a model. There is no automatic retry.

Each full episode retains the original actions, latents, observations,
commands, initial/final sim/controller state, two initial camera pixel arrays,
target RNG tensors, acceptance/phase counters, and owned-cleanup receipts.
`COLLECTION.json` can certify only 50 complete episodes and cleanup. The
new analyzer separately evaluates paired quality and preserves strict camera
pixel evidence; the original failed records are never rewritten. Final
`RESULT.json` is written only after collection, report, failure scan, and
final owned-process cleanup. Partial or stopped work receives `FAILED.json`
with completed keys and leaves all raw evidence intact.
