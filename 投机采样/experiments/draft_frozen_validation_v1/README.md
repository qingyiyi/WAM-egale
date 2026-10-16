# Frozen-draft expanded DEV, four independent shards

This entry is prepared for a later, separately registered local-GPU window. It
cannot run now: no future clock wrapper, deadlines, or registration receipt are
assumed to exist. Today's `draft_freeze_quality_v1` experiment and its 173
bound inference sources stay untouched. No TEST40–49 outcome is read or run.

`DEV-MANIFEST.json` fixes Long10 tasks 0–9 at init 36, 37, 38, and 39. Each
init is one independent 30-episode shard in T, TF_CONT, MLP_CONT order, full
800-step horizon. Four complete shards produce 120 episodes / 40 paired
physical task-init groups. `TEST-MANIFEST.json` is a disabled reference to the
prospectively reserved 40–49 list. A case key is an opaque identifier: runner,
server, and analyzer use the manifest's actual arm/task/init fields and never
parse a key to choose a task or reuse an old T episode.

The target WAM, original 512-BF16 text bank, K2/B2 bonus, threshold,
`prefix_then_cost`, O3, actual reset/cache path, and seven action channels are
the same as today's frozen entry. The fixed seed-23 TF_CONT and MLP_CONT 3000
update checkpoints have hashes in `DEV-MANIFEST.json`. Each draft arm must pass
all 20,237 real CAL queries on the same GPU before its first full episode;
T drafts off. The target's initial CPU/CUDA RNG is captured before loading a
draft and restored for every episode. The actual first two camera images,
physics/controller state, target RNG, commands, actions, latents, acceptance
counters, and all saved output coverage remain auditable. Initial camera pixel
mismatches are reported separately; they never become a tolerance or a
post-hoc claim of exact-input replay.

CPU `prepare(job, manifest=..., shard=..., registration=...)` refuses until
today's actual 50-episode COLLECTION, analysis retention decision, all owned
cleanup including the attached clock, and absence of failure/STOP markers pass.
It binds those original receipts by path and bytes, without copying any old
FAILED file into a new active shard. The predeclared retention decision is
research admission only: engineering quality, independent TEST completion,
3-point noninferiority, and formal performance permission remain false.

A later immutable `draft-frozen-validation-execution-registration/v1` JSON must
specify `FROZEN_BEFORE_EXPANDED_DEV`, `manifest_sha256`,
`test_manifest_sha256`, ordered `shard_order`, four absolute `shard_jobs`,
`cohort_started_at`, `absolute_work_stop`, `absolute_cleanup_deadline`, and
`cohort_metadata_root` equal to the registration's own isolated runtime
directory. It also names a new independent clock `clock_wrapper_path`,
`clock_wrapper_sha256`, and
`clock_provenance_name=DRAFT-FROZEN-VALIDATION-CLOCK-PROVENANCE.json`. Numeric
caps are fixed: 14,400 seconds per actual supervisor birth, 300 seconds
reserved for cleanup, 28,800 charged GPU-job lifetime seconds for the cohort,
43,200 seconds maximum cohort wall time, 3 GiB new output per shard, 5 GiB
cohort output, and 5 GiB free-disk floor. Both server and client have two CPU
threads and one WAM worker/EGL client. Shared GPU is allowed only for quality;
wall times are not formal speedups. The earliest lifetime or absolute deadline
wins. The separate clock must compute the hard stop using both the current
supervisor's real process birth and prior `CLOCK-RESULT` charged lifetimes.
There must be more than 300 seconds of cohort lifetime left before another
shard may prepare. Cohort output counts metadata and all shard directories.

Only the first shard may prepare without prior shard results. Each later shard
requires every previous 30-episode RESULT/COLLECTION, three arm cleanups,
FINAL cleanup, independent CLOCK-RESULT, no FAILED/STOP, and an explicit
`NEXT-SHARD-RELEASE.json` in the immediately previous job. Its status must be
`ALLOW_NEXT_FROZEN_VALIDATION_SHARD`, name the next shard, and bind the prior
RESULT and CLOCK-RESULT SHA values. Preparing or running a shard does not
launch the next. Partial/failed shards block cohort quality pooling and all
subsequent shards. The sum of recorded `CLOCK-RESULT.elapsed_quality_process_seconds`
is charged, including setup, waiting, and analysis; neither a later `prepare`
nor a new attached clock may reset that sum. Current plus prior artifact size
is bounded at 5 GiB.
The current pause/release point is the **30-complete-episode init-shard
boundary**. A formal operator timing request may wait for the active shard to
finish and clean up; this runner does not promise immediate per-episode pause
or resume, and never truncates or reruns a selected episode.

After real registration and independent source review, the CPU-only preparation
for the first shard is:

```bash
python -B 投机采样/experiments/draft_frozen_validation_v1/run.py \
  --prepare \
  --job /abs/runtime/p0-draft-frozen-validation-v1-local-init36-... \
  --manifest /rjs/cxz/论文探索/投机采样/experiments/draft_frozen_validation_v1/DEV-MANIFEST.json \
  --shard init36 \
  --registration /abs/path/to/frozen-execution-registration.json
```

Run takes only `--job` and the already frozen plan. The root-owned detached
launcher must record a real `ROOT-DETACHED-LAUNCH.json` PID/start ticks/argv and
attach the new independently reviewed CPU clock immediately. The supervisor
writes `RUN-STARTED.json`, then waits at most 60 seconds for an exact live
`CLOCK-GUARD.json` before constructing the GPU campaign. No model or client
starts if the clock is absent, stale, foreign, or over budget. Each shard writes
its own COLLECTION, REPORT/PAIRING, and RESULT only after complete episodes,
full CAL and saved-output checks, all arm cleanup, failure-first scan, and
FINAL owned cleanup. The result is research DEV evidence, never a TEST or
formal performance certificate. A separate CPU merge may describe all four
shards only when every original shard terminal receipt and clock is complete.
