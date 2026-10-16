# Expanded DEV conditional dispatcher (source only)

This package has not registered or run a cohort. It schedules only the four
frozen `init36`–`init39` DEV shards after the original today50 result retains
the fixed TF candidate and all original cleanup and clock receipts exist.
`draft_frozen_validation_v1.protocol` remains the admission authority. The
disabled TEST40–49 manifest is never passed to `prepare` or `run`.

Once the operator has separately created and frozen a real registration, the
single invocation is:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python -B /rjs/cxz/论文探索/投机采样/experiments/draft_frozen_validation_dispatch_v1/dispatch.py \
  --registration /rjs/cxz/论文探索/投机采样/results/runtime/p0-draft-frozen-validation-v1-cohort-<ID>/REGISTRATION.json
```

The placeholder above is explanatory CLI syntax, not an existing receipt.
The dispatcher refuses registration before the real today50 retention and
cleanup. Its fixed wall window is 2026-10-17 08:00–20:00 Asia/Shanghai,
including CPU waiting. The dispatcher has a separate 32-hour whole-life cap
from its actual process birth; the 20:00 absolute deadline takes precedence.
The registration's earlier absolute work and cleanup
limits still apply. Each shard has an actual-birth 4-hour cap with 300 seconds
for cleanup, the cohort charges at most 8 hours of GPU-job lifetimes, and the
cohort wall cap is 12 hours. The original protocol and independent clock
enforce 3 GiB per shard, 5 GiB across the cohort, and a 5 GiB free-disk floor.

The dispatcher prepares one complete 30-episode shard, copies the exact
`launch_one.py` source into that job as `ROOT-DETACHED-LAUNCH.py`, and runs the
copy once. The launcher records the real detached supervisor PID/start ticks,
assigns that worker the fixed four CPU cores 12–15 while the launcher and
clock use at most two, inspects the clock against that live process, and starts an independent
detached clock before the supervisor's startup barrier can admit GPU work.
The next shard receives `NEXT-SHARD-RELEASE.json` only after the preceding
original RESULT, complete collection, owned cleanup and clock cleanup are
verified. After all four clean terminals, the frozen analyzer recomputes all
120 saved episodes on CPU and writes `COHORT-ANALYSIS.json`; only its complete
status allows `DISPATCH-COMPLETE.json`. A `PAUSE_AFTER_SHARD` file inside the
registration's metadata directory pauses before the first shard or at a
completed shard boundary; removing it resumes
before the overall deadline. Pause receipt names include the next shard.

There is no retry, skipped episode, TEST launch, or claim of independent
qualification or formal performance. Any failure keeps the original job
receipts and writes `DISPATCH-FAILED.json` in cohort metadata. A stopped
supervisor's independent clock is allowed to finish owned cleanup before the
dispatcher signs failure, for at most 300 seconds within the remaining
deadline; missing cleanup is recorded as unverified. The active shard is never interrupted solely for a
boundary pause. This source package requires final operator review before
real registration or execution.
