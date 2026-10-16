# TF usability: two-site complete DEV

This isolated entry records 40 full LIBERO Long10 episodes at each site: `T`,
`TF_CHILD` (seed 20260923, 2500 updates), `TF_CONT` (same seed, 3000 updates),
and `MLP_CHILD` (matched seed 20260923, 2500 updates), tasks 0–9 in that order.
Local uses init 33; AutoDL uses init 34. Neither site reuses an earlier T.
Horizon 800, K2/B2 bonus, O3, `prefix_then_cost`, and τ
`0.007839680526450805` are fixed. These are DEV quality episodes, not formal
speed measurements or a 3 percentage point noninferiority test.

The common round began 2026-10-16 08:48:35 Beijing. Each site may run at most
10,800 seconds including setup/analysis, with 300 seconds reserved for owned
cleanup. Work stops no later than 12:43:35 and cleanup completes no later than
12:48:35 Beijing; the earlier site-specific deadline wins. Thus two concurrent
sites use at most six GPU hours. Each site is one model server and one EGL
client at a time, with 3 GiB new output and a 5 GiB disk floor. Local server
uses two CPU threads, matching the prior complete local quality runner; the
remote server uses seven of the observed ten CPU quota, with client two.

The real local CPU-only plan was prepared and loaded successfully at
`results/runtime/p0-tf-usable-quality-v1-local-i33-20261016-r1/plan.json`.
It binds 162 actual source/native files, actual TF/MLP child23 and TF-cont23
checkpoints, completed CPU training receipts, full 20,237-row CAL pack, the
eleven-prompt BF16 bank and both its build and full reset-equivalence receipts,
and target weight stat metadata. No GPU episode has been started by prepare.
The exact run command, after independent final-byte review, is:

```bash
python 投机采样/experiments/tf_usable_quality_v1/run.py \
  --job 投机采样/results/runtime/p0-tf-usable-quality-v1-local-i33-20261016-r1
```

Remote deployment must preserve the original absolute `/rjs/cxz/论文探索`
checkpoint lineage alias; never edit `.pt` metadata. Copy the 162 files listed
in the local plan's `source_sha256` to matching paths under the remote project
alias. Copy only these small training artifacts under their original relative
paths: stage1 TF23 and MLP23 `checkpoint.pt` plus `RESULT.json`; stage2 TF23 and
MLP23 `checkpoint.pt`, `RESULT.json`, `cal_predictions.pt`, and stage2
`FINAL-RESULT.json`; TF-cont23 `checkpoint.pt`, `RESULT.json`,
`cal_predictions.pt`, `selected.json`, and `FINAL-RESULT.json`. Keep original
target weights and existing remote BF16 bank/CAL pack in place. The remote
`source_files()` and strict loader determine if any dependency is missing.
Do not copy all six stage2 checkpoints or any target weights merely to satisfy
the first-stage DEV protocol.

The remote CPU-only `--prepare --site autodl` uses the same explicit arguments
as the already successful local prepare, with site-specific paths: new job under
the remote project `results/runtime` with prefix `p0-tf-usable-quality-v1-`,
the installed seed23 checkpoints/training directories, existing
`remote-cpu7-bank/bank.pt` and its `RESULT.json`, original remote
`RESET-BANK-EQUIVALENCE.json` (two T/Aoff pairs, four full episodes), existing
compact `handoff/data/actual_cal20_columnar.pt`, target checkpoint/registry,
and the remote old 104 plan as `--target-weight-stats`. The latter source has a
`target_weight_stats` object and a nested `target` object; pass its exact
checkpoint, id, and registry values. The parser refuses missing explicit
inputs. After remote prepare, `protocol.load(job)` must pass on the remote
actual files before running `run.py --job JOB`.

The full 20,237-query same-GPU original-API parity runs for each new arm before
its first episode. The target CPU/CUDA RNG is captured before draft loading and
CAL, restored before every reset, and saved as real tensors. Each episode
saves actual two-camera initial pixels, initial/final sim state, all commands,
request/acceptance counters, and original actions/latents/observations. The
analysis requires exact task/init/physical/controller/target-RNG pairing and
reports every initial-camera pixel difference. Original strict engineering
screening additionally requires exact initial camera pixels **and** success
and both-success step thresholds. Each site is provisional; only the separate
two-site 20-pair merge can decide a joint research direction. Neither a
complete collection nor one-site numeric success grants formal speed testing.

No retry, shortened episode, changed threshold, changed seed, or reuse of an
old failed baseline is permitted. If the three-hour/common budget stops a job,
retain the PARTIAL/FAILED result, original episode evidence, and owned cleanup
receipt. A completed collection is written before analysis; final `RESULT.json`
is written only after report, failure scan, and final owned process cleanup.

### Historical receipts and the active failure scan

The AutoDL init34 r2 collection completed all 40 episodes, but its terminal
scan matched the documentary `PREDECESSOR-FAILED-ADMISSION.json`. Its original
FAILED and absent RESULT are retained. Separate read-only raw audits do not
replace that execution terminal or grant quality/performance permission.

For future jobs, keep historical failure-admission documents in a separate
parent evidence directory and bind their original paths/hashes in the
cumulative budget ledger. Before launching GPU work, scan the entire active
job directory with the same `*FAILED*.json`/`STOP_REQUESTED` rule. A documentary
name collision must block admission. Do not exempt active failures, move old
receipts after a run, or edit registered thresholds to repair this incident.

Future launchers can run `python experiments/tf_usable_admission_v1/check.py
--job JOB` from the speculative root before the original source/budget
admission. This isolated CPU tool checks layout only; existing running and
failed campaign sources are unchanged.
