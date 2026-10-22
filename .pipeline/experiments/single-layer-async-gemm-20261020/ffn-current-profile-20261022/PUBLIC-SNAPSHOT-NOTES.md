# Public profile snapshot

PROFILE-1.json.gz and PROFILE-2.json.gz are lossless copies of the local raw timestamp JSON. Decompress them to PROFILE-1.json / PROFILE-2.json before running analyze.py. ANALYSIS.json / SUMMARY.json / RESULT.md are directly readable summaries.

NCU binary report, compiled DSO, large tensor fixtures and model weights stay on the authorized research servers. local-counters/RAW.csv contains exported metrics with units. No SSH keys, credentials, environment dumps or robot raw observations are uploaded.

RoboTwin CHECKPOINT.json is a timestamped preparation snapshot, not a live dashboard or completed quality result. GPU0 is reserved for operator work; GPU1 alone owns the new fixed 1800-episode chain.
