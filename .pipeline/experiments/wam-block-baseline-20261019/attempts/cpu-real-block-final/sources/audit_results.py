"""Recompute bounded descriptive summaries from fixed old CSVs, never regrade."""
import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import statistics
import subprocess

REVIEW_COMMIT = "ca5f37e0482c2e38e3738e3d8e71e3b185990a23"
PREFIX = ".pipeline/experiments/operator-torch-alignment-20261018/performance/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def table(path):
        raw = subprocess.check_output(["git", "show", f"{REVIEW_COMMIT}:{path}"], cwd=args.repository)
        return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    points = table(PREFIX + "PERFORMANCE-ONLY.csv")
    full = table(PREFIX + "FULL30.csv")
    selected = [row for row in points if row["part"] == "block" and row["engine"] == "standalone"]
    torch = [row for row in points if row["part"] == "block" and row["engine"] == "torch"]
    tails = [row for row in selected if float(row["engine_over_torch"]) > 1.05]
    variation = [dict(identity={key: row[key] for key in ("length", "batch", "kind", "query_role", "layer")},
                      min_round_ms=float(row["min_round_ms"]), max_round_ms=float(row["max_round_ms"]),
                      max_over_min=float(row["max_round_ms"]) / float(row["min_round_ms"])) for row in torch]
    ratios = [float(row["engine_over_torch"]) for row in selected]
    result = dict(review_commit=REVIEW_COMMIT,
                  scope="old R1 attention-replacement whole-block timings; not new B0 measurements",
                  complete_block_points=len(selected),
                  R1_over_R0=dict(median=statistics.median(ratios), min=min(ratios), max=max(ratios),
                                 geomean=math.exp(statistics.mean(map(math.log, ratios)))),
                  slower_than_5pct=len(tails),
                  slow_group_counts={str(key): value for key, value in Counter(
                      (row["batch"], row["kind"]) for row in tails).items()},
                  slow_points=tails,
                  R0_round_variation=dict(points=len(variation),
                      greater_than_5pct=sum(row["max_over_min"] > 1.05 for row in variation),
                      greater_than_10pct=sum(row["max_over_min"] > 1.10 for row in variation),
                      max_over_min_median=statistics.median(row["max_over_min"] for row in variation),
                      max_over_min_max=max(row["max_over_min"] for row in variation),
                      worst=sorted(variation, key=lambda row: row["max_over_min"], reverse=True)[:6]),
                  full30_points=full,
                  noise_interpretation="OBSERVED round variation only. Host launch gaps, device interference, clocks and thermal effects are not separated; no causal root conclusion.",
                  AA_control="NOT_PRESENT_IN_OLD_MEASUREMENT_DESIGN",
                  permission_changes=False,
                  new_B0_performance="NOT_RUN_NO_GPU_AUTHORIZATION")
    (args.output / "OLD-PERFORMANCE-NOISE-AUDIT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("complete_block_points", "R1_over_R0", "slower_than_5pct", "R0_round_variation")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
