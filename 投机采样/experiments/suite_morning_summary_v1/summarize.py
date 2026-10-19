"""CPU-only audit of returned Goal/Spatial reports; no model or new episodes."""
from collections import Counter
import csv
from datetime import datetime
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    out = root / ".pipeline/experiments/libero-four-suite-morning-20261019"
    result = dict(at=datetime.now().astimezone().isoformat(), suites={},
                  interpretation="descriptive exposed-suite evaluation, not independent TEST",
                  engineering_quality_screen=False, independent_test_complete=False,
                  noninferiority_established=False, formal_performance_permission=False)
    csv_rows = []
    for suite in ("goal", "spatial"):
        source = out / "raw" / suite / "reports/final/RESULT.json"
        original = json.loads(source.read_text())
        rows = original["rows"]
        expected = {(case, arm) for case in range(500) for arm in ("T", "C", "P")}
        assert len(rows) == 1500 and {(r["case_id"], r["arm"]) for r in rows} == expected
        assert all(r["complete"] and r["task"] == r["case_id"] % 10
                   and r["init"] == r["case_id"] // 10 for r in rows)
        index = {(r["case_id"], r["arm"]): r for r in rows}
        for arm, aggregate in original["by_arm"].items():
            selected = [r for r in rows if r["arm"] == arm]
            assert len(selected) == aggregate["complete"] == 500
            assert sum(r["success"] for r in selected) == aggregate["successes"]
            assert sum(r["actual_command_steps"] for r in selected) == aggregate["actual_steps"]["sum"]
            assert sum(r["failure800_penalty"] for r in selected) == aggregate["all_complete_failure800_penalty"]["sum"]
            assert all(r["failure800_penalty"] == (r["actual_command_steps"] if r["success"] else 800)
                       for r in selected)
        comparisons = []
        for old in original["comparisons"]:
            a, b = old["reference"], old["candidate"]
            pairs = [(index[case, a], index[case, b]) for case in range(500)]
            assert sum(x["success"] and not y["success"] for x, y in pairs) == old["lost_success"]
            assert sum(not x["success"] and y["success"] for x, y in pairs) == old["rescued"]
            ratios = [y["actual_command_steps"] / x["actual_command_steps"]
                      for x, y in pairs if x["success"] and y["success"]]
            d = dict(old)
            d["common_success_ratio_over_1p30"] = sum(r > 1.30 for r in ratios)
            comparisons.append(d)
        result["suites"][suite] = dict(
            complete_episodes=1500, case_groups=500, source=str(source.relative_to(root)),
            source_producer_commit=original["source_producer_commit"],
            by_arm=original["by_arm"], comparisons=comparisons,
            component_status_histogram={arm: dict(Counter(r["component_status"] for r in rows if r["arm"] == arm))
                                        for arm in ("T", "C", "P")},
            formal_E2E_speedup="UNKNOWN",
            strict_pairing=original["strict_initial_physics_camera_and_RNG_pairing"],
            FIT_CAL_exposure_retained=True, audit="PASS_RECOMPUTED_COUNTS_STEPS_PENALTIES_AND_PAIRED_WINS")
        for r in rows:
            csv_rows.append({k: r[k] for k in ("suite", "case_id", "task", "init", "arm", "success",
                "actual_command_steps", "failure800_penalty", "policy_chunks", "strict_components_pass",
                "component_status", "exposure")})
    (out / "RESULT.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    with (out / "EPISODES-SUMMARY.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = ["# 10月19日两机晨间核验", "", "Goal于08:41、Spatial于08:00完成各500组三臂，均为北京时间。完整机器人终态与组件审计分列。", "",
             "|套件|臂|成功/500|成功率|800罚分均值|实际命令总数|严格组件通过/500|",
             "|---|---|---:|---:|---:|---:|---:|"]
    for suite, s in result["suites"].items():
        for a, d in s["by_arm"].items():
            lines.append(f"|{suite}|{a}|{d['successes']}|{d['success_rate']:.1%}|{d['all_complete_failure800_penalty']['mean']:.3f}|{d['actual_steps']['sum']}|{d['strict_component_complete']}|")
    lines += ["", "T为Target，C为连续TF，P为同权重TF+PF2。", "",
              "|套件|比较|救回/丢失|共同成功长度P50/P90/max|长度比>1.30|", "|---|---|---|---|---|"]
    for suite, s in result["suites"].items():
        for d in s["comparisons"]:
            q = d["common_success_steps_ratio"]
            lines.append(f"|{suite}|{d['reference']}→{d['candidate']}|{d['rescued']}/{d['lost_success']}|{q['p50']:.4f}/{q['p90']:.4f}/{q['max']:.4f}|{d['common_success_ratio_over_1p30']}/{q['n']}|")
    lines += ["", "分位数使用线性Type7。全体800罚分/实际步数/自然失败全部保留，成功子集长度仅为补充。两套C/P均有FAILED_FIRST_TWO_S_PAIRING，不签严格配对/独立TEST/非劣；四项false保持。", "",
              "并行质量运行中的C_update及ΣT_round/Σg原样保留于RESULT.json，只是探索性成本。正式E2E加速UNKNOWN，不能用target调用减少替代；所有臂使用既定512-token BF16优化reset，冷启动另列。", "",
              "Object旧30局有完整终态，但新14条compact编号冲突、collector准入失败，仅前16条有效；用户已明确批准补14。新v3 namespace的固定14补采与原匹配Target200已双卡实际运行（首2局/臂终态、首补采10shard无collector错误），其后ON3000与Object500组等待依赖自动接续。旧失败不改，原M17截止不重置；见OBJECT-RECOVERY-PLAN.md与RECOVERY-CAPTURE-LIVE.json。", "",
              "本机昨夜CPU顺序B0通过；今晨GPU六真实fixture×两种缓存的66执行通过。A/A最大2.513%超过2.5%门限，计时停止，GPU0显存0且NVML24–25%异常；低开销监控等待状态改变。见../wam-block-gpu-followup-20261019/RESULT.md。", "",
              "原始返回报告保留于raw/（约18MiB，不默认在GitHub重复发布）；RETURN-MANIFEST.json登记源路径/大小/时间。本文统计由summarize.py纯CPU独立核对1500唯一行/套、成败/步数/罚分与配对胜负，未重算原始velocity、严格初态或RNG。"]
    (out / "RESULT.md").write_text("\n".join(lines) + "\n")
    print("PASS: unique complete rows, counts, penalties, paired wins; four scientific gates remain false")


if __name__ == "__main__":
    main()
