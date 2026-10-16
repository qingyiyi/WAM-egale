"""CPU-only diagnostic ratios from a real complete six-arm result and cleanup."""
import argparse
import json
import math
from pathlib import Path
from baseline_contract import ARMS, load, read, require, write


def analyze(job):
    job = Path(job).resolve()
    load(job)
    require(not list(job.rglob('*FAILED*.json')) and not (job / 'STOP_REQUESTED').exists(), 'failure/STOP first')
    result, terminal = read(job / 'BASELINE-RESULT.json'), read(job / 'SUPERVISOR-RESULT.json')
    require(result['status'] == 'COMPLETE_STATIC_SAME_SHAPE_PROMPT_BASELINE_PENDING_CLEANUP'
        and result['all_36_eager_admissions_passed'] and len(result['admission']) == 36
        and result['all_six_queries_all_six_arms_complete'] and len(result['rows']) == 6
        and result['target_forward_intents'] == 410 and result['captured_forward_intents'] == 36
        and result['actual_device_forward_equivalent'] == 374, 'complete full recipe required')
    require(terminal['status'] == 'COMPLETE_STATIC_SAME_SHAPE_PROMPT_BASELINE'
        and terminal['failure'] is None and terminal['own_process_reaped']
        and terminal['all_owned_live_exited'], 'real owned cleanup required')
    comparisons = (('n1_dense_uncached','n1_dense_cached'), ('n7_native_uncached','n7_native_cached'),
                   ('n7_guarded_uncached','n7_guarded_cached'), ('n7_native_cached','n7_guarded_cached'))
    rows = []
    for row in result['rows']:
        require(row['valid_KV_RNG_inputs_slots_versions_and_cached_outputs_unchanged'], 'state gate')
        for arm in ARMS:
            samples = row['timing'][arm]['samples']
            require(len(samples) == 6 and all(s['same_shape_original_bitwise_and_finite']
                and all(math.isfinite(s[k]) and s[k] > 0 for k in ('wall_ms','cuda_ms')) for s in samples),
                'all six original samples required')
        ratios = {}
        for left,right in comparisons:
            ratios[left+'__to__'+right] = {metric: dict(
                speed_ratio=row['timing'][left]['median_'+metric]/row['timing'][right]['median_'+metric],
                latency_reduction_fraction=1-row['timing'][right]['median_'+metric]/row['timing'][left]['median_'+metric])
                for metric in ('wall_ms','cuda_ms')}
        strong = {metric: row['timing']['n7_guarded_cached']['median_'+metric]
            / row['timing']['n1_dense_cached']['median_'+metric] for metric in ('wall_ms','cuda_ms')}
        rows.append(dict(length=row['length'], query=row['query'], total_KV=row['total_KV'],
            same_shape_comparisons=ratios, cached_B14_over_cached_single_state_cost=strong,
            local_break_even_not_sumT_over_sumg=True))
    summary = dict(status='COMPLETE_CPU_STATIC_BASELINE_DIAGNOSTIC_ONLY', rows=rows,
        prefills=result['prefills'], all_six_samples_preserved=True,
        no_cold_or_dynamic_lifecycle_in_hot_time=True, no_formal_robot_performance=True,
        no_trace_success_or_TF_quality_inference=True,
        caution='不同batch的原oracle分别认证；不是比较cached B14与uncached单状态。首次填充wall另列，不能忽略生产失效、Graph重建及未消费首填输出。')
    write(job / 'BASELINE-ANALYSIS.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.job)
    print(json.dumps(dict(status=result['status'], rows=len(result['rows'])), ensure_ascii=False))
