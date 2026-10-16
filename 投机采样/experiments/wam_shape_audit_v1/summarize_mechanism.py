"""Derive mechanism metrics from saved real measurements, never estimate HBM use."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
JOB = ROOT / '投机采样/results/runtime/p0-wam-shape-audit-v1-20261016-r1'


def main():
    micro = json.loads((JOB / 'long-mechanism/REPLAY-RESULT.json').read_text())
    profile_path = ROOT / '投机采样/experiments/megaverifier_analysis_v1/V2-SIX-TRACE-SUMMARY.json'
    profiles = json.loads(profile_path.read_text())
    assert micro['status'] == 'COMPLETE_LONG_FIXTURE_MECHANISM_ONLY'
    assert profiles['status'] == 'PASS_CPU_ATTRIBUTION_6_COMPLETED_PROFILES'
    rows = []
    for r in micro['rows']:
        m, n, k = r['projection_MNK']
        latency = r['projection']['median_ms']
        # Conventional dense matmul operation count, excludes bias and data copies.
        flops = 2 * m * n * k
        t = r['timing']
        rows.append(dict(layer=r['layer'],states=r['n_v'],MNK=[m,n,k],
            projection_ms=latency,projection_matmul_FLOPs=flops,
            derived_matmul_TFLOPs_per_sec=flops / (latency * 1e9),
            sequential_attention_over_batched_core=t['sequential_B2_core']['median_ms'] / t['batched_core']['median_ms'],
            batch_projection_equal_to_sequential=r['projection_seq_vs_batch']['bitwise'],
            attention_subset_equal_to_full=r['subset_vs_full']['bitwise'],
            shared_vs_dense_with_pack=(t['batched_with_history_pack']['median_ms']/t['existing_shared_core']['median_ms']) if r['n_v']==7 else None,
            shared_vs_dense_core=(t['existing_shared_core']['median_ms']/t['batched_core']['median_ms']) if r['n_v']==7 else None))
    interactions = []
    for layer in (0,8,29):
        one = next(r for r in rows if r['layer']==layer and r['states']==1)
        seven = next(r for r in rows if r['layer']==layer and r['states']==7)
        interactions.append(dict(layer=layer,
            projection_row_throughput_ratio=7*one['projection_ms']/seven['projection_ms'],
            meaning='row throughput ratio, NOT measured seven sequential Linear calls or robot speed'))
    profile = []
    for p in profiles['profiles']:
        profile.append(dict(trace=p['trace'],unique_GPU_kernel_duration_sum_ms=p['unique_duration_ms'],
            cross_KV_share=p['cross_attn_KV_projection_60_kernel_ms']/p['unique_duration_ms'],
            self_attention_subgraph_share=p['self_attn_ms']/p['unique_duration_ms'],
            ffn_share=p['ffn_ms']/p['unique_duration_ms'],
            ideal_self_subgraph_elimination_bound=p['unique_duration_ms']/(p['unique_duration_ms']-p['self_attn_ms']),
            bound_is_not_achievable_attention_kernel_speed_or_wall_prediction=True))
    out=dict(status='COMPLETE_CPU_DERIVATION_FROM_SAVED_REAL_MEASUREMENTS',
        sources=[str(JOB/'long-mechanism/REPLAY-RESULT.json'),str(profile_path)],
        actual_micro_rows=rows,descriptive_shape_economy=interactions,historical_profile=profile,
        H1='PARTIAL: real row-throughput/actual kernel changes; hardware counters absent',
        H2='PARTIAL: same frozen attention queries only; full verifier matrix pending',
        H3='UNTESTED: no new scheduling result against best B14 baseline',
        H4='UNTESTED_CURRENT_CANDIDATE: quality not admitted; old A_off results separate',
        no_HBM_or_L2_traffic_inferred=True,no_peak_TFLOPs_assumed=True,
        measurement_limits=['single process pilot, six retained rounds',
            'warm operator Graph timings exclude model loading/pack unless explicitly named',
            'profile kernel sums are neither end-to-end latency nor formal performance',
            'different batch shapes can change arithmetic, so this is not equivalent closed-loop inference'])
    path=JOB/'MECHANISM-DERIVED.json'
    with path.open('x') as f:json.dump(out,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(status=out['status'],rows=len(rows),path=str(path)),ensure_ascii=False))


if __name__=='__main__':main()
