"""CPU-only reduction of all completed, pre-registered timing processes."""
from pathlib import Path
import csv
import hashlib
import json
import statistics as stats

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text())


def dump(name, value):
    (ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_file(name, rows):
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


full = []
for phase in ('cross', 'ffn', 'profile'):
    for process in (1, 2):
        data = read(f'{phase}/TIMING-{process}.json')
        assert data['status'] == 'COMPLETE_SIMPLE_BEFORE_NATIVE_COMPARISON'
        assert all(len(x['kernels']) == 1 for x in data['traces'])
        for case in ('short', 'middle', 'long'):
            medians = {}
            for name in ('native', 'before', 'simple'):
                values = [x['ms'] for x in data['samples'] if x['case'] == case and x['name'] == name]
                assert len(values) == 12
                medians[name] = stats.median(values)
            full.append(dict(phase=phase, process=process, case=case,
                             native_ms=medians['native'], before_ms=medians['before'], after_ms=medians['simple'],
                             latency_reduction_pct=100 * (1 - medians['simple'] / medians['before']),
                             after_slower_than_native_pct=100 * (medians['simple'] / medians['native'] - 1)))
csv_file('FULL-LAYER-ALL.csv', full)

phase_rows, perturbations = [], []
profile = read('profile/RESULT.json')
assert profile['status'] == 'COMPLETE_PHASE_DIAGNOSTIC_WITH_INSTRUMENTATION_CONTROL'
for row in profile['rows']:
    for stage in ('self_Q', 'cross_K', 'cross_V', 'FFN1', 'FFN2'):
        values = {arm: stats.median(x['us'] for x in row['stage_samples']
                                   if x['arm'] == arm and x['stage'] == stage)
                  for arm in ('before_marked', 'after_marked')}
        phase_rows.append(dict(case=row['case'], stage=stage,
                               before_us=values['before_marked'], after_us=values['after_marked'],
                               reduction_pct=100 * (1 - values['after_marked'] / values['before_marked'])))
    for label in ('before', 'after'):
        values = {kind: stats.median(x['ms'] for x in row['samples'] if x['arm'] == label + '_' + kind)
                  for kind in ('plain', 'marked')}
        perturbations.append(dict(case=row['case'], version=label,
                                  plain_ms=values['plain'], marked_ms=values['marked'],
                                  perturbation_pct=100 * (values['marked'] / values['plain'] - 1)))
csv_file('GEMM-PHASES.csv', phase_rows)
csv_file('INSTRUMENTATION-CONTROL.csv', perturbations)

gates = {}
for phase in ('cross', 'ffn'):
    gate = read(phase + '/SIMPLE-GATE-RESULT.json')
    assert gate['all_pass'] and len(gate['rows']) == 9
    gates[phase] = dict(real_signatures=9, shared_intermediates_per_signature=23,
                       valid_state_transitions=sum(len(r['legal_state_sequences']) for r in gate['rows']),
                       output_max_abs=max(r['new_vs_Torch']['max_abs'] for r in gate['rows']),
                       inputs_and_RNG_unchanged=all(r['inputs_unchanged'] and r['RNG_unchanged'] for r in gate['rows']),
                       finite_gate_pass=True, production_admitted=False)

statuses = [read(d + '/STATUS.json') for d in ('cross', 'ffn', 'profile')]
assert all(s['status'] == 'COMPLETE_ATTEMPT' for s in statuses)
jobs = [j for s in statuses for j in s['jobs']]
gpu = sum(j['lifecycle_seconds'] for j in jobs if j['gpu'])
cpu = sum(j['lifecycle_seconds'] for j in jobs if not j['gpu'])
plan = read('PLAN.json')
budget = dict(GPU_worker_lifecycle_seconds=gpu, CPU_build_wall_seconds=cpu,
              GPU_prior_seconds=plan['cumulative_prior_GPU_seconds'],
              GPU_cumulative_seconds=plan['cumulative_prior_GPU_seconds'] + gpu,
              GPU_cumulative_cap_seconds=36000,
              GPU_remaining_seconds=36000 - plan['cumulative_prior_GPU_seconds'] - gpu,
              current_GPU_cap_seconds=1800, absolute_deadline=plan['deadline'],
              job_wall_seconds=sum(s['lifecycle_seconds'] for s in statuses),
              elapsed_first_launch_to_last_exit_seconds=max(s['end'] for s in statuses) - min(s['start'] for s in statuses),
              CUDA_active_seconds='UNKNOWN_NOT_INFERRED_FROM_LIFECYCLE',
              all_workers_reaped=all(j['root_reaped'] and not j['identity_alive'] for j in jobs),
              failures_this_round=0, candidates_tested=2,
              whole_block_calls=2 * (72 + 2 * 651) + 819 + 2 * 651)
assert gpu <= 1800 and cpu <= 2400
dump('RESOURCE-AND-STOP-REPORT.json', budget)

source = ROOT.parents[3] / '算子研究/wam_ops/optimized/csrc/whole_layer.cu'
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == read('profile/SOURCE-MANIFEST.json')[1]['SHA256']
result = dict(status='COMPLETE_RETAIN_CROSS_AND_FFN_NOT_SPEED_PARITY',
              active_source=str(source), active_source_SHA256=source_hash,
              final_direct_comparison=[x for x in full if x['phase'] == 'profile'],
              incremental_comparisons=[x for x in full if x['phase'] != 'profile'],
              finite_gates=gates, GEMM_phase_diagnostic=phase_rows,
              instrumentation_control=perturbations, resources=budget,
              single_kernel_observed=True, cross_stage_ready_implemented=False,
              engineering_quality_screen=False, independent_test_complete=False,
              noninferiority_established=False, formal_performance_permission=False,
              scope='Actual finite fixtures, uncached eager single layer, not robot E2E or strongest cached/Graph baseline')
dump('RESULT.json', result)
dump('FINAL-SELECTION.json', dict(status='RETAIN_BOTH', source_SHA256=source_hash,
                                 cross='stable large whole-layer improvement',
                                 ffn='small consistent ~0.6% incremental whole-layer improvement',
                                 next='FFN primitive/tile and epilogue diagnosis before ready scheduling',
                                 previous_decisions_preserved=True))
print(json.dumps(dict(budget=budget, final=result['final_direct_comparison']), indent=2))
