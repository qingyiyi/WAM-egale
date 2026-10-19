"""CPU-only derivation of the bounded native block evidence; no GPU work."""
from datetime import datetime
from pathlib import Path
import csv
import hashlib
import json
import os
import statistics

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
os.sched_setaffinity(0, {5})


def read(path):
    return json.loads(path.read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def csv_write(name, rows):
    with (HERE / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def result(name):
    return read(HERE / name / 'results/RESULT.json')


now = datetime.now().astimezone().isoformat()
engines = ('original_A', 'baseline', 'original_B', 'optimized')
processes = ('native-vector-isolated-process1', 'native-vector-isolated-process2')
timing_rows, raw_rows = [], []
for process in processes:
    data = result(process)
    assert data['status'] == 'COMPLETE_DYNAMIC_BLOCK_FINITE_PARITY_AND_TIMING'
    assert len(data['rows']) == 9
    owned = read(HERE / process / 'OWNED-RESULT.json')
    assert owned['exit_code'] == 0 and owned['owned_exited']
    assert all(set(item['pids'].splitlines()) <= {str(owned['owned']['pid'])}
               for item in data['resources'])
    for row in data['rows']:
        assert row['inputs_unchanged'] and row['RNG_unchanged']
        assert all(x['passed'] for x in row['numeric'])
        assert len(row['raw']) == 28
        med = row['median_event_ms']
        wall = {e: statistics.median(x['wall_ms'] for x in row['raw'] if x['engine'] == e)
                for e in engines}
        timing_rows.append(dict(process=process, case=row['case'],
            **{e + '_event_ms': med[e] for e in engines},
            **{e + '_wall_ms': wall[e] for e in engines},
            baseline_over_original_A=med['baseline']/med['original_A'],
            optimized_over_original_A=med['optimized']/med['original_A'],
            baseline_over_original_B=med['baseline']/med['original_B'],
            optimized_over_original_B=med['optimized']/med['original_B'],
            optimized_over_baseline=med['optimized']/med['baseline'],
            original_B_over_A=med['original_B']/med['original_A'],
            numeric_and_state_bitwise_pass=True))
        raw_rows.extend(dict(process=process, case=row['case'], **sample) for sample in row['raw'])
csv_write('PERFORMANCE-PER-PROCESS.csv', timing_rows)
csv_write('PERFORMANCE-ALL-RAW.csv', raw_rows)

precision = result('native-vector-intermediates-final')
assert precision['status'] == 'STRICT_SIX_BLOCK_PASS'
assert len(precision['rows']) == 6 and all(x['passed'] for x in precision['rows'])
assert all(len(x['intermediates']) == 23 and len(x['primitives']) == 6 for x in precision['rows'])
state = result('native-vector-state-final')
opt_state = result('native-vector-optimized-states-final')
assert len(state['rows']) == 24 and all(x['passed'] for x in state['rows'])
assert len(opt_state['rows']) == 12 and all(x['passed'] for x in opt_state['rows'])
full = result('native-original-gradflags-final-attempt2')
assert full['status'] == 'COMPLETE_BLOCK_AND_ACTUAL30_FINITE_PARITY_NO_TIMING'
assert full['full_model_forwards'] == 18 and len(full['numeric']) == 42
assert full['original_parameter_grad_true_count'] == 839
assert full['all_parameters_and_classes_restored']
assert all(x['output']['bitwise_pass'] and x['output']['finite_pass'] and
           x['inputs_unchanged'] and x['RNG_unchanged'] and
           all(s['pass_all'] for s in x['state']) for x in full['numeric'])
capture = result('native-video-capture-attempt1')
assert len(capture['complete_policy_calls']) == 3
assert all(x['terminated'] for x in capture['complete_policy_calls'])

owned_rows, forward_rows = [], []
for receipt in sorted(HERE.glob('*/OWNED-RESULT.json')):
    item = read(receipt)
    owned_rows.append(dict(job=receipt.parent.name, status=item['status'],
        exit_code=item['exit_code'], lifecycle_seconds=item['lifecycle_seconds'],
        owned_exited=item['owned_exited'], owned=item['owned'],
        started_at=item['started_at'], finished_at=item['finished_at']))
    candidates = [receipt.parent / 'results/RESULT.json', receipt.parent / 'data/RESULT.json']
    payloads = [read(path) for path in candidates if path.exists()]
    assert len(payloads) <= 1
    d = payloads[0] if payloads else {}
    forward_rows.append(dict(job=receipt.parent.name,
        result_status=d.get('status', 'NO_RESULT_PRE_ADMISSION'),
        standalone_block_intents=d.get('block_forward_intents', d.get('block_forwards', 0)),
        primitive_intents=d.get('primitive_forward_intents', 0),
        model_forward_intents=d.get('full_model_forwards', d.get('target_forward_attempts', 0))))
assert all(x['owned_exited'] for x in owned_rows)
elapsed = sum(x['lifecycle_seconds'] for x in owned_rows)
direct_blocks = sum(x['standalone_block_intents'] for x in forward_rows)
model_calls = sum(x['model_forward_intents'] for x in forward_rows)
primitives = sum(x['primitive_intents'] for x in forward_rows)
assert elapsed <= 3600 and direct_blocks <= 12000 and model_calls <= 360 and primitives <= 20000
csv_write('FORWARD-ACCOUNTING.csv', forward_rows)

storage = sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())
assert storage < 2**30
write('RESOURCE-AND-STOP.json', dict(at=now,
    terminal='GPU_WORK_COMPLETE_OWNED_EXITED', jobs=owned_rows,
    lifecycle_seconds=elapsed, lifecycle_cap_seconds=3600, remaining_seconds=3600-elapsed,
    CUDA_active_seconds='UNKNOWN_NOT_MEASURED',
    instrumented_standalone_block_intents=direct_blocks, primitive_intents=primitives,
    model_forward_intents=model_calls, derived_blocks_inside_model_calls=model_calls*30,
    counter_scope='standalone block and complete model invocations have separate registered counters; derived internal blocks shown separately, not measured CUDA-active work',
    experiment_storage_bytes=storage, storage_cap_bytes=2**30,
    shared_correctness_only=True, final_timing_no_foreign_visible_PID=True,
    absolute_deadlines_not_reset=True, foreign_process_signals=False,
    clocks_services_changed=False, production_admitted=False))

files = []
for base in (ROOT/'算子研究/wam_ops', ROOT/'算子研究/tools'):
    for p in sorted(base.rglob('*')):
        if not p.is_file() or {'__pycache__','build'} & set(p.parts):
            continue
        if p.suffix in ('.py', '.cu', '.cuh', '.h', '.so') or p.name.startswith('LICENSE'):
            files.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
write('SOURCE-DELIVERED.json', dict(at=now, files=files,
    earlier_source_manifest='SOURCE-FINAL-NATIVE.json',
    math_binary='v7 vector-copy/CUB layout; same binary as final intermediates/state/30-layer/timing gates',
    last_adapter_change='read-only detached parameter aliases and inference-only guard; verified by default-grad-flags actual30 replay',
    scope='source and small operator binaries only; no model-weight hashes'))

summary = []
for case in [x['case'] for x in result(processes[0])['rows']]:
    rows = [x for x in timing_rows if x['case'] == case]
    ratios = [x[e+'_over_original_'+o] for x in rows for e in ('baseline','optimized') for o in ('A','B')]
    summary.append(dict(case=case,
        original_event_ms_two_process_mean=statistics.mean(x['original_A_event_ms'] for x in rows),
        baseline_event_ms_two_process_mean=statistics.mean(x['baseline_event_ms'] for x in rows),
        optimized_event_ms_two_process_mean=statistics.mean(x['optimized_event_ms'] for x in rows),
        baseline_ratio_range=[min(x['baseline_over_original_A'] for x in rows),max(x['baseline_over_original_A'] for x in rows)],
        optimized_ratio_range=[min(x['optimized_over_original_A'] for x in rows),max(x['optimized_over_original_A'] for x in rows)],
        worst_baseline_or_optimized_over_either_original=max(ratios)))
write('RESULT.json', dict(at=now,status='FINITE_NATIVE_BLOCK_PARITY_AND_TWO_PROCESS_DYNAMIC_TIMING_COMPLETE',
    complete_native_baseline=True,complete_native_optimized=True,
    finite_bitwise_gates_passed=True, actual_action30_queries=6, actual30_calls_final=18,
    key_intermediates_per_action_case=23, primitives_per_action_case=6,
    action_video_state_gates=24,optimized_action_state_gates=12,
    original_grad_flags_preserved=True, dynamic_block_cases_per_process=9,
    independent_timing_processes=2, raw_timing_samples=len(raw_rows),performance=summary,
    AA_max_abs_relative= max(abs(x['original_B_over_A']-1) for x in timing_rows),
    strongest_shared_prefix_graph_comparison=False,full_chunk_or_episode_speedup='NOT_MEASURED',
    candidate_scheduler_innovation='NOT_ESTABLISHED',cuda_graph_supported=False,
    engineering_quality_screen=False,independent_test_complete=False,
    noninferiority_established=False,formal_performance_permission=False))

table = ['| 输入 | Torch毫秒 | native baseline毫秒 | optimized毫秒 | baseline/Torch两进程范围 | optimized/Torch两进程范围 |',
         '|---|---:|---:|---:|---:|---:|']
for s in summary:
    table.append('| '+s['case']+' | '+ ' | '.join(f'{s[k]:.4f}' for k in
        ('original_event_ms_two_process_mean','baseline_event_ms_two_process_mean','optimized_event_ms_two_process_mean'))+
        ' | '+'–'.join(f'{v:.4f}' for v in s['baseline_ratio_range'])+
        ' | '+'–'.join(f'{v:.4f}' for v in s['optimized_ratio_range'])+' |')
(HERE/'PERFORMANCE-TABLE.md').write_text('\n'.join(table)+'\n')
print(json.dumps(dict(lifecycle_seconds=elapsed, standalone_blocks=direct_blocks,
    model_calls=model_calls,primitives=primitives,storage_bytes=storage,
    timing_samples=len(raw_rows),AA_max_abs_relative=max(abs(x['original_B_over_A']-1) for x in timing_rows)),ensure_ascii=False))
