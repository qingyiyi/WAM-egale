"""CPU-only numerical reporting and six paired R/S verifier comparison."""
import math
import statistics as S


def aggregate_strict(equivalence_strict, off, finish):
    return bool(equivalence_strict and off['strict_bitwise_all']
                and finish['strict_all_S_numerical_checks'])


def numerical(actual, gold):
    import torch
    if actual.device.type != 'cpu' or gold.device.type != 'cpu' or actual.shape != gold.shape:
        raise ValueError('numerical reporting must be outside timer on matching CPU tensors')
    x = actual.double(); y = gold.double(); d = x-y
    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
    if not finite:
        return dict(bitwise=False, finite=False, max_abs_error=None, relative_RMS_error=None)
    den = max(float(y.square().mean().sqrt()), 1e-12)
    return dict(bitwise=(actual.dtype==gold.dtype and torch.equal(
                    actual.contiguous().view(torch.uint8),gold.contiguous().view(torch.uint8))), finite=True,
                max_abs_error=float(d.abs().max()),
                RMS_error=float(d.square().mean().sqrt()),
                relative_RMS_error=float(d.square().mean().sqrt())/den,
                changed_elements=int((actual != gold).sum()), total_elements=actual.numel(),
                dtype=str(actual.dtype), shape=list(actual.shape))


def off_analysis(rows, *, diagnostic_allowed=False):
    groups={'R':{},'S':{}}
    for row in rows:
        if (row['status']!='PASS' or not row['profile_OFF'] or row['query_count']!=29
                or not row['cold_setup_staging_keys_release_included'] or row['arm'] not in groups
                or not math.isfinite(row['wall_sec']) or row['wall_sec']<=0):
            raise ValueError('invalid OFF row')
        if not row['bitwise_all_outputs'] and (row['arm']!='S' or not diagnostic_allowed):
            raise ValueError('unapproved nonbitwise OFF row')
        if row['block'] in groups[row['arm']]: raise ValueError('duplicate block')
        groups[row['arm']][row['block']]=row
    metrics={}
    for arm,g in groups.items():
        if sorted(g)!=list(range(6)): raise ValueError('all six paired blocks required')
        raw=[g[b]['wall_sec'] for b in range(6)]; med=S.median(raw)
        metrics[arm]=dict(raw_sec=raw,median_sec=med,
            MAD_over_median=S.median(abs(v-med) for v in raw)/med,
            CUDA_event_ms=[g[b]['CUDA_event_ms'] for b in range(6)],
            max_extra_peak_bytes=max(g[b]['extra_peak_bytes'] for b in range(6)))
    stable=all(v['MAD_over_median']<=.07 for v in metrics.values())
    strict=all(v['bitwise_all_outputs'] for v in rows)
    ratios=[groups['S'][b]['wall_sec']/groups['R'][b]['wall_sec'] for b in range(6)]
    ratio=metrics['S']['median_sec']/metrics['R']['median_sec']
    signal=('positive_pilot' if ratio<=.95 and all(r<1 for r in ratios)
            else 'negative_pilot' if ratio>=1.05 and all(r>1 for r in ratios) else 'flat_or_mixed')
    return dict(status='FAIL' if not stable else 'PASS' if strict else 'DIAGNOSTIC_ONLY_NOT_ACCEPTED',
                scope='single-worker full30 changing29 verifier replay, not full policy/episode',
                metrics=metrics,ratio_S_over_R=ratio,speedup_R_over_S=1/ratio,
                reduction_percent=100*(1-ratio),paired_ratios=ratios,
                signal=signal if stable else 'unstable',strict_bitwise_all=strict,
                success_environment_steps_E2E_proven=False,no_samples_removed=True,
                setup_staging_keys_journal_release_included=True,stability_MAD_cap=.07)
