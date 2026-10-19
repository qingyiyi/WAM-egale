"""Non-overlapping accounting and named mixed residuals from existing timers."""
from collections import Counter
import csv
import json
from pathlib import Path

B = Path(__file__).resolve().parent


def main():
    d = json.loads((B / 'ISOLATED-RESULT.json').read_text())
    snapshot = json.loads((B / 'ISOLATED-SNAPSHOT.json').read_text())
    grouped = {}
    episode_coverage = []
    for episode in snapshot['rows']:
        if not episode['complete']:
            continue
        observed_g = 0
        for policy in episode.get('policy_timing') or []:
            group = grouped.setdefault((episode['suite'], episode['arm'], policy['role']),
                                       dict(hist=Counter(), seconds=0.0, chunks=0))
            group['chunks'] += 1
            for r in policy['timing'].get('rounds', []):
                group['hist'][r['g']] += 1
                group['seconds'] += r['cuda_seconds']
                observed_g += r['g']
        episode_coverage.append(observed_g == episode['timing']['sum_g'])
    round_rows = []
    for (suite, arm, role), g in sorted(grouped.items()):
        n = sum(g['hist'].values())
        committed = sum(k*v for k, v in g['hist'].items())
        round_rows.append(dict(suite=suite, arm=arm, scheduled_role=role,
            policy_chunks=g['chunks'], rounds=n, sum_g=committed,
            mean_g=committed/n, sum_round_seconds=g['seconds'],
            mean_round_seconds=g['seconds']/n, seconds_per_g=g['seconds']/committed,
            g1_rounds=g['hist'][1], g2_rounds=g['hist'][2], g3_rounds=g['hist'][3],
            other_g_histogram=json.dumps({str(k):v for k,v in g['hist'].items() if k not in (1,2,3)})))
    (B/'ROUND-STATISTICS.json').write_text(json.dumps(dict(
        at_UTC=snapshot['at_UTC'], rows=round_rows,
        complete_episodes=len(episode_coverage),
        all_per_episode_g_matches_original_counter=all(episode_coverage),
        independent_samples=False, new_GPU_calls=0), indent=2)+'\n')
    with (B/'ROUND-STATISTICS.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(round_rows[0]));w.writeheader();w.writerows(round_rows)
    rows = []
    for name, suite in d['suites'].items():
        for arm, a in suite['arms'].items():
            if not a['N']:
                continue
            n, chunks = a['N'], a['policy_chunks']
            rpc = a['policy_RPC_seconds'] + a['cache_RPC_seconds'] + a['reset_RPC_seconds']
            rows.append(dict(suite=name, arm=arm, complete_episodes=n, successes=a['success'],
                episode_mean_seconds=a['episode_sec'] / n,
                policy_seconds_per_chunk=a['policy_server_seconds'] / chunks,
                action_CUDA_seconds_per_chunk=a['action_seconds'] / chunks,
                policy_minus_action_mixed_seconds_per_chunk=(a['policy_server_seconds'] - a['action_seconds']) / chunks,
                policy_RPC_minus_native_mixed_seconds_per_episode=(a['policy_RPC_seconds'] - a['policy_server_seconds']) / n,
                reset_RPC_seconds_per_episode=a['reset_RPC_seconds'] / n,
                cache_RPC_seconds_per_episode=a['cache_RPC_seconds'] / n,
                non_RPC_residual_seconds_per_episode=(a['episode_sec'] - rpc) / n,
                round_seconds_per_valid_g=a['seconds_per_valid_update']))
    with (B / 'COST-LEDGER.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    target = [r for r in rows if r['arm'] == 'T']
    chunk_bounds = [min(r['policy_seconds_per_chunk'] for r in target),
                    max(r['policy_seconds_per_chunk'] for r in target)]
    action_share = sum(r['action_CUDA_seconds_per_chunk'] for r in target) / sum(
        r['policy_seconds_per_chunk'] for r in target)
    residual_bounds = [min(r['policy_minus_action_mixed_seconds_per_chunk'] for r in rows),
                       max(r['policy_minus_action_mixed_seconds_per_chunk'] for r in rows)]
    text = [
        '# 生成与任务成本账本', '',
        f'来源{d["at_UTC"]}，{d["complete_rows"]}/360完整计时局；未结束套件仍PARTIAL。均为原观测器下工程时间，四科学权限false。', '',
        '|套件/臂|完整/成功N|暖episode均值秒|完整policy秒/chunk|action CUDA秒/chunk|非action混合残余秒/chunk|有效更新毫秒/g|',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for r in rows:
        text.append(f'|{r["suite"]}/{r["arm"]}|{r["complete_episodes"]}/{r["successes"]}|'
                    f'{r["episode_mean_seconds"]:.2f}|{r["policy_seconds_per_chunk"]:.3f}|'
                    f'{r["action_CUDA_seconds_per_chunk"]:.3f}|'
                    f'{r["policy_minus_action_mixed_seconds_per_chunk"]:.3f}|'
                    f'{r["round_seconds_per_valid_g"]*1000:.3f}|')
    text += [
        '', f'OBSERVED：当前Target完整policy约{chunk_bounds[0]:.2f}–{chunk_bounds[1]:.2f}秒/chunk，'
        f'action约占其计时{action_share:.0%}（套件均值的描述比）；连续TF有效更新成本降低，'
        f'但仍有约{residual_bounds[0]:.2f}–{residual_bounds[1]:.2f}秒/chunk的非action混合残余。'
        '这与“局部action改善大于完整policy改善”一致，不需要用target调用数替代实际延迟。具体范围随未完成数据更新，以表为准。', '',
        '这里非action残余=server full-policy host减action CUDA，包含video/条件/VAE/host提交等，不能独立归因为video或纯仿真。RPC−server还含前后审计和传输；episode−全部RPC还含环境初始化/执行与客户端处理，不能全叫env.step。reset/cache/policy原RPC分列CSV，不重复相加进更大区间。', '',
        '全体暖episode均值包含失败，双方成功速度只作补充见ISOLATED-RESULT.md；此表没有剔除超长成功或失败以改善速度。原6000完整质量另见RESULT.md，不用此固定init计时面板的成功数外推500局。', '',
        '所有臂使用原等价512-token BF16 reset bank。冷setup是phase共享成本，未按臂编造cold E2E；没有重新测无审计生产版，也没有将新whole-layer kernel接入这些TF/PF2作业。',
    ]
    text += ['', '## 实际推进与单轮成本', '',
        '|套件/臂/路由|action轮数|Σg|平均g/轮|平均轮毫秒|毫秒/g|',
        '|---|---:|---:|---:|---:|---:|']
    for r in round_rows:
        text.append(f'|{r["suite"]}/{r["arm"]}/{r["scheduled_role"]}|{r["rounds"]}|{r["sum_g"]}|'
                    f'{r["mean_g"]:.4f}|{r["mean_round_seconds"]*1000:.3f}|{r["seconds_per_g"]*1000:.3f}|')
    text += ['', '每个完整episode内重加所有原round.g，与原COUNTER.sum_g逐项核对：'+str(all(episode_coverage))+ '。g1/g2/g3直方图见ROUND-STATISTICS.csv；T均按实际单步推进，P的S/T分别统计，不把T记成draft接受100%。',
        '', '这是实际合法scheduler更新数；不是候选数、环境步或任务样本数。当前主树验证7候选，S轮通常只推进1–3步；S轮比单Target轮昂贵时，仍可能因一次推进多步降低ΣT_round/Σg。包含anchor/fallback/bonus/terminal的原计时边界保留，不能用g−1反推出未保存的accepted depth，不能将轮数当独立episode。']
    (B / 'COST-LEDGER.md').write_text('\n'.join(text) + '\n')
    print('cost rows', len(rows))


if __name__ == '__main__':
    main()
