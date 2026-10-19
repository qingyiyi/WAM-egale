"""Non-overlapping accounting and named mixed residuals from existing timers."""
import csv
import json
from pathlib import Path

B = Path(__file__).resolve().parent


def main():
    d = json.loads((B / 'ISOLATED-RESULT.json').read_text())
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
    (B / 'COST-LEDGER.md').write_text('\n'.join(text) + '\n')
    print('cost rows', len(rows))


if __name__ == '__main__':
    main()
