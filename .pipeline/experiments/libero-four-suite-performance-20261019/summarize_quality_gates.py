"""State the old engineering gates without promoting complete computation."""
import json
from pathlib import Path

B = Path(__file__).resolve().parent


def main():
    source = json.loads((B / 'RESULT.json').read_text())
    text = [
        '# 四套完整质量结果与原登记门', '',
        'OBSERVED：6000完整episode，四套各10任务×50初态×T/C/P。表中成功是原环境成功判据，不是脚本exit0或“无FAILED”。所有自然失败保留，实际命令数与失败800罚分分列。原始逐局来源、分任务成功和成本见RESULT.json及EPISODES.csv。', '',
        '|套件|T成功/500|TF成功/500|PF2成功/500|TF−T百分点|PF2−T百分点|',
        '|---|---:|---:|---:|---:|---:|',
    ]
    gates = {}
    for name, s in source['SUITES'].items():
        counts = [s['by_arm'][a]['successes'] for a in ('T', 'C', 'P')]
        text.append(f'|{name}|{counts[0]}|{counts[1]}|{counts[2]}|'
                    f'{100*s["comparisons"]["C"]["SR_difference"]:+.1f}|'
                    f'{100*s["comparisons"]["P"]["SR_difference"]:+.1f}|')
    text += ['', '|套件/方法|T胜/候选败|候选胜/T败|双方成功N|长度P50/P90/max|>1.30局数|',
             '|---|---:|---:|---:|---|---:|']
    for name, s in source['SUITES'].items():
        gates[name] = {}
        for arm in ('C', 'P'):
            c = s['comparisons'][arm]
            q = c['successful_length_ratio']
            gates[name][arm] = dict(
                success_at_least_target=s['by_arm'][arm]['successes'] >= s['by_arm']['T']['successes'],
                success_at_least_matched_MLP='UNKNOWN: no matched 500-episode MLP arm',
                P50_at_most_1p10=q['p50'] <= 1.10,
                P90_at_most_1p20=q['p90'] <= 1.20,
                max_at_most_1p30=q['max'] <= 1.30,
                strict_first_two_S_and_component_certificate='FAILED; old audit retained',
                no_new_registered_risk_failure='UNKNOWN: descriptive full-suite panel does not replace original risk-pair audit; historical failures retained',
                independent_TEST='NOT_ESTABLISHED',
                noninferiority='NOT_ESTABLISHED',
                overall_original_quality_gate=False,
            )
            text.append(f'|{name}/{arm}|{c["lost_success"]}|{c["rescued"]}|{c["both_success"]}|'
                        f'{q["p50"]:.4f}/{q["p90"]:.4f}/{q["max"]:.4f}|'
                        f'{c["successful_length_ratio_over1p30"]}|')
    text += [
        '', '长度是双方成功子集的补充统计，分位数为type7线性插值；不能删掉失败后只报长度。八个套件/方法组合P50/P90均满足旧工程数值，但max均超过1.30，因此没有一组完整满足原长度门。全体actual steps/失败罚分的均值、总数与逐局值见RESULT.md/json；actual796没有补造为800命令。', '',
        '|套件/方法|成功数≥T|成功数≥匹配MLP|P50≤1.10|P90≤1.20|max≤1.30|整体门|',
        '|---|---|---|---|---|---|---|',
    ]
    for name, arms in gates.items():
        for arm, g in arms.items():
            text.append(f'|{name}/{arm}|{g["success_at_least_target"]}|UNKNOWN|'
                        f'{g["P50_at_most_1p10"]}|{g["P90_at_most_1p20"]}|'
                        f'{g["max_at_most_1p30"]}|False|')
    text += [
        '', 'UNKNOWN：这不是独立TEST。数据暴露、选择与首两S严格配对/组件FAILED保留；当前不能据名义配对计数签出独立成功率差区间或3百分点非劣结论。没有全套匹配MLP臂，也没有把历史小DEV的MLP数拼入本表。“无新增登记风险失败”未由这个描述性面板重新签证，历史风险失败不能被其新成功覆盖。重复init49的计时轮次不增加独立质量N。', '',
        'OBSERVED：PF2不是普遍改善。Long比Target少10次成功，Spatial少8次；Goal/Object观察成功数更高，但仍有长度长尾和旧审计限制。连续TF的Long/Goal/Object观察成功数不低于Target，Spatial低6次。不同套的训练身份保留原manifest，不把它们当同一权重跨套泛化。', '',
        'RECOMMENDATION：当前保留TF64为系统研究结构，权重/接受/调度的质量证书未收口。不要默认打开PF2，不因这些开发结果新增训练或扩大TEST。后续若获准，只针对完整日志里的少数慢成功长尾和首两S数值/状态隔离定位风险，先固定数据隔离与协议，再决定是否值得质量修订。', '',
        '`engineering_quality_screen=false`、`independent_test_complete=false`、`noninferiority_established=false`、`formal_performance_permission=false`。旧FAILED/retention/NO_GO/paused72/default A_off不改。',
    ]
    (B / 'QUALITY-GATES.json').write_text(json.dumps(gates, indent=2) + '\n')
    (B / 'QUALITY-RESULT.md').write_text('\n'.join(text) + '\n')
    print('6000 terminals; eight overall gates remain false')


if __name__ == '__main__':
    main()
