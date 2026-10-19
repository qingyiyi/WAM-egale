"""Locate differences using already saved observables and same-input controls."""
import csv
import json
from pathlib import Path

B = Path(__file__).resolve().parent


def main():
    data = json.loads((B / 'remote-evidence/validate-1/RESULT.json').read_text())
    rows = []
    for r in data['rows']:
        q, f = [r['same_input_GEMM_controls'][k] for k in ('first_Q', 'FFN2')]
        m = r['intermediates']
        early = ('modulation', 'norm1', 'modulated1', 'self_q_norm',
                 'self_k_norm', 'self_v', 'self_q_rope', 'self_k_rope')
        rows.append(dict(
            case=r['case'], first_saved_DAG_difference=r['first_different_intermediate'],
            Q_control_different=q['different_elements'], Q_control_elements=q['elements'],
            Q_control_max_abs=q['max_abs'], Q_control_relative_L2=q['rel_l2'],
            FFN2_control_different=f['different_elements'], FFN2_control_elements=f['elements'],
            FFN2_control_max_abs=f['max_abs'], FFN2_control_relative_L2=f['rel_l2'],
            cross_K_bitwise=m['cross_k']['bitwise_pass'], cross_V_bitwise=m['cross_v']['bitwise_pass'],
            all_saved_self_attention_upstream_bitwise=all(m[k]['bitwise_pass'] for k in early),
            self_attention_bitwise=m['self_attention']['bitwise_pass'],
            output_relative_L2=r['original_vs_whole']['rel_l2'],
        ))
    with (B / 'PRECISION-CUTS.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    text = [
        '# 原精度失败定位：复用已保存对照，0新GPU调用', '',
        'OBSERVED：主算术源df8b386…，9个真实block签名来自既有捕获；共享输入/条件的多个历史长度不是独立统计样本。这里读取validate-1原JSON，没有重新运行teacher、完整模型或机器人。', '',
        '|输入|首个已保存DAG差异|同输入Q不同元素|同输入FFN2不同元素|cross K/V位相等|',
        '|---|---|---:|---:|---|',
    ]
    for r in rows:
        text.append(f'|{r["case"]}|{r["first_saved_DAG_difference"]}|'
                    f'{r["Q_control_different"]}/{r["Q_control_elements"]}|'
                    f'{r["FFN2_control_different"]}/{r["FFN2_control_elements"]}|'
                    f'{r["cross_K_bitwise"]}/{r["cross_V_bitwise"]}|')
    text += [
        '', 'OBSERVED：相同输入控制直接将新原型的modulated1交给原to_q、将新gelu交给原FFN2，再比较对应新输出，因此这些差异不能全部归咎于上游误差传播。B2 action的Q只有68个元素不同，B14的Q约13.3万–14.1万个；FFN2九组均不同。它们的hidden误差不能解释成物理动作容差。', '',
        'OBSERVED：三份video输入的所有已保存self-attention上游量（包含Q/K norm、V、RoPE）逐位相同，首个已保存差异在self attention。需要同时处理attention边界与GEMM契约，修好一个FFN2不足以让整层通过。九组cross K/V及其原条件输入对照通过；不能笼统宣称所有GEMM都不可能匹配。', '',
        'UNKNOWN：主审计没有保存每个原投影pre-norm输出，故“first_saved_DAG_difference”不是已经证明的全图最早根因。独立Q控制给出更早GEMM算术差异证据，但不会追溯补造缺失主路径tensor。video attention边界仍需区分原Flash归约/在线softmax顺序、有效KV顺序和可见性，不能单凭小误差指定某个浮点指令为根因。', '',
        '本机公共cuBLASLt metadata重建显示B2 Q splitK1、B2 FFN2 splitK7、B14 Q/FFN2 splitK2；这不是远端旧调用现场拦截。B2 Q在splitK1也有差异，所以“加splitK即可全部解决”没有依据。原四种累加公式已失败，不能继续无限猜公式。', '',
        'RECOMMENDATION：原baseline保持冻结可用。后续设备原语先按相同输入分别处理小M Q、FFN2、Flash attention三个数值边界，再接回整层；保留原BF16舍入、FP32打分和全部KV/RNG要求。若在固定Torch库契约下无法一致，先登记新数值与闭环验证路线，不能把本轮match-v1或普通allclose当成原质量证书。',
    ]
    (B / 'PRECISION-CUTS.md').write_text('\n'.join(text) + '\n')
    print('9 saved signatures; zero new GPU calls')


if __name__ == '__main__':
    main()
