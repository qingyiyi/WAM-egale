"""按真实融合边界输出全表；历史值明确标注，不把阶段相加当整层。"""
from pathlib import Path
import datetime
import json
import statistics

P = Path(__file__).resolve().parent
OLD = P.parent / 'rtx5090-block-port-20261023/ffn-local-handoff-20261024'
STRUCTURE = [
('布局准备','历史slot表；CTA0扫描','owner0；其余等待','8W','global索引/计数','grid1','layout完成'),
('LN1＋调制1','224×3072；224行','全grid分行','W3–6归约；8W搬运','寄存器/shared→global M1','CTA归约＋grid1','完整行归约'),
('Self QKV','3×96个64×128；K3072=9×320+192','共享288任务；owner0..169','W0 TMA；W1–3后处理；W4–7 MMA','双输入shared；acc寄存器→shared→global','grid2；每tile显式CTA5；mbarrier','输入ready/empty；结果full/empty'),
('Self Q RMS＋RoPE','112双行tile；每行3072','owner0..111','两组各4W','global Q→shared/寄存器→global','CTA归约＋grid1','完整行后旋转'),
('Self K RMS＋RoPE','112双行tile；每行3072','owner0..111','两组各4W','global K→shared/寄存器→global','CTA归约＋grid1','完整行后旋转'),
('Self Attention/私有KV寻址','短/中/长192/336/192逻辑tile；K64循环5/18/34','96/168/96双组任务','两组各4W MMA','共享历史只读；私有K/V分开；acc寄存器','FA内部同步＋grid1','有效slot/mask/候选边界'),
('Self输出投影','96个64×128；48×K64','96任务；动态owner','W0 TMA；W1–3后处理；W4–7 MMA','双输入shared；acc→shared→global','grid2；每tile显式CTA5；mbarrier','输入/结果槽ready/free'),
('残差1','336×2048元素','336静态任务','8W向量','global→寄存器→global R1','grid1','输出投影完成'),
('LN2','224×3072；224行','全grid分行','W3–6归约；8W搬运','寄存器/shared→global N2','CTA归约＋grid1','完整行'),
('Cross Q投影','96个64×128；48×K64','96任务；动态owner','W0 TMA；W1–3后处理；W4–7 MMA','双输入shared；acc→shared→global','grid2；每tile显式CTA5；mbarrier','输入/结果槽ready/free'),
('Cross Q RMS','112双行tile','owner0..111','两组各4W','global Q→shared/寄存器→global','CTA归约＋grid1','完整行'),
('Cross Attention','336逻辑tile/168双组；Q16/K512/D128','168双组任务','两组各4W MMA','缓存Cross K/V只读；acc寄存器','FA内部同步＋grid1','按相同条件组取K/V'),
('Cross输出投影','96个64×128；48×K64','96任务；动态owner','W0 TMA；W1–3后处理；W4–7 MMA','双输入shared；acc→shared→global','grid2；每tile显式CTA5；mbarrier','输入/结果槽ready/free'),
('残差2','336×2048元素','336静态任务','8W向量','global→寄存器→global R2','grid1','Cross输出投影完成'),
('LN3＋调制3','224×3072；224行','全grid分行','W3–6归约；8W搬运','寄存器/shared→global M3','CTA归约＋grid1','完整行'),
('FFN1＋GELU＋FFN2＋门控残差','FFN1 108×128×256＋12×32×256；FFN2 96×7=672任务，M64/N128/K2048','生产owner0..119；消费owner0..169','FFN1主8W/尾4W MMA；FFN2 W3–6 MMA；8W提交','ACT global；P寄存器/shared；R0..5 global；OUT global','局部ready＋tile有序提交；出口grid1','同行两组生产完成；acquire s→release s+1'),
('退出','无数学tile','全部170CTA','8W','无新增状态','grid1','所有输出/状态完成'),
]


def report(task, evaluation='当前A的源码/历史证据；无新性能测量', current=None):
    historical = json.loads((OLD / 'final-comparison/data/RESULT.json').read_text())
    manifest = json.loads((P / 'SOURCE-MANIFEST.json').read_text())
    stamp = datetime.datetime.now().astimezone().isoformat()
    lines = [f'# {task}：当前完整block阶段表', '', f'时间：{stamp}；评估变体：{evaluation}；实际保留：SELF-QKV-TMA-4MMA。',
             '设备RTX5090/SM120，GPU-e5e7821f-9633-6cc8-d06c-d0d0df4c9256；三份真实B14/Q16/block0，KV288/1152/2160；warm CrossKV；eager/no Graph。',
             'A固定SOURCE-MANIFEST.json与A-source/、独立A.so；当前历史资源170CTA×256线程、1CTA/SM、shared100352B、255reg、静态local784B；动态spill UNKNOWN。',
             '源码SHA256：'+manifest['source_files']['csrc/whole_layer.cu']['sha256']+'；二进制SHA256：'+manifest['source_files']['wam_whole_qkv_tma_cuda.cpython-312-x86_64-linux-gnu.so']['sha256']+'。', '',
             '|阶段|形状/tile|CTA任务/owner|warp|ACT/partial/acc存储|同步|ready/free/提交条件|', '|---|---|---|---|---|---|---|']
    for i, row in enumerate(STRUCTURE, 1):
        lines.append('|' + str(i)+'. '+ '|'.join(row) + '|')
    lines += ['', 'warm Cross K/V投影与K RMS：NOT_EXECUTED；层外预计算不计入以上17行，不删除其数学工作。',
              'FFN粒度：FFN1 hidden tile256，主M128/K_step32、尾M32/K_step64；ACT以四个生产组发布；FFN2数值K2048，内部K_step64，输出N128。FFN1重算1×。主循环shared与FP32 result复用同一union、不能同时保存两者；段内acc在寄存器；七段递推见NUMERIC-CONTRACT.md。', '',
              '|阶段|A嵌入整层μs：短/中/长|原Torch对应μs：短/中/长|Torch/A：短/中/长|状态/run|','|---|---:|---:|---:|---|']
    for i, row in enumerate(STRUCTURE):
        own = [r['region_medians_ms']['A'][i]*1000 for r in historical['rows']]
        fair = [r['torch_region_ms'].get(str(i)) for r in historical['rows']]
        fmt = lambda a: '/'.join(f'{x:.3f}' for x in a)
        if all(v is not None for v in fair):
            ref = fmt([x*1000 for x in fair]); ratio = '/'.join(f'{x*1000/y:.3f}×' for x,y in zip(fair,own))
        else:
            ref = ratio = 'NA'
        lines.append(f'|{i+1}. {row[0]}|{fmt(own)}|{ref}|{ratio}|HISTORICAL_ONLY；10/24 final-comparison|')
    lines += ['', '以上阶段均为旧A同轮profile区间及原Torch独立eager区域，不是本轮新测，也不能相加代替整层。新测正常完整block结果单列。',
              '', '|KV|历史原Torch/A整层|历史Guarded/A整层|本轮A/候选|本轮正确性/决定|','|---|---:|---:|---|---|',
              '|288|1.2514×|1.1127×|NOT_MEASURED|T0仅源码核对，保留A|',
              '|1152|1.7546×|1.0845×|NOT_MEASURED|T0仅源码核对，保留A|',
              '|2160|2.1902×|1.0453×|NOT_MEASURED|T0仅源码核对，保留A|',
              '', '历史正常整层来自旧E任务两进程中的冻结A，不是E性能。本轮未改准入、容差、质量权限或production_admitted=false。']
    if current:
        lines += ['', current]
    text = '\n'.join(lines)+'\n'
    (P / 'snapshots' / (task+'-STAGE-TABLE.md')).write_text(text)
    (P / 'STAGE-TABLE.md').write_text(text)
    return text


if __name__ == '__main__':
    import sys
    print(report(sys.argv[1] if len(sys.argv)>1 else 'T0'))
