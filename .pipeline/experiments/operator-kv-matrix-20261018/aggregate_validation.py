"""Aggregate all actual operator/block and full-query validation evidence, CPU only."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import csv
import json
import os

ROOT=Path('/rjs/cxz/论文探索')
OUT=Path(__file__).resolve().parent
RT=ROOT/'投机采样/results/runtime'


def load(p):return json.loads(p.read_text())


def csvwrite(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def main():
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    vjob=RT/'p0-operator-kv-validation-shared-v1-20261018-r8'
    ijob=RT/'p0-operator-saved-input-io-shared-v1-20261018-r9'
    v=load(vjob/'process1/RESULT.json');io=load(ijob/'RESULT.json')
    perf=load(OUT/'PERFORMANCE-ONLY-SUMMARY.json')
    numeric=[];full=[];saved=[];issues=[]
    def require(ok,why):
        if not ok:issues.append(why)
    require(v['status']=='COMPLETE_54_POINT_AND_FULL30_QUERY_VALIDATION' and len(v['points'])==54,'full54-point correctness coverage')
    require(len(v['complete_model_queries'])==6,'six full30 model queries')
    for index,p in enumerate(v['points']):
        identity={k:p[k] for k in ('length','batch','kind','query_role','query_index','layer')}
        require(p['valid_KV_preserved'] and p['valid_metadata_preserved'],str(identity)+'validKV')
        for part,s in [('self',p['attention']['self']),('cross',p['attention']['cross']),('block',p['block'])]:
            for engine,a in s['accuracy'].items():
                numeric.append(dict(**identity,part=part,engine=engine,**a,
                    valid_KV_preserved=p['valid_KV_preserved'],valid_metadata_preserved=p['valid_metadata_preserved'],
                    source_pointer=f'{vjob}/process1/RESULT.json#/points/{index}/{part}'))
                require(a['finite'],str(identity)+'nonfinite '+part+'/'+engine)
                if engine=='torch':require(a['byte_equal'],str(identity)+'Torch reference replay')
    for index,q in enumerate(v['complete_model_queries']):
        for engine,e in q['engines'].items():
            for kind in ('raw','CFG','active_action_CFG'):
                full.append(dict(length=q['length'],query=q['query'],engine=engine,output_kind=kind,**e[kind],
                    all30_valid_KV_and_metadata_unchanged=e['all30_valid_KV_and_metadata_unchanged'],
                    inputs_unchanged=e['inputs_unchanged'],RNG_unchanged=e['RNG_unchanged'],
                    strict_bitwise_input_output_state_pass=e['strict_bitwise_input_output_state_pass'],
                    source_pointer=f'{vjob}/process1/RESULT.json#/complete_model_queries/{index}/engines/{engine}'))
            require(e['all30_valid_KV_and_metadata_unchanged'] and e['inputs_unchanged'] and e['RNG_unchanged'],str(q['length'])+engine+'state/input/RNG change')
    for p in io['rows']:
        for engine,e in p['engines'].items():
            saved.append(dict(identity=str(p['identity']),engine=engine,**e['output'],
                input_bytes_and_descriptors_unchanged=e['all_passed_input_bytes_and_descriptors_unchanged'],
                shape_dtype_device_correct=e['shape_dtype_device_correct'],RNG_unchanged=p['RNG_unchanged'],source=p['source']))
            require(e['all_passed_input_bytes_and_descriptors_unchanged'] and e['shape_dtype_device_correct'] and p['RNG_unchanged'],str(p['identity'])+engine+'saved-I/O contract')
    require(io['status']=='COMPLETE_NINE_REAL_SAVED_INPUT_IO_VALIDATION' and len(io['rows'])==9,'nine saved performance input replay')
    require(load(vjob/'TERMINAL.json')['owned_children_exited'],'model validation cleanup')
    require(load(ijob/'TERMINAL.json')['owned_child_exited'],'saved input cleanup')
    groups=[]
    for part in ('self','cross','block'):
        for batch in (2,14):
            subset=[r for r in numeric if r['part']==part and r['batch']==batch]
            for engine in sorted({r['engine'] for r in subset}):
                selected=[r for r in subset if r['engine']==engine]
                groups.append(dict(part=part,batch=batch,engine=engine,n=len(selected),
                    allclose02_passed=sum(r['allclose_02'] for r in selected),byte_equal=sum(r['byte_equal'] for r in selected),
                    max_abs=max(r['max_abs'] for r in selected),max_rel_l2=max(r['rel_l2'] for r in selected)))
    model_groups=[]
    for engine in ('torch','standalone','shared','guarded'):
        rows=[r for r in full if r['engine']==engine and r['output_kind']=='active_action_CFG']
        model_groups.append(dict(engine=engine,n=len(rows),allclose02_passed=sum(r['allclose_02'] for r in rows),
            byte_equal=sum(r['byte_equal'] for r in rows),max_abs=max(r['max_abs'] for r in rows),
            max_rel_l2=max(r['rel_l2'] for r in rows),strict_contract_passed=sum(r['strict_bitwise_input_output_state_pass'] for r in rows)))
    r8=load(vjob/'REGISTRATION.json');t8=load(vjob/'TERMINAL.json');t9=load(ijob/'TERMINAL.json')
    spent=r8['prior_lifecycle_seconds']+t8['wall_seconds']+t9['wall_seconds']
    result=dict(at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),
        status='COMPLETE_EVIDENCE_COLLECTION_WITH_STANDALONE_FULL30_FAILURE',CPU_audit_issues=issues,
        performance_points=perf['points'],performance_cells=perf['performance_cells'],performance_event_samples=perf['event_samples'],
        correctness_points=len(v['points']),numeric_comparisons=len(numeric),full_model_queries=6,full_model_engine_runs=24,
        saved_input_cases=9,saved_input_engine_comparisons=len(saved),numeric_groups=groups,full_model_groups=model_groups,
        all54_valid_KV_and_metadata_preserved=all(p['valid_KV_preserved'] and p['valid_metadata_preserved'] for p in v['points']),
        GPU_lifecycle_spent_seconds=spent,original_GPU_lifecycle_cap_seconds=1800,unused_lifecycle_seconds=1800-spent,
        CUDA_active_seconds='UNKNOWN',new_IO_validation_was_shared_and_not_timed=True,
        performance_resource_status='EXPLORATORY_NVML_UTILIZATION_COUNTER_UNQUALIFIED',
        integration_permission=False,engineering_quality_screen=False,independent_test_complete=False,
        noninferiority_established=False,formal_performance_permission=False,
        source_roots=[str(vjob),str(ijob),perf['job']])
    csvwrite('PRECISION.csv',numeric);csvwrite('FULL30-IO.csv',full);csvwrite('SAVED-INPUT-IO.csv',saved);csvwrite('PRECISION-GROUPED.csv',groups)
    (OUT/'SUMMARY-COMBINED.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    lines=['# 原始与verifier输入：性能、精度和完整输入输出验证','',f'更新：{result["at"]}。数据采集完整，standalone完整30层出现真实数值失败，按原记录保留。','',
        '结论：现有standalone SDPA在54个真实单attention/block点满足原.02绝对加.02相对容差，但多数并非逐位一致；全30层同时替换self/cross后，六例均不逐位一致，后段三例连该容差也未通过。shared-prefix与guarded路径在六个完整B14 query的原始/CFG输出及有效KV、输入字节、RNG检查上逐位通过。','',
        '这支持把Torch/shared-prefix/guarded继续用作这组真实verifier的参考；不能把standalone现有有限单block证书提升为严格同算法完整Transformer证书。原tile源码、原报告与失败未改。','',
        '## 覆盖与计时边界','',
        '- 新性能：54固定点，三KV×B2 video/B2 action/B14 action×first/last×层0/8/29；684性能单元、18360 Event及等量同步wall样本，单进程3轮。精度在用户允许共享GPU后独立完成，不混入性能计时。',
        '- B2 action与B14 action实际K288/1152/2160，B2 video实际K272/1136/2144。H24/D128/BF16；B14仍为7个独立Q16、两CFG历史组、各自private KV。',
        '- 54点共414次输出比较和54次有效KV/metadata检查；六个完整30层B14 query×四引擎共24次完整模型执行；另对性能阶段真实保存的九输入执行27次直接算子输入/输出/布局/device/RNG检查。',
        '- 性能阶段独占窗口内未见其他可见GPU进程，但NVML利用率空context仍非零，原因未确认；延迟继续标探索性。三轮/单进程也不足证明约1%的微小差异稳定。','',
        '性能全覆盖也不等于稳定对齐：standalone self/cross Graph-core相对Torch的54点比例中位数分别.9812/.9817；self无慢点，cross一例慢约.52%。完整block有14/54慢点，比例范围.2465–3.0713，中位数.9665；保留极端波动，尚未定位原因。guarded相对强shared完整block中位数.9960、7/18更慢，也不能由最好长KV点宣称普遍新增收益。详见PERFORMANCE-COVERAGE-CAUTIONS.json。','',
        '## 单attention与真实block精度','',
        '| 部分 | batch | 实现 | .02通过/比较数 | 逐字节相同 | max_abs最大值 | rel_L2最大值 |','|---|---|---|---|---|---|---|']
    for g in groups:
        lines.append(f'| {g["part"]} | {g["batch"]} | {g["engine"]} | {g["allclose02_passed"]}/{g["n"]} | {g["byte_equal"]}/{g["n"]} | {g["max_abs"]:.8g} | {g["max_rel_l2"]:.8g} |')
    lines+=['','原容差逐元素为`abs(actual-ref)<=0.02+0.02*abs(ref)`。不是统一2%误差，max_abs大于.02仍可能通过相对项；完整block B2最坏max_abs=.25，不隐藏。','',
        '## 完整30层：实际有效action通道的CFG方向','',
        '| 实现 | .02通过/6 | 逐字节一致/6 | max_abs | rel_L2最大值 | 输入/KV/RNG严格门通过/6 |','|---|---|---|---|---|---|']
    for g in model_groups:
        lines.append(f'| {g["engine"]} | {g["allclose02_passed"]}/6 | {g["byte_equal"]}/6 | {g["max_abs"]:.8g} | {g["max_rel_l2"]:.8g} | {g["strict_contract_passed"]}/6 |')
    lines+=['','standalone六例逐项（CFG有效action）：','', '| KV | query | max_abs | rel_L2 | .02容差 |','|---|---|---|---|---|']
    for r in full:
        if r['engine']=='standalone' and r['output_kind']=='active_action_CFG':
            lines.append(f'| {r["length"]} | {r["query"]} | {r["max_abs"]:.8g} | {r["rel_l2"]:.8g} | {r["allclose_02"]} |')
    lines+=['','此处standalone同时替换self与cross；尚未做self-only/cross-only消融，不能把完整输出差异唯一归因到某一部分或某个浮点指令。观察到了累计输出偏差，但未识别完整数值根因。','',
        '## 保存输入的直接I/O检查','',
        f'九组固定layer0/first真实性能输入，共{len(saved)}实现比较；实际通过输入字节、shape/stride/dtype/device/data_ptr不变、输出shape/dtype/device及RNG检查。无效torch.empty KV槽未比较。原输入370763331B。这里只覆盖九个保存代表输入，不伪称54输入全部逐元素保存。','',
        '## 可写与不可外推','',
        '- 可写：固定真实范围内的表中性能与误差；shared/guarded六个完整B14 query严格输入输出/有效状态通过；standalone完整query数值失败。',
        '- 不可写：任意输入严格等价、全部54形状稳定加速、完整30层B2/video已验证、接受决策完全不变、完整scheduler/chunk/decoded action/机器人闭环通过。',
        '- 六query来自既有真实A_off verifier输入，不是当前TF的新完整机器人质量验证；没有SR/trace/E2E新结论。四项false与原NO_GO保留。',
        '- 原长重复双进程方案按用户新窗口收尾31点PARTIAL，未补造第二进程；新性能协议单独登记，不据最好轮选结果。','',
        '## 完整数据与源码','',
        '[性能与阶段耗时](PERFORMANCE-ONLY-REPORT.md) · [所有684性能单元](PERFORMANCE-ONLY.csv) · [414精度单元](PRECISION.csv) · [完整30层](FULL30-IO.csv) · [保存输入I/O](SAVED-INPUT-IO.csv) · [机器摘要](SUMMARY-COMBINED.json)','',
        f'原始GPU结果：`{vjob}/process1/RESULT.json`，`{ijob}/RESULT.json`。完整模型输出张量全部保存在前一目录；误差统计由真实GPU记录，本CPU聚合不冒充独立重算未保存的所有逐元素输出。','',
        '执行入口：`投机采样/experiments/operator_kv_matrix_v1/`；旧tile实现与其HW只读。新fixture压缩仅去重已CPU逐字节验证的七候选重复组，未对数值做量化或更改。','',
        '实现名：standalone为原tile `_parity.sdpa`（完整block/30层同时替self与cross）；shared为既有`_shared_torch_v1.shared`（仅替self、cross仍Torch）；G1为candidate grouped(...,1)，只测self kernel；guarded为候选几何guard下group1/2（仅替self）。G1没有本次独立完整block/30层证书。','',
        '另有[24份完整输出CPU独立核对](CPU-SAVED-OUTPUT-AUDIT.json)：原始字节一致标志、max_abs、.02容差标志与GPU记录一致；rel_L2仅允许CPU/GPU归约舍入的2e-6相对审计误差。未在CPU执行WAM或初始化CUDA。','',
        f'累计登记生命周期{spent:.3f}/1800秒，未用{1800-spent:.3f}秒；CUDA active UNKNOWN。不自动追加融合、训练或性能实验。CPU聚合问题：{issues}。']
    (OUT/'COMBINED-RESULT.md').write_text('\n'.join(lines)+'\n')
    (OUT/'CURRENT-STATUS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','numeric_comparisons','full_model_groups','GPU_lifecycle_spent_seconds','unused_lifecycle_seconds','CPU_audit_issues')},ensure_ascii=False))


if __name__=='__main__':main()
