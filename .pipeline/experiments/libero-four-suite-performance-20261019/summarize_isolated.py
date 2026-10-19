"""Ratio-of-sums and failed attempts; never promote the old quality certificate."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics

BASE=Path(__file__).resolve().parent


def ratio(a,b):return a/b if b else None
def total(rows,key):
    if not rows or any(r.get(key) is None for r in rows):return None
    return sum(r[key] for r in rows)


def main():
    data=json.loads((BASE/'ISOLATED-SNAPSHOT.json').read_text())
    rows=[]
    for source in data['rows']:
        t=source.get('timing') or {};client=source.get('client_infer_timing') or {}
        r={k:source.get(k) for k in ('suite','repetition','case_id','task','init','arm','complete',
            'status','success','steps','penalty','policy_chunks','episode_sec','client_total_wall_sec',
            'component_status','async_drained','registered_GPU','source')}
        r.update(action_seconds=t.get('action_total_sec'),commits=t.get('action_g'),
            round_seconds=t.get('sum_T_round_sec'),round_g=t.get('sum_g'),
            policy_server_seconds=t.get('full_policy_elapsed_sec'),
            policy_RPC_seconds=client.get('policy_infer_wall_sec'),
            cache_RPC_seconds=client.get('cache_infer_wall_sec'),
            reset_RPC_seconds=client.get('reset_infer_wall_sec'),
            round_coverage_complete=t.get('round_coverage_complete'),
            action_measurement_complete=t.get('action_measurement_complete'))
        rows.append(r)
    result=dict(status='COMPLETE_ENGINEERING_TIMING' if data['complete_rows']==360 else 'PARTIAL_ENGINEERING_TIMING',
        at_UTC=data['at_UTC'],complete_rows=data['complete_rows'],expected_rows=360,
        quality_sample_count_increased=False,reused_init='49; three independent server-process timing repetitions',
        old_quality_permissions=dict(engineering_quality_screen=False,independent_test_complete=False,
            noninferiority_established=False,formal_performance_permission=False),
        source_registration=data['source_registration'],phases=data['phases'],queues=data.get('queues'),suites={})
    for suite in ('long','goal','spatial','object'):
        subset=[r for r in rows if r['suite']==suite and r['complete']]
        s=dict(complete=len(subset),expected=90,arms={},repetitions={},pair_metrics={})
        s['component_audit_counts']=dict(Counter(str(r['component_status']) for r in subset))
        s['round_coverage_complete']=all(r['round_coverage_complete'] is True for r in subset)
        s['action_measurement_complete']=all(r['action_measurement_complete'] is True for r in subset)
        for arm in ('T','C','P'):
            rs=[r for r in subset if r['arm']==arm]
            a=dict(N=len(rs),success=sum(r['success'] is True for r in rs),
                actual_steps=total(rs,'steps'),failure800_penalty=total(rs,'penalty'),
                policy_chunks=total(rs,'policy_chunks'),episode_sec=total(rs,'episode_sec'),
                client_total_wall_sec=total(rs,'client_total_wall_sec'),
                policy_server_seconds=total(rs,'policy_server_seconds'),
                policy_RPC_seconds=total(rs,'policy_RPC_seconds'),cache_RPC_seconds=total(rs,'cache_RPC_seconds'),
                reset_RPC_seconds=total(rs,'reset_RPC_seconds'),action_seconds=total(rs,'action_seconds'),
                commits=total(rs,'commits'),round_seconds=total(rs,'round_seconds'),round_g=total(rs,'round_g'),
                component_audit_statuses=sorted(set(str(r['component_status']) for r in rs)))
            a['component_audit_counts']=dict(Counter(str(r['component_status']) for r in rs))
            a['seconds_per_valid_update']=ratio(a['round_seconds'],a['round_g']) if a['round_seconds'] is not None else None
            a['seconds_per_policy_chunk']=ratio(a['policy_server_seconds'],a['policy_chunks']) if a['policy_server_seconds'] is not None else None
            a['success_per_total_episode_hour']=ratio(a['success']*3600,a['episode_sec']) if a['episode_sec'] is not None else None
            s['arms'][arm]=a
        indexed={(r['repetition'],r['case_id'],r['arm']):r for r in subset}
        for arm in ('C','P'):
            pairs=[(t,indexed[(rep,case,arm)]) for (rep,case,a),t in indexed.items()
                   if a=='T' and (rep,case,arm) in indexed]
            successful=[(t,v) for t,v in pairs if t['success'] and v['success']]
            pm=dict(complete_pairs=len(pairs),both_success=len(successful),
                target_success_spec_failure=sum(t['success'] and not v['success'] for t,v in pairs),
                target_failure_spec_success=sum(not t['success'] and v['success'] for t,v in pairs))
            for key in ('episode_sec','client_total_wall_sec','policy_server_seconds','policy_RPC_seconds'):
                if pairs and all(t[key] is not None and v[key] is not None for t,v in pairs):
                    pm[key+'_all_pair_speedup']=ratio(sum(t[key] for t,v in pairs),sum(v[key] for t,v in pairs))
                if successful and all(t[key] is not None and v[key] is not None for t,v in successful):
                    pm[key+'_both_success_speedup']=ratio(sum(t[key] for t,v in successful),sum(v[key] for t,v in successful))
            if pairs and all(t['round_g'] and v['round_g'] for t,v in pairs):
                target=sum(t['round_seconds'] for t,v in pairs)/sum(t['round_g'] for t,v in pairs)
                other=sum(v['round_seconds'] for t,v in pairs)/sum(v['round_g'] for t,v in pairs)
                pm['valid_update_speedup']=target/other
                pm['policy_cost_per_chunk_speedup']=(sum(t['policy_server_seconds'] for t,v in pairs)/sum(t['policy_chunks'] for t,v in pairs))/(
                    sum(v['policy_server_seconds'] for t,v in pairs)/sum(v['policy_chunks'] for t,v in pairs))
            lengths=[v['steps']/t['steps'] for t,v in successful]
            if lengths:
                # Type-7 linear interpolation, same as numpy default.
                ordered=sorted(lengths)
                def q(frac):
                    pos=frac*(len(ordered)-1);i=int(pos);return ordered[i]+(ordered[min(i+1,len(ordered)-1)]-ordered[i])*(pos-i)
                pm['both_success_step_ratio_type7']=dict(P50=q(.5),P90=q(.9),max=max(lengths))
            pm['per_complete_process_repetition']={}
            for rep in (1,2,3):
                rp=[(t,v) for t,v in pairs if t['repetition']==rep]
                if len([r for r in subset if r['repetition']==rep])!=30 or len(rp)!=10:
                    continue
                sp=[(t,v) for t,v in rp if t['success'] and v['success']]
                record=dict(complete_pairs=len(rp),both_success=len(sp),
                    episode_all_pair_speedup=ratio(sum(t['episode_sec'] for t,v in rp),
                                                  sum(v['episode_sec'] for t,v in rp)),
                    episode_both_success_speedup=ratio(sum(t['episode_sec'] for t,v in sp),
                                                      sum(v['episode_sec'] for t,v in sp)) if sp else None,
                    valid_update_speedup=(sum(t['round_seconds'] for t,v in rp)/sum(t['round_g'] for t,v in rp))/(
                        sum(v['round_seconds'] for t,v in rp)/sum(v['round_g'] for t,v in rp)),
                    policy_cost_per_chunk_speedup=(sum(t['policy_server_seconds'] for t,v in rp)/sum(t['policy_chunks'] for t,v in rp))/(
                        sum(v['policy_server_seconds'] for t,v in rp)/sum(v['policy_chunks'] for t,v in rp)))
                pm['per_complete_process_repetition'][rep]=record
            pm['timing_repetition_summary']={}
            for key in ('episode_all_pair_speedup','episode_both_success_speedup',
                        'valid_update_speedup','policy_cost_per_chunk_speedup'):
                values=[r[key] for r in pm['per_complete_process_repetition'].values() if r[key] is not None]
                if values:
                    pm['timing_repetition_summary'][key]=dict(process_repetitions=len(values),
                        arithmetic_mean=statistics.mean(values),
                        sample_SD=statistics.stdev(values) if len(values)>1 else None,
                        min=min(values),max=max(values),quality_N_not_increased=True)
            s['pair_metrics'][arm]=pm
        for rep in (1,2,3):
            sr=[r for r in subset if r['repetition']==rep]
            s['repetitions'][rep]=dict(complete=len(sr),expected=30,
                arm_episode_seconds={a:total([r for r in sr if r['arm']==a],'episode_sec') for a in ('T','C','P')})
        result['suites'][suite]=s
    (BASE/'ISOLATED-RESULT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (BASE/'ISOLATED-EPISODES.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# 四套新独占工程计时','',f'状态：{result["status"]}，{result["complete_rows"]}/360完整局；来源 {result["at_UTC"]}。',
        '', '固定10任务/init49，三次独立server进程重复。计时重复不增加独立质量N；原6000局成功率另见RESULT.md。四质量权限保持false，原FAILED/训练曝光/组件审计与长尾不改。',
        '', '| 套件 | T/C/P完整数 | T/C/P成功数 | TF暖episode | PF2暖episode | TF有效更新 | PF2有效更新 |',
        '|---|---|---|---|---|---|---|']
    for name,s in result['suites'].items():
        def fmt(arm,key):
            v=s['pair_metrics'][arm].get(key);return f'{v:.4f}×' if v is not None else 'UNKNOWN'
        text.append('|'+name+'|'+'/'.join(str(s['arms'][a]['N']) for a in ('T','C','P'))+'|'+
            '/'.join(str(s['arms'][a]['success']) for a in ('T','C','P'))+'|'+
            '|'.join([fmt('C','episode_sec_all_pair_speedup'),fmt('P','episode_sec_all_pair_speedup'),
                      fmt('C','valid_update_speedup'),fmt('P','valid_update_speedup')])+'|')
    text.extend(['','上述比率均为已完成同case、同重复的时间总和之比。未齐套件为PARTIAL，不能当完成主表；自然失败时间也在全样本分子/分母。双方成功子集、实际步数/800罚分及每次重复分列JSON和CSV。',
        '', '有效更新成本=ΣT_round/Σg。GPU0 Goal/Object、GPU1 Spatial/Long；不跨卡拼接同case。原COUNTER中AutoDL/exploratory字符串原样保留，实际硬件/隔离来源以新登记和NVML观察为准。',
        '', 'warm episode从环境构建后、reset前计到自然终态，含初始化/稳定、reset/policy/cache/环境，排除env.close与证据落盘。client_total还包含入口导入、环境构建、env.close及raw证据写入，不能当纯任务E2E。模型冷启动是每phase共享setup，另列，未测三臂分别cold端到端。详见TIME-BOUNDARY.md。',
        '', 'Object repeat1入口在任何server/episode前被旧Goal/Spatial白名单拒绝，0teacher/0局，保留且不重试。仅接续原先未启动的Object repeats2/3；原360预期保持，最大330完整，整面板按PARTIAL报告，不把缺30局计为算法失败。',
        '', '## 双方成功与固定chunk口径', '',
        '|套件|TF双方成功N / 暖episode|PF2双方成功N / 暖episode|TF完整policy/chunk|PF2完整policy/chunk|',
        '|---|---|---|---:|---:|'])
    for name,s in result['suites'].items():
        def value(arm,key):
            v=s['pair_metrics'][arm].get(key);return f'{v:.4f}×' if v is not None else 'UNKNOWN'
        text.append('|'+name+'|'+str(s['pair_metrics']['C']['both_success'])+' / '+
            value('C','episode_sec_both_success_speedup')+'|'+str(s['pair_metrics']['P']['both_success'])+' / '+
            value('P','episode_sec_both_success_speedup')+'|'+value('C','policy_cost_per_chunk_speedup')+'|'+
            value('P','policy_cost_per_chunk_speedup')+'|')
    text.extend(['', '固定chunk比率是各臂Σ完整policy时间/Σpolicy chunks之比；自然观察与历史不同，不能称同一输入chunk重放。双方成功子集只作补充，上表全体完整尝试仍保留失败时间。',
        '', '## 组件门与测量覆盖', '', '|套件|组件审计终态计数|round/action计时覆盖齐全|', '|---|---|---|'])
    for name,s in result['suites'].items():
        text.append('|'+name+'|'+', '.join(f'{k}: {v}' for k,v in s['component_audit_counts'].items())+'|'+
                    str(s['round_coverage_complete'] and s['action_measurement_complete'])+'|')
    text.extend(['', 'FAILED_FIRST_TWO_S_PAIRING/FAILED_COMPONENT_AUDIT原样保留，包含在完整尝试表中。资源巡检未观察到同卡额外模型进程不等于算法、首两S或质量门通过；本表仅为未准入候选的工程成本观察，不能签正式性能、独立TEST或闭环质量保证。'])
    text.extend(['', '## 独立server进程重复的波动', '',
        '|套件/方法|完整计时重复数|暖episode速度比均值±样本SD|完整policy/chunk速度比均值±样本SD|',
        '|---|---:|---|---|'])
    for name,s in result['suites'].items():
        for arm in ('C','P'):
            q=s['pair_metrics'][arm]['timing_repetition_summary']
            def show(key):
                if key not in q:return 'UNKNOWN'
                d=q[key];sd=f'{d["sample_SD"]:.4f}' if d['sample_SD'] is not None else 'NA'
                return f'{d["arithmetic_mean"]:.4f}±{sd}'
            n=q.get('episode_all_pair_speedup',{}).get('process_repetitions',0)
            text.append('|'+name+'/'+arm+'|'+str(n)+'|'+show('episode_all_pair_speedup')+'|'+show('policy_cost_per_chunk_speedup')+'|')
    text.extend(['', '此处是同一10case子面板的进程级计时重复，Object仅两轮；不是三轮各500局成功率，也不是新增独立初态。均值是各重复速度比的算术平均，上方主表是匹配时间总和之比，两者口径分开。三或两重复的样本SD仅描述当前波动，不冒充总体置信区间。'])
    (BASE/'ISOLATED-RESULT.md').write_text('\n'.join(text)+'\n')
    print(result['status'],result['complete_rows'])


if __name__=='__main__':main()
