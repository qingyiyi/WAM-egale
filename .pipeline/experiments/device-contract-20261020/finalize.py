"""Close this finite attempt from immutable receipts; no GPU code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def main():
    raw=HERE/'remote-evidence';fixed=raw/'entry-corrected-1'
    first=json.loads((raw/'STATUS.json').read_text())
    oldreg=json.loads((raw/'REGISTRATION.json').read_text())
    state=json.loads((fixed/'STATUS.json').read_text())
    metadata=json.loads((fixed/'metadata-1/RESULT.json').read_text())
    result=json.loads((fixed/'contract-1/RESULT.json').read_text())
    final=json.loads((raw/'FINAL-RESOURCE-OBSERVATION.json').read_text())
    assert len(result['rows'])==4
    assert all(r['repeat']['bitwise_pass'] and r['immutable'] and r['RNG_unchanged'] for r in result['rows'])
    assert not any(r['result']['bitwise_pass'] for r in result['rows'])
    assert not any(r['matching_active'] for r in final['owned'])
    assert not final['compute'].strip()
    jobs=first['jobs']+state['jobs']
    total_seconds=(datetime.fromisoformat(state['at_UTC'])-datetime.fromisoformat(oldreg['T0'])).total_seconds()
    summary=dict(status='COMPLETE_BOUNDED_DEVICE_CONTRACT_NO_GO',
        initial_entry_implementation_failure_preserved=True,
        metadata_geometry_match_count=4,primitive_contracts_passed=0,primitive_contracts_evaluated=4,
        repeat_input_RNG_pass_count=4,full_block_calls=metadata['full_block_calls']+result['full_block_calls'],
        primitive_calls=result['primitive_calls'],public_heuristic_queries=metadata['heuristic_queries'],
        full30=0,robot_episodes=0,production_admitted=False,asynchronous_whole_layer_integrated=False,
        numerical_gate_covers_attention=False,strong_baseline_performance_measured=False,
        total_reserved_lifecycle_seconds=total_seconds,CUDA_active_seconds='UNKNOWN_NOT_MEASURED',
        sum_child_lifecycle_seconds=sum(j['lifecycle_seconds'] for j in jobs),
        sum_GPU_worker_lifecycle_seconds=sum(j['lifecycle_seconds'] for j in jobs if j['name']!='build-1'),
        all_child_roots_reaped=all(j['root_reaped'] for j in jobs),
        parent_exit_codes='UNKNOWN_NOT_REAPED_BY_LOCAL_SSH; observed no active owned PID',
        owned_matching_active=0,final_GPU_observation=final['gpu'],
        storage_bytes=final['storage_bytes'],lifecycle_budget_seconds=2700,
        unused_budget_seconds_not_reusable=2700-total_seconds,
        engineering_quality_screen=False,independent_test_complete=False,
        noninferiority_established=False,formal_performance_permission=False)
    (HERE/'RESULT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    table=[]
    for r in result['rows']:
        d=r['result'];table.append(f"|B{r['batch']} {r['name']}|{r['M']}×{r['N']}×{r['K']}|{d['different_elements']}/{d['elements']}|{d['max_abs']:.6g}|{100*d['rel_l2']:.5f}%|FAIL|")
    (HERE/'RESULT.md').write_text('''# M37设备数值契约：四项NO-GO，有界收口

OBSERVED：当前远端公共cuBLASLt描述符重建确认4个Q/FFN2的tile、stage、split-K与预先候选依据一致；不是原Torch内部调用拦截。实现使用固定CUTLASS v3.5.1（f7b19de32c5d1f3cedfc735c2849f12b537522ee）的真实SM80 multistage mainloop、epilogue与serial split-K semaphore，不调用host BLAS计算候选，不重复旧四种仅改变workspace累加公式。该组件单独编译成功，不是整层或warp职责流水实现。

|同输入原语|逻辑M×N×K|不同元素|最大绝对差|相对L2|原逐位门|
|---|---|---|---|---|---|
'''+ '\n'.join(table)+f'''

四项候选重复调用自身逐位一致，输入/参数与CPU/CUDA RNG保持；四项原Torch比较全部失败。Q/FFN2金标准直接使用原Torch Linear，输入来自已冻结顺序native的同一次debug，而不是失败WMMA的上游输出。没有把隐藏激活误差当成物理动作误差。

宏几何：B2 Q逻辑tile64×64、K64/stage5/split1；B2 FFN2 tile64×256、K64/stage3/split7；B14二者tile128×128、K64/stage3/split2。候选warp shape由实现选择，不能说已读取vendor warp shape；B14 Q vendor CTA swizzle=1，候选采用CUTLASS identity swizzle。公共metadata不揭示Tensor Core accumulator映射、K归约/舍入细节和全部epilogue。不能把宏几何门当作全实现一致，更不能由此单独指定失败根因。

HYPOTHESIS：设备累加、split归约和epilogue顺序中仍存在未匹配差异。当前结果只排除了“匹配这些公共宏参数即可自动逐位”的方案，不证明任何特定算子或误差是机器人失败根因。

UNKNOWN：vendor内部算术顺序，attention重新嵌入8warp执行器后的完整数值契约，新组件的正式性能。没有反汇编闭源库、修改Torch matmul设置、降低原门或以小误差发PASS。

RECOMMENDATION：这一几何匹配路线止损，停止异步整层接入；保留已对齐baseline和上次整层未准入原型。后继若要采用新算术，需要独立登记数值及闭环重验；若保持原逐位契约，必须先有可验证的实际设备实现依据，而非继续盲猜split-K。原Torch host包装调用cuBLASLt，并不提供其内部设备代码，复制host调用链不能解决单kernel内复用问题。

预算实际：4个参考block、12原语调用（4原Torch、8候选含重复）、8公共heuristic、0full30/机器人局。总预留生命周期{total_seconds:.3f}秒（包含首次入口失败、人工修正等待、CPU编译和清理），CUDA active时间UNKNOWN；不是把全部生命周期记成CUDA时间。全部child root已真实reap，最终owned活动0、两卡0MiB/0%。[资源](RESOURCE-AND-STOP-REPORT.md)。

最初metadata-1因冻结baseline DSO中的本机绝对依赖而在第一个block前失败，0block/0heuristic。原始失败、原return0但status FAILED、清理与耗时均保留；新entry-corrected-1只替换本轮debug入口为既有远端sequential数学链，不改其源码/DSO，不重置T0或截止。[实施记录](ENTRY-IMPLEMENTATION-ISSUE.md)。

源码：[contract_linear.cu](../../../算子研究/wam_ops/optimized/csrc/contract_linear.cu)、[独立构建](../../../算子研究/wam_ops/optimized/build_contract.py)、[同输入核验](validate_contract.py)。原始结果：[4原语](remote-evidence/entry-corrected-1/contract-1/RESULT.json)、[8公共描述符](remote-evidence/entry-corrected-1/metadata-1/RESULT.json)、[原失败](remote-evidence/metadata-1/RESULT.json)、[最终资源](remote-evidence/FINAL-RESOURCE-OBSERVATION.json)。

旧FAILED/NO_GO/四false、M34/M35原终态、默认A_off不变；baseline/投机/原tile与保护文件不变。没有新性能或质量放行。
''')
    (HERE/'RESOURCE-AND-STOP-REPORT.md').write_text(f'''# M37资源与止损

- 原T0：{oldreg['T0']}；工作截止{oldreg['work_stop']}，清理截止{oldreg['cleanup_deadline']}，没有重置。
- 数值终态：{state['at_UTC']}，完整预留生命周期{total_seconds:.3f}秒；45分钟上限内收口。剩余额度不作为后续自动授权。
- 全child生命周期和{summary['sum_child_lifecycle_seconds']:.3f}秒；GPU worker生命周期和{summary['sum_GPU_worker_lifecycle_seconds']:.3f}秒（含模型读入/CPU工作，非CUDA active）。CPU编译57.644秒，2核。不得把有重叠边界相加成计费小时。
- 原始实施失败1次，后续修正1次；0 timeout，无完整episode自动重试。4参考block、12 primitive API调用、8 heuristic。
- 最终GPU0/1无compute context；全部登记PID+出生时间观测无活动owned。child实际return/reap见原receipt；父进程退出码未知，只记已观测不活动，不补exit0。
- 新namespace存储{final['storage_bytes']}字节，另有workspace唯一DSO，见最终观察；新增总量远小于512MiB。模型与fixture未复制。
- 仅新optimized组件；本机88项保护核对及远端86已部署项核对，结果分别见FROZEN-LOCAL.json/FROZEN-REMOTE.json。原远端缺state/rules的88项检查false不改。
- 四原语全部未通过原逐位门；不做异步整层集成/性能扩测/参数网格。attention契约仍未完成。没有新训练、完整机器人任务、TEST、第二环境、30层或租卡。
''')
    terminal=dict(summary,at_UTC=datetime.now(timezone.utc).isoformat())
    (HERE/'TERMINAL.json').write_text(json.dumps(terminal,ensure_ascii=False,indent=2)+'\n')
    p=HERE/'PLAN.md';s=p.read_text()
    for tag in ('C2','C3','C4','C5'):s=s.replace(f'- [ ] {tag}：',f'- [x] {tag}：',1)
    p.write_text(s+'\n实际终态：四原语NO-GO，数值门失败，异步整层集成未触发；有界工作完成，不代表科学准入。见RESULT.md。\n')
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [HERE/'validate_contract.py',HERE/'inspect_gemm_remote.py',HERE/'run_corrected_entry.py',
                       ROOT/'算子研究/wam_ops/optimized/csrc/contract_linear.cu',
                       ROOT/'算子研究/wam_ops/optimized/build_contract.py']}
    (HERE/'SOURCE-MANIFEST-FINAL.json').write_text(json.dumps(sources,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k in ('status','full_block_calls','primitive_calls',
        'total_reserved_lifecycle_seconds','owned_matching_active','storage_bytes')},ensure_ascii=False))

if __name__=='__main__':main()
