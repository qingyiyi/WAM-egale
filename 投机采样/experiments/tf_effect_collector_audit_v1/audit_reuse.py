"""New, explicit complete recorded baseline reuse qualification contract.

The old OFF/ON-per-request-KV contract remains unchanged and PARTIAL. This
module checks one full ON replay against the original complete M11 recording,
per-chunk recorded RNG, and RESET/final complete valid KV. It does not certify
all intermediate request KV or manufacture a second new OFF replay.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import signal
import sys
import textwrap
import time
import traceback

OWN=Path(__file__).resolve().parent;EXP=OWN.parent
if str(EXP) not in sys.path:sys.path.insert(0,str(EXP))
from tf_effect_collector_audit_v1 import audit as original

ROOT,RT,COMMIT,PERMISSIONS=original.ROOT,original.RT,original.COMMIT,original.PERMISSIONS
read,write,sha,stat,require=original.read,original.write,original.sha,original.stat,original.require
SCHEMA='tf-effect-collector-baseline-reuse-cpu-audit/v1'
PASS='PASS_FULL_50_CHUNK_QUALIFICATION_AUDIT_BASELINE_REUSED'
GPU_PASS='PASS_FULL_RECORDED_EPISODE_COLLECTOR_EQUIVALENCE_BASELINE_REUSED'
FIRST=RT/'p0-tf-effect-collector-qualification-v1-20261016-r1'
SECOND=RT/'p0-tf-effect-collector-qualification-v1-20261016-r2'
RECEIPTS=('plan.json','PREREGISTRATION.json','WORKER-RESULT.json','SUPERVISOR-RESULT.json',
    'FINAL-OWNED-CLEANUP.json','CLOCK-GUARD.json','CLOCK-RESULT.json','RUN-STARTED.json','ON-REQUEST-PROOF.json')
SCOPE=('one complete ON replay versus original complete recorded OFF episode; '
       'all50 policy outputs/teacher/proposals/routes and per-chunk RNG; '
       'RESET and final complete valid KV/RNG; no new OFF replay and no all-request KV claim')


def row_auditor():
    """Reuse the existing full-row math AST verbatim, excluding its old gates.

    This is an extractor, not a source edit or weakened numerical checker.
    Only loop-independent terminal gates and result wording are new here.
    """
    tree=ast.parse(textwrap.dedent(inspect.getsource(original.run)))
    fn=tree.body[0];body=next(n for n in fn.body if isinstance(n,ast.Try)).body
    starts=[i for i,n in enumerate(body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='total_trees' for t in n.targets)]
    ends=[i for i,n in enumerate(body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)
        and any(isinstance(a,ast.Constant) and a.value=='full frozen task4 current-TF query counts differ' for a in n.value.args)]
    require(len(starts)==len(ends)==1 and starts[0]<ends[0], 'reviewed full-row audit AST boundaries changed')
    nodes=body[starts[0]:ends[0]+1]
    require(sum(isinstance(n,ast.For) for n in nodes)==1,'one original complete50 outer loop required')
    selected_dump=ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False)
    arguments=ast.arguments(posonlyargs=[],args=[ast.arg(arg=k) for k in ('p','job','q','counter','torch')],
        vararg=None,kwonlyargs=[],kw_defaults=[],kwarg=None,defaults=[])
    result=ast.Return(value=ast.Dict(keys=[ast.Constant(value=k) for k in
        ('total_trees','total_rows','total_reached','max_norm_error','cpu_depth_flips','cpu_edge_flips','chunk_reports')],
        values=[ast.Name(id=k,ctx=ast.Load()) for k in
        ('total_trees','total_rows','total_reached','max_norm_error','cpu_depth_flips','cpu_edge_flips','chunk_reports')]))
    wrapper=ast.FunctionDef(name='_original_full50_rows',args=arguments,body=nodes+[result],decorator_list=[])
    module=ast.Module(body=[wrapper],type_ignores=[]);ast.fix_missing_locations(module)
    require(ast.dump(ast.Module(body=wrapper.body[:-1],type_ignores=[]),include_attributes=False)==selected_dump,
        'extracted row checks must remain exactly the original AST')
    namespace=dict(vars(original));exec(compile(module,str(OWN/'audit.py')+':verbatim-full50-row-audit','exec'),namespace)
    import hashlib
    proof=dict(source=str(Path(original.__file__).resolve()),source_sha256=sha(original.__file__),
        extracted_checks_ast_sha256=hashlib.sha256(selected_dump.encode()).hexdigest(),
        all_original_row_checks_AST_unchanged=True,old_terminal_and_reporting_contract_not_reused=True)
    return namespace['_original_full50_rows'],proof


def prepare(args):
    original.runtime();job=args.job.resolve();qualification=args.qualification.resolve(strict=True)
    require(job.parent==RT and not job.exists(),'fresh CPU audit namespace required')
    require(300<=args.wall_cap_seconds<=1800 and 16<=args.storage_cap_mib<=128,'bounded CPU audit cap required')
    job.mkdir();birth=time.time()
    reg=dict(schema=SCHEMA,source_commit=COMMIT,qualification=str(qualification),birth_epoch=birth,
        deadline_epoch=birth+args.wall_cap_seconds,wall_cap_seconds=args.wall_cap_seconds,
        storage_cap_bytes=args.storage_cap_mib*2**20,cpu_threads=2,GPU=False,permissions=PERMISSIONS,
        source_sha256={str(Path(__file__).resolve()):sha(__file__),str(Path(original.__file__).resolve()):sha(original.__file__)},
        validation_scope=SCOPE,cleanup_plan='single pinned CPU process, alarm plus caller external timeout; no workers/models/CUDA',
        numerical_norm_atol=1e-6,evidence_path=str(job),terminal_status='REGISTERED_NOT_PREPARED')
    write(job/'PREREGISTRATION.json',reg)
    try:
        original.alarm(reg)
        require(all((qualification/n).is_file() for n in RECEIPTS),'new reuse qualification is not terminal; no incomplete-job audit')
        q=read(qualification/'plan.json');require(q['commit']==COMMIT and q['permissions']==PERMISSIONS,'fixed base/permissions differ')
        require(read(qualification/'WORKER-RESULT.json')['status']==GPU_PASS,'new GPU reuse contract not passed')
        ep=Path(q['source_episode']).resolve(strict=True)
        traces=sorted((ep/'direction_trace').glob('chunk-*.pt'));teachers=sorted((qualification/'teacher').rglob('chunk-*.pt'))
        require(len(traces)==len(teachers)==50,'complete50 original and ON shards required')
        evidence=[qualification/n for n in RECEIPTS]+[ep/n for n in ('COUNTER.json','RESET.json','FINAL-SERVER-STATE.json')]
        for old in (FIRST,SECOND):
            evidence += [old/n for n in ('plan.json','FAILED.json','FINAL-OWNED-CLEANUP.json')]
        require(all(f.is_file() for f in evidence),'all original baselines and failed/PARTIAL predecessors must remain')
        _,proof=row_auditor()
        p=dict(**reg,source_episode=str(ep),traces=[str(f) for f in traces],teachers=[str(f) for f in teachers],
            evidence_stat={str(f):stat(f) for f in evidence+traces+teachers},
            receipt_sha256={str(f):sha(f) for f in evidence},row_check_source_proof=proof,
            preregistration_sha256=sha(job/'PREREGISTRATION.json'))
        write(job/'PLAN.json',p);write(job/'PREPARED.json',dict(status='PREPARED_NEW_BASELINE_REUSE_CONTRACT_CPU_ONLY',
            validation_scope=SCOPE,source_checks=proof,GPU=False))
        return p
    except BaseException as exc:
        write(job/'FAILED.json',dict(status='FAILED_NEW_REUSE_AUDIT_PREPARATION',error=repr(exc),traceback=traceback.format_exc(),permissions=PERMISSIONS))
        raise
    finally:signal.alarm(0)


def terminal_proof(qualification,q,counter,ep):
    w=read(qualification/'WORKER-RESULT.json');s=read(qualification/'SUPERVISOR-RESULT.json')
    c=read(qualification/'CLOCK-RESULT.json');g=read(qualification/'CLOCK-GUARD.json')
    cleanup=read(qualification/'FINAL-OWNED-CLEANUP.json')
    require(w['status']==GPU_PASS and w['permissions']==PERMISSIONS and w['baseline_reused'] is True
        and w['extra_teacher_forwards']==0 and w['new_robot_episodes']==0 and w['actual_o3_preserved'] is True,
        'explicit new complete ON/original-baseline GPU contract required')
    for name in ('all_original_actions_video_commands_exact','per_policy_original_RNG_routes_counts_exact',
                 'chunk_RNG_exact','reset_and_final_full_valid_KV_RNG_exact'):
        require(w[name] is True,'missing new required equivalence gate: '+name)
    passes=w['passes'];require(len(passes)==1 and passes[0]['collector']=='ON'
        and passes[0]['policy_chunks']==50 and passes[0]['target_forwards']==counter['target_forwards']
        and passes[0]['accepted_trees']==len(counter['acceptance'])
        and passes[0]['original_final_valid_KV_RNG_exact'] is True
        and passes[0]['original_per_policy_RNG_exact'] is True and passes[0]['OFF_baseline_reused'] is True,
        'exact one complete50 ON pass required')
    require(w['final_all_valid_KV_RNG_exact'] is True and w['per_request_all_KV_comparison'] is False
        and q['OFF_baseline_reused'] is True and q['per_request_KV_comparison'] is False,
        'new scope must explicitly avoid all-intermediate-request KV claim')
    require(w['target_accounting']['target_intents']==counter['target_forwards']
        and w['target_accounting']['failed']==w['target_accounting']['active']==0,'no added model forward or unfinished target request allowed')
    require(s['status']=='COMPLETE' and s['failure'] is None and s['permissions']==PERMISSIONS
        and s['cleanup']==cleanup and cleanup['all_roots_reaped'] is True,'new supervisor/owned cleanup not complete')
    require(c['status']=='CLEAN_OBSERVED_SUPERVISOR_EXIT_NOT_QUALITY_PASS'
        and c['all_observed_owned_live_processes_exited'] is True and c['error'] is None
        and c['external_stop_reason'] is None and c['no_foreign_signals'] is True,'new independent clock not clean')
    require(g['clock']['plan_sha256']==sha(qualification/'plan.json') and g['no_budget_reset'] is True,'new guard not bound to actual original budget')
    first=read(FIRST/'plan.json');old=[]
    for path in (FIRST,SECOND):
        p=read(path/'plan.json');failure=read(path/'FAILED.json');receipt=read(path/'FINAL-OWNED-CLEANUP.json')
        require(receipt['all_roots_reaped'] is True and p['absolute_cleanup_deadline']==first['absolute_cleanup_deadline'],
            'predecessor cleanup or cumulative absolute deadline differs')
        old.append(dict(job=str(path),status=failure['status'],failure_receipt_retained=True))
    require(q['absolute_cleanup_deadline']==first['absolute_cleanup_deadline']
        and q['absolute_work_stop']==first['absolute_work_stop'],'all attempts must retain original 45min absolute deadlines')
    require(c['clock']['process_birth_monotonic']+c['elapsed_quality_process_seconds']<=c['clock']['hard_stop_monotonic']+1,
        'original cumulative absolute deadline exceeded')
    for name in ('FAILED.json','WORKER-FAILED.json','CLOCK-FAILED.json','CLOCK-STOP.json'):
        require(not (qualification/name).exists(),'new reuse qualification has a failure/stop receipt')
    receipts=read(qualification/'ON-REQUEST-PROOF.json')
    require(len(receipts)==len(counter['requests']),'complete original request chronology required')
    for index,(item,source) in enumerate(zip(receipts,counter['requests'])):
        require(item['index']==index and item['kind']==source['kind'] and item['forwards']==source['target_forwards'],
            'full recorded request target count mismatch')
        require(item['all_valid30layer_KV_RNG_exact']==(index in (0,len(receipts)-1)),
            'only RESET/final request full-KV proofs belong to the new scope')
    pins=list(s.get('observed_pins',[]))+[p['pin'] for p in c['process_observations']]+[g['supervisor_pin'],g['guard_pin']]
    unique={(p['pid'],p['start_ticks'],p['uid']):p for p in pins}
    observations=[original.live_pin(p) for p in unique.values()]
    require(observations and not any(p['live'] for p in observations),'new qualification has a live pinned owned process')
    worker=EXP/'tf_effect_closure_reuse_v3/replay_worker.py'
    require(q['source_sha256'][str(worker)]==sha(worker)
        and any(str(worker) in p.get('argv',[]) for p in pins), 'actual executed reuse worker source/provenance differs')
    require((ep/'RESET.json').is_file() and (ep/'FINAL-SERVER-STATE.json').is_file(),'original full RESET/final state evidence required')
    return dict(status='PASS_NEW_BASELINE_REUSE_TERMINAL_CONTRACT',validation_scope=SCOPE,
        recorded_full_episode_baseline_reused=True,new_ON_passes=1,new_OFF_passes=0,
        all_intermediate_request_KV_equivalence_established=False,per_chunk_RNG_exact=True,
        reset_and_final_full_valid_KV_RNG_exact=True,predecessors_retained=old,
        total_replay_target_forwards=w['target_accounting']['target_intents'],new_teacher_forwards=0,
        process_observations=observations,clock_is_not_robot_quality_certificate=True)


def run(job):
    original.runtime();job=job.resolve(strict=True);p=read(job/'PLAN.json');reg=read(job/'PREREGISTRATION.json')
    require(p['schema']==SCHEMA and p['permissions']==PERMISSIONS and p['source_commit']==COMMIT,'new CPU audit identity differs')
    require(not (job/'STARTED.json').exists() and not (job/'FAILED.json').exists(),'one attempt; no overwrite or retry')
    require(sha(job/'PREREGISTRATION.json')==p['preregistration_sha256'],'registration changed')
    for f,h in p['source_sha256'].items():require(sha(f)==h,'new or reused audit source changed')
    start=time.time();write(job/'STARTED.json',dict(status='RUNNING_CPU_ONLY_NEW_REUSE_CONTRACT',pid=os.getpid()))
    try:
        original.alarm(reg)
        for f,s in p['evidence_stat'].items():require(stat(f)==s,'frozen source artifact changed')
        for f,h in p['receipt_sha256'].items():require(sha(f)==h,'frozen source receipt changed')
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1);require(not torch.cuda.is_initialized(),'no CUDA context allowed')
        qualification=Path(p['qualification']);q=read(qualification/'plan.json');ep=Path(p['source_episode'])
        counter=read(ep/'COUNTER.json');terminal=terminal_proof(qualification,q,counter,ep)
        check,proof=row_auditor();require(proof==p['row_check_source_proof'],'verbatim row check proof changed')
        values=check(p,job,q,counter,torch)
        for f,s in p['evidence_stat'].items():require(stat(f)==s,'source evidence changed during audit')
        result=dict(schema=SCHEMA,status=PASS,source_commit=COMMIT,qualification=str(qualification),source_episode=str(ep),
            validation_scope=SCOPE,complete_chunks=50,actual_teacher_trees=values['total_trees'],actual_teacher_rows=values['total_rows'],
            actually_reached_rows=values['total_reached'],root_rows=values['total_trees'],counterfactual_parent_rows=values['total_rows']-values['total_reached'],
            all_state_previous_target_proposal_bytes_exact=True,all_sigma_time_step_route_cost_flags_exact=True,counter_coverage_checked=True,
            all_original_teacher_proposal_routes_exact=True,
            teacher_cpu_norm_max_abs=values['max_norm_error'],cpu_norm_atol=p['numerical_norm_atol'],
            cpu_norm_bitwise_GPU_equivalence_claimed=False,cpu_norm_uses_separate_FP32_operands_before_subtraction=True,
            cpu_reduction_depth_flips=values['cpu_depth_flips'],cpu_reduction_edge_gate_flips=values['cpu_edge_flips'],
            deployment_acceptance_uses_original_GPU_costs=True,collector_equivalence_passed=True,
            baseline_reused=True,new_ON_passes=1,new_OFF_passes=0,new_policy_chunk_replays=50,
            all_intermediate_request_KV_equivalence_established=False,per_chunk_RNG_exact=True,
            reset_and_final_full_valid_KV_RNG_exact=True,old_two_new_pass_allrequestKV_contract_passed=False,
            terminal_proof=terminal,row_check_source_proof=proof,chunk_reports=values['chunk_reports'],
            data_role='previously_unblinded_DEV_not_new_FIT_CAL',new_robot_episodes=0,new_teacher_forwards=0,
            training_steps=0,GPU=False,CUDA_initialized=False,permissions=PERMISSIONS,
            elapsed_run_sec=time.time()-start,elapsed_registered_wall_sec=time.time()-p['birth_epoch'],
            evidence=[dict(path=f,sha256=h) for f,h in p['receipt_sha256'].items()],performance_claim=False,
            source_evidence_modified=False)
        write(job/'RESULT.json',result)
        with (job/'RESULT.md').open('x') as handle:
            handle.write('# 使用完整原记录基线的 O3 teacher CPU 验收\n\n'
                f'终态 `{PASS}`。只新重放 ON 完整50个policy chunk，并复用原完整M11记录作OFF基线。'
                '全2712行teacher/proposal/实际step/route/六cost/flags已按旧逐行数学完整验收；CPU范数容差1e-6，'
                f'实测最大差{values["max_norm_error"]:.9g}，不声称CPU/GPU reduction逐位相同。\n\n'
                'GPU新合同只保证完整原动作/video/路径、每chunk RNG以及RESET与最终全有效KV/RNG相等；'
                '没有新增第二遍OFF，也没有每个中间请求的全KV等价结论。旧r1失败/r2 PARTIAL保留，原45分钟累计截止不重置。'
                '该资格不放行机器人质量、TEST、非劣或正式性能；四旧权限保持false。\n')
        original.storage(job,p['storage_cap_bytes']);return result
    except BaseException as exc:
        write(job/'FAILED.json',dict(status='FAILED_NEW_REUSE_CPU_AUDIT',error=repr(exc),traceback=traceback.format_exc(),permissions=PERMISSIONS));raise
    finally:
        signal.alarm(0)
        try:
            module=sys.modules.get('torch');write(job/'CPU-CLEANUP.json',dict(status='NO_CHILDREN_MODEL_OR_GPU_CREATED',
                pid=os.getpid(),CUDA_initialized=False if module is None else module.cuda.is_initialized(),no_foreign_signals=True))
        except Exception as exc:print('REUSE_CPU_AUDIT_CLEANUP_MARKER_ERROR',repr(exc),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--prepare',action='store_true');group.add_argument('--run',action='store_true')
    parser.add_argument('--qualification',type=Path);parser.add_argument('--wall-cap-seconds',type=int,default=900)
    parser.add_argument('--storage-cap-mib',type=int,default=64);args=parser.parse_args()
    if args.prepare:
        if args.qualification is None:parser.error('--qualification required for prepare')
        p=prepare(args);print(dict(status='PREPARED_NEW_REUSE_CONTRACT_CPU_ONLY',job=str(args.job.resolve()),deadline_epoch=p['deadline_epoch']))
    else:
        r=run(args.job);print(dict(status=r['status'],trees=r['actual_teacher_trees'],rows=r['actual_teacher_rows']))


if __name__=='__main__':main()
