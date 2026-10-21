"""Read-only process inspection and derived resource report, no process control."""
from pathlib import Path
import datetime
import json
import time

B = Path(__file__).resolve().parent
now = time.time()
remote = json.loads((B/'BUDGET-CHECKPOINT.json').read_text())
local = []
for path in sorted(B.glob('local-ncu*/STATUS.json')):
    s = json.loads(path.read_text())
    identity = s.get('owned_guard', {})
    alive = False
    if identity:
        try:
            fields = Path(f"/proc/{identity['pid']}/stat").read_text().rsplit(')', 1)[1].split()
            alive = int(fields[19]) == identity['birth_ticks'] and fields[0] != 'Z'
        except FileNotFoundError:
            pass
    complete = s['status'] in ('COMPLETE', 'FAILED_OR_PARTIAL', 'EXPIRED_WITHOUT_GPU_WINDOW')
    elapsed = s.get('job_lifecycle_charged_seconds', s.get('lifecycle_seconds', (s.get('end', now)-s['start'])))
    local.append(dict(namespace=path.parent.name, status=s['status'], terminal=complete,
                      charged_seconds=elapsed, GPU_worker_seconds=s.get('GPU_lifecycle_seconds', 0),
                      guard_identity=identity, guard_alive=alive,
                      identity_alive=s.get('identity_alive'), failure=s.get('error', s.get('foreign_processes'))))
night = remote['GPU_lifecycle_seconds'] + sum(x['charged_seconds'] for x in local)
prior = 2538.679523
data = dict(observed_at=datetime.datetime.now().astimezone().isoformat(),
    status='CLOSED' if all(x['terminal'] and not x['guard_alive'] for x in local) else 'NCU_WINDOW_PENDING',
    deadline='2026-10-22T09:00:00+08:00', prior_M39_GPU_lifecycle_seconds=prior,
    remote_GPU_lifecycle_seconds=remote['GPU_lifecycle_seconds'],
    remote_CPU_build_prepare_seconds=remote['CPU_prepare_seconds'],
    remote_measured_at=remote['timestamp'], remote_owned_remaining=remote['owned_remaining'],
    local_attempts=local, local_worker_GPU_seconds=sum(x['GPU_worker_seconds'] for x in local),
    night_charged_seconds_including_local_wait=night, cumulative_charged_seconds=prior+night,
    cumulative_cap_seconds=36000, night_cap_seconds=30000,
    remaining_seconds=min(36000-prior-night,30000-night),
    CUDA_active_seconds='UNKNOWN: not measured separately; lifecycle is not CUDA active time',
    CPU_analysis_seconds='UNMETERED; build/prepare values are not total CPU work',
    local_stored_bytes=sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),
    remote_stored_bytes_at_checkpoint=remote['stored_bytes'], storage_cap_bytes=3*1024**3,
    stopped_implementation_candidates=['01 generic Self-Q','05 wide FFN1 K64','07 one-warp RMS','09 native-like small projection split','12 two-warp RMS'],
    precision_failure_preserved='extended-state-08 short/drop1; repaired by08b, not changed to PASS',
    failed_or_partial_remote=[x for x in remote['jobs'] if x['status']!='COMPLETE_ATTEMPT'],
    ready_DAG_started=False, new_robot_experiments_started=False,
    engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
assert night <= 30000 and prior+night <= 36000
(B/'RESOURCE-AND-STOP-REPORT.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
lines=['# 资源与停止记录', '', f"更新时间：{data['observed_at']}；状态：{data['status']}。", '',
 f"远端GPU阶段生命周期 {remote['GPU_lifecycle_seconds']:.6f} 秒；CPU构建/准备 {remote['CPU_prepare_seconds']:.6f} 秒（不是全部CPU分析时间）。远端owned在checkpoint均退出。", '',
 f"本轮保守计费 {night:.6f} 秒，包括本机自然空闲排队；原M39累计 {prior+night:.6f}/36000秒，剩余 {data['remaining_seconds']:.6f}秒，09:00硬截止更早优先。排队计入预算但不表示占用了GPU。CUDA active时间未测。", '',
 '|本机作业|状态|含等待生命周期秒|GPU worker秒|guard仍存活|','|---|---|---:|---:|---|']
lines += [f"|{x['namespace']}|{x['status']}|{x['charged_seconds']:.3f}|{x['GPU_worker_seconds']:.3f}|{x['guard_alive']}|" for x in local]
lines += ['', '12个实现候选及08b正确性修复已终态；仅11及其前序有效改动进入最终版本。较慢候选、数值失败、测量包装器错误、NCU外部占用中断均留存；失败和等待计预算。未启动ready-DAG、30层融合、新机器人任务或扩大质量测试。', '',
 '资源失败清单见JSON；初始Self-Q/point计时包装错误、两个native元数据wrapper错误、最终stage格式抽取CPU失败分别保留独立终态。脚本returncode0不覆盖guard对外部干扰的FAILED_OR_PARTIAL。', '',
 f"当前本地证据 {data['local_stored_bytes']/1024**2:.2f}MiB，远端checkpoint {data['remote_stored_bytes_at_checkpoint']/1024**2:.2f}MiB；总新增上限3GiB。模型/fixture没有逐候选复制。", '',
 '四项科学权限false；关闭本轮作业不等于M39性能完全对齐，不等于RoboTwin扩大获准。', '']
(B/'RESOURCE-AND-STOP-REPORT.md').write_text('\n'.join(lines))
print(json.dumps({k:data[k] for k in ('status','cumulative_charged_seconds','remaining_seconds','local_stored_bytes')},ensure_ascii=False))
