"""Standard-library-only, mutually exclusive attribution of immutable Kineto traces.

GPU kernel activities are the accounting units. Nested inclusive CPU/device
operator totals are never added. Links use Chrome flows, runtime correlation,
and External id; missing/conflicting evidence stays unassigned.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics

MODULES={'norm1','attn1','norm2','attn2','norm3','ffn'}
POINTWISE={'add','add_','mul','mul_','sub','sub_','div','div_','pow','sqrt','rsqrt','neg','gelu','silu','relu','tanh'}
CAST_COPY={'to','_to_copy','copy_','contiguous','clone'}

def canonical(x):
    return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)

def categories(e):
    return {s.strip().lower() for s in str(e.get('cat','')).split(',') if s.strip()}

def is_kernel(e):
    return e.get('ph')=='X' and 'kernel' in categories(e)

def thread(e):
    return canonical([e.get('pid'),e.get('tid')])

def arg(e,name):
    wanted=name.lower().replace('_','').replace(' ','')
    for k,v in e.get('args',{}).items():
        if str(k).lower().replace('_','').replace(' ','')==wanted:
            return v
    return None

def identifier(value):
    return None if value is None or value in (-1,0,'-1','0','') else str(value)

def valid_interval(e):
    return (e.get('ph')=='X' and isinstance(e.get('ts'),(int,float))
            and isinstance(e.get('dur'),(int,float)) and math.isfinite(e['ts'])
            and math.isfinite(e['dur']) and e['dur']>=0)

class Intervals:
    """Centered interval tree for read-only timestamp containment queries."""
    def __init__(self,events):
        self.empty=not events
        if self.empty:return
        self.center=statistics.median(e['ts'] for e in events)
        left=[];right=[];middle=[]
        for e in events:
            if e['ts']+e['dur']<self.center:left.append(e)
            elif e['ts']>self.center:right.append(e)
            else:middle.append(e)
        self.starts=sorted(middle,key=lambda e:e['ts'])
        self.ends=sorted(middle,key=lambda e:e['ts']+e['dur'],reverse=True)
        self.left=Intervals(left) if left else None
        self.right=Intervals(right) if right else None
    def at(self,t):
        if self.empty:return []
        result=[]
        if t<=self.center:
            for e in self.starts:
                if e['ts']>t:break
                result.append(e)
            if self.left:result.extend(self.left.at(t))
        else:
            for e in self.ends:
                if e['ts']+e['dur']<t:break
                result.append(e)
            if self.right:result.extend(self.right.at(t))
        return result

def interval_indexes(events):
    grouped=defaultdict(list)
    for e in events:grouped[thread(e)].append(e)
    return {k:Intervals(v) for k,v in grouped.items()}

def activity_key(e):
    # Correlation alone or kernel name alone is NOT unique. Retain separate
    # occurrences on different streams/devices/timestamps or external ids.
    return canonical([e.get('name'),e.get('pid'),e.get('tid'),e['ts'],e['dur'],
        arg(e,'device'),arg(e,'context'),arg(e,'stream'),arg(e,'correlation'),arg(e,'External id')])

def flow_key(e):
    if 'id' in e:return canonical([e.get('cat'),e.get('scope'),e['id']])
    if 'id2' in e:
        ids=e['id2']
        return canonical([e.get('cat'),e.get('scope'),ids,e.get('pid') if 'local' in ids else None])
    return None

def op_group(name):
    if not name or not name.startswith('aten::'):return 'unknown_operator'
    op=name.split('::',1)[1].split('.',1)[0]
    if op in POINTWISE:return 'pointwise'
    if op in CAST_COPY:return 'cast_copy'
    if 'norm' in op:return 'normalization'
    if op in {'mm','bmm','addmm','baddbmm','linear','matmul'}:return 'matmul'
    if 'attention' in op:return 'attention'
    return 'other_aten'

def union_duration(events):
    # Chrome timestamps can be ~1e12 us; adding tiny durations in float loses
    # precision and may make a nonoverlap union exceed the activity sum.
    by_device=defaultdict(list)
    for e in events:
        lo=Decimal(str(e['ts']));hi=lo+Decimal(str(e['dur']))
        by_device[canonical([arg(e,'device'),e.get('pid')])].append((lo,hi))
    total=Decimal(0)
    for spans in by_device.values():
        end=None
        for lo,hi in sorted(spans):
            if end is None or lo>end:total+=hi-lo;end=hi
            elif hi>end:total+=hi-end;end=hi
    return float(total)

def analyze_document(document):
    events=document['traceEvents'] if isinstance(document,dict) else document
    if not isinstance(events,list):raise ValueError('traceEvents must be a list')
    kernels=[];seen={};duplicates=[]
    for i,e in enumerate(events):
        if not is_kernel(e):continue
        if not valid_interval(e):raise ValueError('nonfinite/invalid GPU kernel interval')
        key=activity_key(e)
        if key in seen:
            duplicates.append(dict(first_trace_index=seen[key],duplicate_trace_index=i))
        else:
            seen[key]=i;kernels.append(dict(e,_trace_index=i,_kernel_index=len(kernels)))
    if not kernels:raise ValueError('trace has no CUDA kernel activities')
    def cpu_event(e):
        cats=categories(e)
        return valid_interval(e) and not (cats & {'kernel','gpu_memcpy','gpu_memset'}) and not any(c.startswith('gpu_') for c in cats)
    cpu=[e for e in events if cpu_event(e)]
    markers=[e for e in cpu if str(e.get('name','')).startswith(('mv/candidate/','mv/block/'))]
    cpu_index=interval_indexes(cpu);marker_index=interval_indexes(markers)
    gpu_index=interval_indexes(kernels)
    ext=defaultdict(list);correlations=defaultdict(list)
    for e in cpu:
        ex=identifier(arg(e,'External id'));co=identifier(arg(e,'correlation'))
        if ex is not None:ext[ex].append(e)
        if co is not None:correlations[co].append(e)
    def cpu_point(pid_tid,t):
        ix=cpu_index.get(pid_tid);return ix.at(t) if ix else []
    def scope_at(pid_tid,t,bounds=None):
        ix=marker_index.get(pid_tid);ms=ix.at(t) if ix else []
        if bounds is not None:
            lo,hi=bounds
            ms=[e for e in ms if e['ts']<=lo and e['ts']+e['dur']>=hi]
        candidates=[e for e in ms if str(e['name']).startswith('mv/candidate/')]
        if not candidates:return None
        # Candidate names are expected unique per file. Refuse crossing scopes.
        if len({e['name'] for e in candidates})!=1:return None
        block=[]
        for e in ms:
            m=re.fullmatch(r'mv/block/(\d+)/(block|norm1|attn1|norm2|attn2|norm3|ffn)',str(e['name']))
            if m:block.append((int(m[1]),m[2],e))
        ids={b[0] for b in block}
        if len(ids)>1:return None
        if not block:return ('candidate',)
        b=next(iter(ids));modules={v[1] for v in block if v[1]!='block'}
        if len(modules)>1:return None
        return ('candidate',f'block{b:02d}',next(iter(modules))) if modules else ('candidate',f'block{b:02d}')
    def source_info(e,method,t=None,pid_tid=None):
        point_source=t is not None
        t=e['ts']+e.get('dur',0)/2 if t is None else t
        pid_tid=thread(e) if pid_tid is None else pid_tid
        bounds=None if point_source else (e['ts'],e['ts']+e.get('dur',0))
        # An External id can identify an outer operator spanning children.
        # Only containing ancestors apply; midpoint children are not evidence.
        path=scope_at(pid_tid,t,bounds)
        ops=[x for x in cpu_point(pid_tid,t) if str(x.get('name','')).startswith('aten::')]
        if bounds is not None:
            lo,hi=bounds
            ops=[x for x in ops if x['ts']<=lo and x['ts']+x['dur']>=hi]
        if str(e.get('name','')).startswith('aten::'):ops.append(e)
        ops.sort(key=lambda x:(x['dur'],-x['ts'],x['name']))
        return dict(method=method,path=path,operator=ops[0]['name'] if ops else None,
                    source_name=e.get('name'),source_thread=pid_tid,source_ts=t)
    flow_groups=defaultdict(list)
    for e in events:
        if e.get('ph') in {'s','t','f'} and flow_key(e) is not None:flow_groups[flow_key(e)].append(e)
    flow_sources=defaultdict(list);flow_stats=defaultdict(int)
    for group in flow_groups.values():
        starts=[e for e in group if e['ph']=='s'];finishes=[e for e in group if e['ph']=='f']
        if len(starts)!=1:flow_stats['nonunique_or_missing_start_groups']+=1;continue
        start=starts[0]
        if not isinstance(start.get('ts'),(int,float)) or not math.isfinite(start['ts']):continue
        for finish in finishes:
            t=finish.get('ts');ix=gpu_index.get(thread(finish))
            if ix is None or not isinstance(t,(int,float)) or not math.isfinite(t):continue
            matches=ix.at(t)
            exact=[e for e in matches if e['ts']==t]
            if exact:matches=exact
            if len(matches)!=1:
                flow_stats['ambiguous_or_unmatched_gpu_finish']+=1;continue
            info=source_info(start,'flow',t=start['ts'],pid_tid=thread(start))
            flow_sources[matches[0]['_kernel_index']].append(info)
            flow_stats['linked_gpu_finishes']+=1
    attributed=[]
    for k in kernels:
        evidence=list(flow_sources[k['_kernel_index']])
        co=identifier(arg(k,'correlation'));ex=identifier(arg(k,'External id'))
        for e in correlations.get(co,[]):evidence.append(source_info(e,'correlation'))
        for e in ext.get(ex,[]):evidence.append(source_info(e,'external_id'))
        paths=[tuple(v['path']) for v in evidence if v['path'] is not None]
        chosen=None;reason=None
        if not paths:reason='no_cpu_link_inside_candidate'
        else:
            chosen=max(paths,key=len)
            if any(chosen[:len(path)]!=path for path in paths):chosen=None;reason='conflicting_cpu_scope_links'
        compatible=[v for v in evidence if v['path'] is not None and chosen is not None and chosen[:len(v['path'])]==tuple(v['path'])]
        # Prefer most local evidence with an actual aten source. Different op
        # names are not silently resolved when equivalent provenance disagrees.
        local=[v for v in compatible if tuple(v['path'])==chosen and v['operator']]
        operators=sorted({v['operator'] for v in local})
        operator=operators[0] if len(operators)==1 else None
        group=op_group(operator)
        if chosen is None:scope='unassigned';bucket='unassigned'
        elif len(chosen)==1:scope='candidate/outside_blocks';bucket=scope+'/'+group
        elif len(chosen)==2:scope=chosen[1]+'/outside_modules';bucket=scope+'/'+group
        else:scope=chosen[1]+'/'+chosen[2];bucket=scope
        attributed.append(dict(kernel_index=k['_kernel_index'],trace_index=k['_trace_index'],
            name=k.get('name'),ts_us=k['ts'],duration_us=k['dur'],device=arg(k,'device'),
            stream=arg(k,'stream'),pid=k.get('pid'),tid=k.get('tid'),correlation=co,external_id=ex,
            scope=scope,bucket=bucket,operator=operator,operator_group=group,
            operator_candidates=operators,unassigned_reason=reason,
            link_methods=sorted({v['method'] for v in compatible}),evidence=evidence))
    groups=defaultdict(lambda:dict(kernel_count=0,duration_sum_us=0.,kernel_names=defaultdict(lambda:dict(count=0,duration_sum_us=0.))))
    for r in attributed:
        g=groups[r['bucket']];g['kernel_count']+=1;g['duration_sum_us']+=r['duration_us']
        n=g['kernel_names'][str(r['name'])];n['count']+=1;n['duration_sum_us']+=r['duration_us']
    total=sum(e['dur'] for e in kernels);unassigned=[r for r in attributed if r['scope']=='unassigned']
    unattributed_time=sum(r['duration_us'] for r in unassigned)
    assert sum(g['kernel_count'] for g in groups.values())==len(kernels)
    assert math.isclose(sum(g['duration_sum_us'] for g in groups.values()),total,rel_tol=1e-12,abs_tol=1e-6)
    for g in groups.values():g['fraction_of_unique_kernel_duration_sum']=g['duration_sum_us']/total if total else 0.
    module_totals=defaultdict(lambda:dict(kernel_count=0,duration_sum_us=0.))
    marker_categories=defaultdict(lambda:defaultdict(int))
    for e in events:
        if e.get('ph')=='X' and str(e.get('name','')).startswith(('mv/candidate/','mv/block/')):
            marker_categories[str(e.get('cat',''))][str(e['name'])]+=1
    for r in attributed:
        key=r['scope'].split('/')[-1] if r['scope']!='unassigned' else 'unassigned'
        if key=='outside_modules':key+=':'+r['operator_group']
        g=module_totals[key];g['kernel_count']+=1;g['duration_sum_us']+=r['duration_us']
    return dict(schema='mutually-exclusive-kernel-attribution/v1',
        accounting_unit='unique GPU kernel activity, not nested operator inclusive totals',
        profile_is_not_performance_benchmark=True,kernel_duration_sum_is_not_wall_time=True,
        raw_kernel_events=len(kernels)+len(duplicates),unique_kernel_count=len(kernels),
        duplicate_activity_count=len(duplicates),duplicate_activities=duplicates,
        unique_kernel_duration_sum_us=total,device_union_kernel_busy_us=union_duration(kernels),
        union_semantics='union of observed kernel intervals per device, then sum across devices; not complete request wall time',
        unassigned_kernel_count=len(unassigned),unassigned_kernel_fraction=len(unassigned)/len(kernels),
        unassigned_duration_us=unattributed_time,unassigned_duration_fraction=unattributed_time/total if total else 0.,
        scope_link_methods=['Chrome flow','runtime correlation','External id'],
        conflicts_fail_closed=True,flow_diagnostics=dict(flow_stats),
        module_totals=dict(module_totals),buckets=dict(groups),kernels=attributed,
        marker_categories=dict(marker_categories),
        cuda_copy_set_events_excluded=sum(e.get('ph')=='X' and bool(categories(e)&{'gpu_memcpy','gpu_memset'}) for e in events),
        limitations=['Captured first/last CFG14 queries only; no all-step or task-quality claim.',
                    'Kernel duration sum is work accounting; stream overlap and CPU launch gaps require separate timeline analysis.',
                    'Missing/ambiguous CPU associations are retained as unassigned; operator family is unknown when aten evidence disagrees.'])

def immutable_json(path,expected_sha=None):
    p=Path(path);before=p.stat();raw=p.read_bytes();after=p.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise ValueError('file changed while reading: '+str(p))
    sha=hashlib.sha256(raw).hexdigest()
    if expected_sha is not None and sha!=expected_sha:raise ValueError('published trace hash mismatch: '+str(p))
    return json.loads(raw),sha

def analyze_artifact(marker,retained_failure=False):
    row,marker_sha=immutable_json(marker)
    if row.get('mode')!='profile':raise ValueError('requires a profile publication marker')
    if retained_failure:
        failed,failed_sha=immutable_json(Path(marker).resolve().parents[3]/'FAILED.json')
        if failed.get('status')!='FAILED' or not failed.get('owned_roots_exited'):
            raise ValueError('retained failure analysis requires terminated FAILED campaign')
        if row.get('status')!='FAIL_MISSING_ACTIVITY_OR_MARKERS':
            raise ValueError('retained mode only accepts explicit activity/marker failure artifact')
    elif row.get('status')!='PASS' or not row.get('activity_validation',{}).get('valid'):
        raise ValueError('requires a completed PASS key_averages publication marker')
    trace=Path(row['trace']);doc,trace_sha=immutable_json(trace,row['trace_sha256'])
    result=analyze_document(doc)
    result.update(marker_path=str(Path(marker).resolve()),marker_sha256=marker_sha,
                  trace_path=str(trace.resolve()),trace_sha256=trace_sha,candidate_index=row['candidate_index'],
                  publication_status=row['status'],retained_failed_campaign=retained_failure)
    if retained_failure:result['failed_campaign_marker_sha256']=failed_sha
    if result['raw_kernel_events']!=row['activity_validation']['cuda_kernel_activity_count']:
        raise ValueError('profile activity count disagrees with published validator')
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=Path);ap.add_argument('--artifact',type=Path)
    ap.add_argument('--retained-failed-artifact',type=Path)
    ap.add_argument('--output',required=True,type=Path);a=ap.parse_args()
    if sum(bool(x) for x in (a.worker,a.artifact,a.retained_failed_artifact))!=1:
        raise ValueError('exactly one of --worker/--artifact/--retained-failed-artifact required')
    if os.sched_getaffinity(0)!={0}:raise RuntimeError('CPU0-only analysis required')
    if a.worker:
        result,_=immutable_json(a.worker/'WORKER-RESULT.json')
        if result.get('status')!='PASS_MEGAVERIFIER_PROFILE' or result.get('profiled_verifier_calls')!=6:
            raise ValueError('requires completed six-profile worker result; never read active files')
        artifacts=[Path(x['key_averages_path']) for d in result['details'] if d['arm']=='PROFILE_CURRENT'
                   for x in d['observation']['artifacts']]
        if len(artifacts)!=6 or len(set(artifacts))!=6:raise ValueError('six unique profile markers required')
    else:artifacts=[a.artifact or a.retained_failed_artifact]
    analyzed=[analyze_artifact(p,bool(a.retained_failed_artifact)) for p in artifacts]
    output=dict(status='CPU_DIAGNOSTIC_RETAINED_FAILED_PROFILE' if a.retained_failed_artifact else 'PASS_CPU_TRACE_ATTRIBUTION',
                campaign_success_claimed=False,
                analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),torch_imported=False,GPU_executed=False,
                profile_count=len(analyzed),profiles=analyzed,
                interpretation='Per-profile mutually exclusive work attribution; not a speedup or complete request latency.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(output,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    for r in analyzed:
        print(json.dumps(dict(trace=r['trace_path'],unique_kernels=r['unique_kernel_count'],
            duration_sum_us=r['unique_kernel_duration_sum_us'],unassigned_duration_fraction=r['unassigned_duration_fraction'],
            module_totals=r['module_totals']),ensure_ascii=False))

if __name__=='__main__':main()
