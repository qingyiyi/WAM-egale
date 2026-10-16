"""Fail-closed identity/coverage gate for one completed EGL collection episode."""

from collections import Counter
import hashlib
import torch


def valid_provenance(prov, task, init, plan_sha256, draft_sha256, target_checkpoint_id):
    return (isinstance(prov, dict) and prov.get('task') == task
            and prov.get('init') == init and prov.get('split') == 'train'
            and prov.get('draft_sha256') == draft_sha256
            and prov.get('plan_sha256') == plan_sha256
            and prov.get('target_checkpoint_id') == target_checkpoint_id)


def valid_row_tensor(row):
    for key in ('state', 'previous', 'target'):
        value = row.get(key)
        if (not isinstance(value, torch.Tensor) or value.shape != (112,)
                or value.dtype != torch.float32 or not bool(torch.isfinite(value).all())):
            return False
    return (hashlib.sha256(row['state'].contiguous().numpy().tobytes()).hexdigest()
            == row.get('state_sha256') == row.get('target_query_state_sha256'))


def validate(item, episode, repro, summary, markers, payloads, events, *, plan_sha256,
             draft_sha256, target_checkpoint_id, threshold):
    task, init = item['task'], item['init']
    attempt = episode.get('episode_attempt_id')
    if (not isinstance(attempt,str) or not attempt
        or episode.get('status')!='completed'
        or episode.get('task_index')!=task or episode.get('episode_index')!=init
        or episode.get('truncated_after_max_policy_chunks')
        or not (episode.get('success') or episode.get('horizon_reached'))):
        raise ValueError('episode task/init/attempt/termination mismatch')
    if (repro.get('command_count')!=episode.get('env_steps')
        or repro.get('actual_EGL_initialization_attempts')!=1
        or repro.get('no_policy_chunk_truncation') is not True
        or not isinstance(repro.get('init_state_sha256'),str)
        or len(repro['init_state_sha256'])!=64):
        raise ValueError('EGL command count or init provenance mismatch')
    chunks=episode.get('infer_chunks')
    if type(chunks) is not int or chunks<=0:
        raise ValueError('missing completed policy chunks')
    if not isinstance(events,list) or len(events)!=chunks:
        raise ValueError('chunk event count mismatch')
    steps=0
    for index,event in enumerate(events):
        before=event.get('env_steps_before');after=event.get('env_steps_after')
        if (event.get('event_schema_version')!='wam-p0-chunk-event/v1'
            or event.get('parallel_worker_id')!=item['id']
            or event.get('task_index')!=task or event.get('episode_index')!=init
            or event.get('episode_attempt_id')!=attempt
            or event.get('rollout_chunk')!=index
            or event.get('action_shape')!=[7,4,4]
            or type(before) is not int or type(after) is not int
            or before!=steps or after<=before or after-before>16
            or event.get('env_steps_delta')!=after-before):
            raise ValueError('chunk event identity/step continuity mismatch')
        steps=after
    if steps!=episode.get('env_steps'):
        raise ValueError('chunk event final steps mismatch')
    if (summary.get('errors') or summary.get('extra_teacher_forwards')!=0
        or summary.get('saved_trees')!=len(payloads)
        or len(payloads)==0 or len(markers)!=chunks):
        raise ValueError('label summary or chunk-marker coverage mismatch')
    marker_trees={}
    for m in markers:
        context=m.get('context',{})
        index=context.get('rollout_chunk')
        if (m.get('complete') is not True
            or m.get('attempted_trees')!=m.get('saved_trees')
            or m.get('nodes_per_tree')!=3
            or context.get('task_index')!=task or context.get('episode_index')!=init
            or context.get('episode_attempt_id')!=attempt
            or not valid_provenance(m.get('provenance'),task,init,plan_sha256,
                                    draft_sha256,target_checkpoint_id)
            or type(index) is not int or not 0<=index<chunks or index in marker_trees):
            raise ValueError('chunk marker task/init/attempt/coverage mismatch')
        marker_trees[index]=m['saved_trees']
    if set(marker_trees)!=set(range(chunks)):
        raise ValueError('missing chunk marker')
    trees=Counter();rows=Counter();unique=set()
    for payload in payloads:
        c=payload.get('contract',{})
        prov=c.get('provenance',{})
        rs=payload.get('rows',[])
        if (payload.get('schema_version')!='bonus-prefix-same-query-teacher/v2'
            or payload.get('extra_teacher_forwards')!=0
            or payload.get('captures_all_three_predictor_nodes') is not True
            or len(rs)!=3 or {r.get('node') for r in rs}!={0,1,2}
            or c.get('mode')!='verifier_k2_b2_bonus' or c.get('bonus_enabled') is not True
            or c.get('selector')!='prefix_then_cost' or c.get('threshold')!=threshold
            or c.get('action_updates_per_chunk')!=50
            or c.get('used_action_channel_ids')!=list(range(7))
            or c.get('cfg_scale')!=1. or c.get('cache_mode')!='original'
            or not valid_provenance(prov,task,init,plan_sha256,
                                    draft_sha256,target_checkpoint_id)):
            raise ValueError('teacher tree source/production contract mismatch')
        chunk=None;step=None
        for r in rs:
            if (r.get('task_index')!=task or r.get('episode_index')!=init
                or r.get('episode_attempt_id')!=attempt or r.get('split')!='train'
                or r.get('source_state_dtype')!='torch.bfloat16'
                or not valid_row_tensor(r)):
                raise ValueError('same-query row task/init/attempt/source mismatch')
            ix=r.get('rollout_chunk');si=r.get('tree_step_index')
            if (type(ix) is not int or type(si) is not int or not 0<=ix<chunks
                or chunk is not None and ix!=chunk or step is not None and si!=step):
                raise ValueError('mixed or invalid tree row identity')
            chunk,step=ix,si
            key=(attempt,ix,si,r['node'])
            if key in unique:raise ValueError('duplicate same-query row identity')
            unique.add(key);rows[ix]+=1
        trees[chunk]+=1
    for index in range(chunks):
        if trees[index]!=marker_trees[index] or rows[index]!=3*marker_trees[index]:
            raise ValueError('tree/marker/three-node count mismatch')
    if sum(trees.values())!=summary['saved_trees']:
        raise ValueError('label total mismatch')
    return dict(status='PASS_COMPLETE_EPISODE_IDENTITY_AND_SAME_QUERY_COVERAGE',
                task=task,init=init,episode_attempt_id=attempt,env_steps=episode['env_steps'],
                infer_chunks=chunks,trees=sum(trees.values()),rows=sum(rows.values()),
                chunk_events=len(events),
                failed_or_horizon_episodes_kept=True,no_extra_teacher_forwards=True)
