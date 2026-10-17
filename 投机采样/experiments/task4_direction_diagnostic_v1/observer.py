"""Observe existing directions and authoritative transitions; no extra forward."""
from __future__ import annotations
import hashlib
from pathlib import Path
import time
import types

import torch

def cpu(x):
    if isinstance(x,torch.Tensor):return x.detach().contiguous().cpu().clone()
    raise TypeError(type(x))

def digest(x):return hashlib.sha256(cpu(x).view(torch.uint8).numpy().tobytes()).hexdigest()

class Observer:
    def __init__(self,server,job,arm):
        self.s,self.job,self.arm=server,Path(job),arm
        self.active=None;self.round=None;self.accounting=None
        server._direction_diag=self

    def compact(self,x):
        return cpu(self.s._p0_action_compact(x[0]))

    def install_action(self):
        original=self.s._infer_action_verifier_k2_b2
        path=Path(__file__).with_name('action_observed.py')
        scope=dict(original.__func__.__globals__)
        exec(compile(path.read_text(),str(path),'exec'),scope)
        self.s._infer_action_verifier_k2_b2=types.MethodType(scope[original.__func__.__name__],self.s)

    def install_io(self,accounting):
        self.accounting=accounting
        scheduler=self.s.action_scheduler
        old_step=scheduler.step
        def step(direction,timestep,state,*args,**kwargs):
            result=old_step(direction,timestep,state,*args,**kwargs)
            if self.active is not None:
                if isinstance(result,tuple):raise ValueError('diagnostic expects original return_dict=False tensor')
                item=dict(id=len(self.active['scheduler_calls']),timestep=float(timestep),
                    state=self.compact(state),direction=self.compact(direction),raw_next=self.compact(result),
                    projected_next=None,round_id=None if self.round is None else self.round['id'])
                self.active['scheduler_calls'].append(item)
                if self.arm=='T':
                    i=len(self.active['commits'])
                    sigmas=self.s.action_scheduler.sigmas
                    self.active['commits'].append(dict(step=i,source='target',bonus=False,scheduler_call=item['id'],
                        state=item['state'],direction=item['direction'],sigma=float(sigmas[i]),
                        next_sigma=float(sigmas[i+1]) if i+1<len(sigmas) else 0.))
            return result
        scheduler.step=step
        old_post=self.s.postprocess_action
        def post(_s,action):
            result=old_post(action)
            if self.active is not None:
                import numpy as np
                self.active['final_normalized_action']=cpu(action)
                self.active['decoded_physical_action']=np.array(result,copy=True)
            return result
        self.s.postprocess_action=types.MethodType(post,self.s)
        old_infer=self.s.infer
        def infer(_s,request):
            policy=not request.get('reset') and not request.get('compute_kv_cache')
            if not policy:return old_infer(request)
            if self.active is not None:raise RuntimeError('nested policy diagnostic')
            context=dict(request.get('p0_trace_context') or {})
            self.active=dict(schema='teacher-to-command-chunk/v1',arm=self.arm,context=context,
                task=context['task_index'],init=context['episode_index'],chunk=context['rollout_chunk'],
                scheduler_calls=[],commits=[],rounds=[],action_timesteps=None,
                CPU_rng_before=digest(torch.get_rng_state()),CUDA_rng_before=digest(torch.cuda.get_rng_state(self.s.device)),
                target_forwards_before=accounting.attempted,performance_claim=False)
            started=time.monotonic()
            try:
                result=old_infer(request)
                import numpy as np
                self.active['submitted_server_action']=np.array(result['action'],copy=True)
                self.active.update(target_forwards=accounting.attempted-self.active['target_forwards_before'],
                    CPU_rng_after=digest(torch.get_rng_state()),CUDA_rng_after=digest(torch.cuda.get_rng_state(self.s.device)),
                    instrumented_policy_wall_sec=time.monotonic()-started,terminal_status='COMPLETE_POLICY_CHUNK')
                expected=len(self.s.action_scheduler.sigmas)
                if len(self.active['commits'])!=expected:raise ValueError('actual committed scheduler update coverage mismatch')
                if [r['step'] for r in self.active['commits']]!=list(range(expected)):
                    raise ValueError('actual commit ordering differs')
                if self.round is not None:raise ValueError('unclosed diagnostic round')
                folder=self.job/'episodes'/context['parallel_worker_id']/'direction_trace'
                folder.mkdir(parents=True,exist_ok=True)
                path=folder/f"chunk-{context['rollout_chunk']:03d}.pt"
                if path.exists():raise FileExistsError(path)
                torch.save(self.active,path)
                return result
            finally:
                self.active=None;self.round=None
        self.s.infer=types.MethodType(infer,self.s)

    def begin_action(self,actions,timesteps,frame_st_id,count):
        if self.active is None:raise RuntimeError('action outside logged policy')
        self.active.update(initial_action_state=cpu(actions),action_timesteps=cpu(timesteps),
            action_sigmas=cpu(self.s.action_scheduler.sigmas),frame_st_id=int(frame_st_id),expected_updates=int(count))

    def begin_round(self,step,remaining):
        if self.round is not None:raise RuntimeError('unclosed prior round')
        self.round=dict(id=len(self.active['rounds']),start_step=int(step),remaining=int(remaining),
            started_monotonic=time.monotonic(),target_before=self.accounting.attempted,
            commits_before=len(self.active['commits']),kind='target_single')

    def projected(self,result,timestep,offset):
        call=self.active['scheduler_calls'][-1]
        if call['timestep']!=float(timestep):raise ValueError('scheduler projection order changed')
        call.update(projected_next=cpu(result),offset=int(offset))

    def tree(self,tree,step):
        if self.round['start_step']!=step:raise ValueError('tree step mismatch')
        self.round.update(kind='speculative_tree',candidate_states=cpu(tree.verifier_states()),
            root_previous_direction=cpu(tree.root_previous_direction),first_directions=cpu(tree.first_directions),
            second_directions=cpu(tree.second_directions),sigma_path=list(tree.sigma_path),
            target_timesteps=list(tree.target_timesteps),candidate_count=7)

    def score(self,tree,scores,step):
        self.round.update(teacher_target_directions=cpu(scores.target_directions),
            first_costs=list(scores.first_costs),second_costs=[list(x) for x in scores.second_costs])

    def executed(self,tree,scores,route,result,step):
        self.round.update(accepted_depth=int(result.accepted_depth),selected_parent=int(route.parent),
            selected_child=int(route.child),execution_next_state=cpu(result.next_state),
            threshold=float(self.s._p0_action_draft_threshold))

    def commit(self,step,state,direction,source,bonus):
        state_cpu,direction_cpu=cpu(state),cpu(direction)
        matches=[x for x in self.active['scheduler_calls'] if x['round_id']==self.round['id']
            and torch.equal(x['state'],state_cpu) and torch.equal(x['direction'],direction_cpu)]
        if not matches:raise ValueError('committed update lacks exact authoritative scheduler transition')
        call=matches[-1]
        sigmas=self.s.action_scheduler.sigmas
        self.active['commits'].append(dict(step=int(step),source=source,bonus=bool(bonus),
            scheduler_call=call['id'],state=state_cpu,direction=direction_cpu,
            sigma=float(sigmas[step]),next_sigma=float(sigmas[step+1]) if step+1<len(sigmas) else 0.))

    def end_round(self,end_step):
        if self.round is None:raise RuntimeError('round missing')
        value=self.round
        g=len(self.active['commits'])-value['commits_before']
        if g!=int(end_step)-value['start_step'] or g not in (1,2,3):raise ValueError('effective commit conservation failed')
        value.update(g=g,end_step=int(end_step),target_forwards=self.accounting.attempted-value['target_before'],
            instrumented_round_wall_sec=time.monotonic()-value['started_monotonic'],
            clean_performance_measurement=False)
        self.active['rounds'].append(value);self.round=None

    def end_action(self,actions):self.active['final_scheduler_action']=cpu(actions)
