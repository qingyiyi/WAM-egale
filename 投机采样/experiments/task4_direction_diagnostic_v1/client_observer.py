"""Full episode observation at transport, submitted action and LIBERO entry."""
import json
from pathlib import Path
import numpy as np

def install(client,out):
    out=Path(out)
    original_init,original_episode=client._init_env,client._run_episode
    active=dict(chunk=None,command_offset=0,frames=None)
    submitted=[];environment=[];chunks=[]
    def init(env,init_state):
        obs,timestep=original_init(env,init_state)
        old_step,underlying=env.step,env.env.step
        def internal(command,*args,**kwargs):
            value=np.array(command,dtype=np.float32,copy=True)
            result=underlying(command,*args,**kwargs)
            environment.append(dict(chunk=active['chunk'],command=value.tolist(),
                env_index=len(environment),done=bool(result[2])))
            return result
        env.env.step=internal
        def step(command):
            k=active['command_offset'];chunk=active['chunk'];pred=active['frames']
            first_frame=1 if chunk==0 else 0
            frame=first_frame+k//pred.shape[2];action=k%pred.shape[2]
            value=np.array(command,dtype=np.float32,copy=True)
            expected=np.asarray(pred[:,frame,action],dtype=np.float32)
            if value.tobytes()!=expected.tobytes():raise ValueError('transport to submitted action mapping differs')
            submitted.append(dict(chunk=chunk,command_offset=k,frame=frame,subaction=action,
                command=value.tolist(),env_index=len(submitted)))
            active['command_offset']+=1
            return old_step(command)
        env.step=step
        return obs,timestep
    def episode(**kwargs):
        model=kwargs['model'];old_infer=model.infer
        def infer(request):
            result=old_infer(request)
            if not request.get('reset') and not request.get('compute_kv_cache'):
                context=dict(request['p0_trace_context']);chunk=int(context['rollout_chunk'])
                frames=np.array(result['action'],copy=True)
                active.update(chunk=chunk,command_offset=0,frames=frames)
                chunks.append(dict(chunk=chunk,context=context,received_action=frames.tolist(),
                    dtype=str(frames.dtype),shape=list(frames.shape),env_steps_before=context['env_steps_before']))
            return result
        model.infer=infer
        try:
            record=original_episode(**kwargs)
            if len(submitted)!=len(environment) or len(submitted)!=record['env_steps']:
                raise ValueError('submitted/environment/full task command coverage differs')
            for a,b in zip(submitted,environment):
                if a['chunk']!=b['chunk'] or a['command']!=b['command']:
                    raise ValueError('command changed before LIBERO env.step')
            payload=dict(schema='physical-action-to-environment-command/v1',chunks=chunks,
                submitted=submitted,environment=environment,complete_episode=True,
                termination_reason='success' if record['success'] else 'full_horizon',
                env_steps=record['env_steps'],success=record['success'],performance_claim=False,
                environment_command_boundary='argument to underlying env.env.step; not actuator ctrl')
            with (out/'COMMAND-MAPPING.json').open('x') as f:json.dump(payload,f,allow_nan=False)
            return record
        finally:model.infer=old_infer
    client._init_env,client._run_episode=init,episode
