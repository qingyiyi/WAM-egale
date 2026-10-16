"""Offline independent complete-output and full-episode pilot comparison."""
import argparse
import json
import math
from pathlib import Path
import sys

from task_contract import OWN, plan, write, digest, stamp

def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    args=ap.parse_args();job=args.job.resolve();p=plan(job)
    collection=json.loads((job/'GPU-COLLECTION.json').read_text())
    if (collection['status']!='PASS_GPU_FULL_EPISODE_COLLECTION'
            or collection['completed_episodes']!=4 or collection['model_launches']!=4
            or collection['reused_target_episodes']!=2 or collection['full_episode_data_total']!=6):
        raise ValueError('complete formal collection required')
    import numpy as np
    import torch
    from probe import snapshot_module
    snap=snapshot_module()
    if torch.cuda.is_initialized():raise ValueError('offline audit must be CPU')
    data={};meta={};count={};output_pairs=0;elements=0
    for arm in ('T','D','S'):
        for w,init in enumerate((3,4)):
            root=Path(p['reference_target_job']) if arm=='T' else job
            out=root/'campaigns'/arm/'workers'/('worker'+str(w))
            rr=rows(out/'client/episodes.jsonl')
            if len(rr)!=1:raise ValueError('one full episode per init/arm')
            r=rr[0]
            if (r['episode_index']!=init or r['status']!='completed' or r['libero_horizon']!=800
                    or r['max_policy_chunks'] is not None or r['truncated_after_max_policy_chunks']
                    or not (r['success'] or r['horizon_reached'])):raise ValueError('full horizon audit')
            mm=json.loads((out/'client/repro_metadata.json').read_text())
            cc=json.loads((out/'COUNTER-CLOSED.json').read_text())
            if (cc['target_intents']!=cc['host_returned'] or cc['failed_forwards']
                    or cc['request_active'] or cc['active_forwards']):raise ValueError('closed counting')
            if mm['command_count']!=r['env_steps'] or mm['actual_EGL_initialization_attempts']!=1:
                raise ValueError('actual client coverage')
            if arm=='S' and (cc['shared']['requests']!=r['infer_chunks']
                    or not cc['shared']['last_request_clean'] or not cc['shared']['candidates']):
                raise ValueError('all actual chunks used accepted S0')
            if arm!='S' and cc['shared']['enabled']:raise ValueError('baseline sharing changed')
            data[(arm,init)]=r;meta[(arm,init)]=mm;count[(arm,init)]=cc
    paired=[]
    initial_keys=('model_xml_sha256','init_state_sha256','initial_state_hashes',
                  'initial_obs_hashes','initial_rng')
    for w,init in enumerate((3,4)):
        for k in initial_keys:
            if any(meta[(a,init)][k]!=meta[('T',init)][k] for a in ('D','S')):
                raise ValueError('paired initial environment mismatch: '+k)
        dd=meta[('D',init)];ss=meta[('S',init)]
        if dd['final_state_hashes']!=ss['final_state_hashes'] or dd['final_rng']!=ss['final_rng']:
            raise ValueError('D/S terminal environment/RNG differs')
        droot=job/'campaigns/D/workers'/('worker'+str(w))
        sroot=job/'campaigns/S/workers'/('worker'+str(w))
        command_d=np.load(droot/'client/commands.npy');command_s=np.load(sroot/'client/commands.npy')
        if not np.array_equal(command_d,command_s):raise ValueError('D/S actual command bytes differ')
        dpaths={x.name:x for x in (droot/'server_output').rglob('*.pt')}
        spaths={x.name:x for x in (sroot/'server_output').rglob('*.pt')}
        if dpaths.keys()!=spaths.keys():raise ValueError('D/S full original artifact coverage differs')
        expected_outputs=2*data[('D',init)]['infer_chunks']
        actual_outputs=0
        for name,path in sorted(dpaths.items()):
            d=torch.load(path,map_location='cpu',weights_only=False)
            s=torch.load(spaths[name],map_location='cpu',weights_only=False)
            if snap.fingerprint(d)!=snap.fingerprint(s):raise ValueError('D/S original output/observation bytes differ: '+name)
            if name.startswith(('latents_','actions_')):
                if not isinstance(d,torch.Tensor):raise ValueError('saved actual tensors required')
                if name.startswith('actions_') and d.shape[1]!=30:raise ValueError('full30 output coverage')
                output_pairs+=1;elements+=d.numel();actual_outputs+=1
        if actual_outputs!=expected_outputs:raise ValueError('all chunks action/video saved')
        if count[('D',init)]['calls']!=count[('S',init)]['calls']:
            raise ValueError('D/S model-forward route distribution differs')
        if count[('D',init)]['phase_counters']!=count[('S',init)]['phase_counters']:
            raise ValueError('D/S accepted depths or committed route differs')
        t,d,s=(data[(a,init)] for a in ('T','D','S'))
        if (d['success'],d['env_steps'],d['infer_chunks'])!=(s['success'],s['env_steps'],s['infer_chunks']):
            raise ValueError('D/S full episode trajectory coverage differs')
        paired.append(dict(init=init,T=t,D=d,S=s,
          T_over_D_E2E=t['elapsed_sec']/d['elapsed_sec'],
          T_over_S_E2E=t['elapsed_sec']/s['elapsed_sec'],
          D_over_S_incremental=d['elapsed_sec']/s['elapsed_sec'],
          T_over_S_outer=t['outer_episode_wall_sec']/s['outer_episode_wall_sec'],
          T_over_S_model_calls=t['model_request_wall_sec']/s['model_request_wall_sec'],
          S_over_T_env_steps=s['env_steps']/t['env_steps'],
          quality_and_length_goal=bool(t['success'] and s['success'] and s['env_steps']<=1.05*t['env_steps']),
          D_S_all_commands_outputs_and_terminal_state_bitwise=True,
          phase_counters=count[('S',init)]['phase_counters'],
          target_intents={a:count[(a,init)]['target_intents'] for a in ('T','D','S')}))
    sumtime={a:sum(data[(a,i)]['elapsed_sec'] for i in (3,4)) for a in ('T','D','S')}
    geom=lambda k:math.exp(sum(math.log(v[k]) for v in paired)/len(paired))
    ratios=[v['T_over_S_E2E'] for v in paired]
    verdict='positive_pilot' if all(v>1 for v in ratios) else 'negative_pilot' if all(v<1 for v in ratios) else 'mixed_pilot'
    result=dict(status='PASS_FULL_EPISODE_OFFLINE_AUDIT',timestamp=stamp(),job=str(job),
          plan_sha256=digest(job/'plan.json'),collection=collection,paired=paired,
          total_elapsed_seconds=sumtime,T_over_S_total=sumtime['T']/sumtime['S'],
          D_over_S_total=sumtime['D']/sumtime['S'],
          T_over_S_geomean=geom('T_over_S_E2E'),D_over_S_geomean=geom('D_over_S_incremental'),
          E2E_verdict=verdict,quality_and_length_goal_all=all(v['quality_and_length_goal'] for v in paired),
          D_S_bitwise_all_full_outputs=True,full_output_tensor_pairs=output_pairs,
          full_output_elements=elements,CUDA_initialized=torch.cuda.is_initialized(),
          reused_target_job=p['reference_target_job'],
          caveat='libero_10 task0 initial states3/4, one sample each; parallel2; frozen target episodes from v1 before launch-race failure reused with identical model/client code; not a multi-task stable speedup claim',
          next_step='trajectory environment-step control only after this original-policy E2E pilot')
    write(job/'FINAL_RESULT.json',result,exclusive=True,durable=True)
    print(json.dumps({k:result[k] for k in ('status','T_over_S_total','D_over_S_total',
          'E2E_verdict','quality_and_length_goal_all','full_output_tensor_pairs')},ensure_ascii=False))

if __name__=='__main__':main()
