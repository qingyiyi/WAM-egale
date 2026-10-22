"""Use pinned RoboTwin full experts to freeze six-task FIT/CAL/DEV scenes.

No policy episode, shortened episode or teacher call is used for seed selection.
Non-stability/infeasibility skips follow the original benchmark; infrastructure
exceptions stop instead of silently searching for a different scene.
"""
from pathlib import Path
import ast,copy,datetime,importlib,os,random,sys,time,traceback,types,json
import numpy as np
import torch
R=Path('/root/cxz/robotwin');S=R/'source/RoboTwin-pinned';L=R/'source/lingbot-va-pinned'
J=Path(sys.argv[1]);J.mkdir(exist_ok=False);config=json.loads((J.parent/'WORKER.json').read_text());os.sched_setaffinity(0,set(config['CPU']['client']));torch.set_num_threads(2);torch.set_num_interop_threads(1)
t0=time.time()
previous=R/'runs/prefilter-4/STATUS.json';prior=json.loads(previous.read_text())
assert prior['status']=='COMPLETE_FIXED_SCENES_NO_POLICY_QUALITY_RESULT'
state=copy.deepcopy(prior);state.update(status='RUNNING_EXPERT_FILTER_ONLY',start=t0,expert_attempts=0,policy_episodes=0,teacher_forwards=0,source_previous=str(previous))
for key in ['end','error','traceback','lifecycle_seconds']:state.pop(key,None)
for rec in state['tasks']:
 rec['prior_expert_attempts']=rec['expert_attempts'];rec['expert_attempts']=0
 rec['next_seed']=max(x['seed'] for x in rec['valid'])+1

def save():
 p=J/'STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(p)
def jsonable(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,(np.integer,np.floating)):return x.item()
 if isinstance(x,dict):return {str(k):jsonable(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
 return x

def physical(env):
 scene=env.scene;actors=[];articulations=[]
 for a in scene.get_all_actors():
  pose=a.get_pose();row=dict(name=a.get_name(),p=np.asarray(pose.p).tolist(),q=np.asarray(pose.q).tolist())
  for name in ['get_velocity','get_angular_velocity']:
   row[name]=np.asarray(getattr(a,name)()).tolist() if hasattr(a,name) else 'UNKNOWN_API_UNAVAILABLE'
  actors.append(row)
 for a in scene.get_all_articulations():
  pose=a.get_pose();articulations.append(dict(name=a.get_name(),p=np.asarray(pose.p).tolist(),q=np.asarray(pose.q).tolist(),qpos=np.asarray(a.get_qpos()).tolist(),qvel=np.asarray(a.get_qvel()).tolist()))
 return dict(actors=actors,articulations=articulations)

def capture(env,instruction,seed,info):
 rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_CPU=torch.random.get_rng_state(),torch_CUDA=torch.cuda.get_rng_state_all())
 observation=env.get_obs();ph=physical(env)
 # Observation read is not allowed to advance the evaluation RNG stream.
 random.setstate(rng['python']);np.random.set_state(rng['numpy']);torch.random.set_rng_state(rng['torch_CPU']);torch.cuda.set_rng_state_all(rng['torch_CUDA'])
 return dict(seed=seed,instruction=instruction,physical=ph,initial_observation=observation,RNG=rng,expert_info=info)

def role(i):
 if i<8:return dict(stage='TARGET',split='FIT' if i<6 else 'CAL')
 if i<11:return dict(stage='PARENT',split='FIT' if i<10 else 'CAL')
 if i<14:return dict(stage='CURRENT',split='FIT' if i<13 else 'CAL')
 return dict(stage='PRIOR_DEV' if i<19 else 'NEW_EXPANDED_DEV',split='DEV')
try:
 torch.cuda.init() # Runtime initialization for the real expert, before Vulkan.
 from device_binding import install
 state['render_device']=install(J/'RENDER-DEVICE.json');save()
 for p in (L,S):sys.path.insert(0,str(p))
 os.chdir(S)
 path=L/'evaluation/robotwin/eval_polict_client_openpi.py';tree=ast.parse(path.read_text())
 class Root(ast.NodeTransformer):
  def visit_Assign(self,n):
   if any(isinstance(t,ast.Name) and t.id=='robowin_root' for t in n.targets):n.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(value=str(S))],keywords=[])
   return self.generic_visit(n)
 tree=ast.fix_missing_locations(Root().visit(tree));m=types.ModuleType('_robotwin_pinned_client');m.__file__=str(path);sys.modules[m.__name__]=m;exec(compile(tree,str(path),'exec'),m.__dict__)
 def expert_filter(task_name,env,args,model,st_seed,test_num=100,**kwargs):
  rec=next(t for t in state['tasks'] if t['task']==task_name);save();args=copy.deepcopy(args);args['eval_mode']=True;args['render_freq']=0;args['eval_video_log']=False
  seed=rec['next_seed']
  while len(rec['valid'])<100:
   if time.time()-t0>13500 or rec['expert_attempts']>=600:raise TimeoutError('pre-registered expert filter wall/candidate cap')
   rec['expert_attempts']+=1;state['expert_attempts']+=1;rec['active_seed']=seed;save()
   try:
    env.setup_demo(now_ep_num=len(rec['valid']),seed=seed,is_test=True,**args)
    info=env.play_once();env.close_env()
   except m.UnStableError as e:
    env.close_env();rec['skipped'].append(dict(seed=seed,reason='UnStableError',detail=str(e)));seed+=1;save();continue
   except AssertionError as e:
    detail=traceback.format_exc()
    try:env.close_env()
    except BaseException:pass
    if str(e)!='target_pose cannot be None for move action.' or 'in grasp_actor' not in detail:raise
    rec['skipped'].append(dict(seed=seed,reason='original_expert_no_grasp_pose',traceback=detail));seed+=1;save();continue
   except BaseException:
    try:env.close_env()
    except BaseException:pass
    raise
   if not(env.plan_success and env.check_success()):
    rec['skipped'].append(dict(seed=seed,reason='original_expert_infeasible_or_unsuccessful'));seed+=1;save();continue
   env.setup_demo(now_ep_num=len(rec['valid']),seed=seed,is_test=True,**args)
   descriptions=m.generate_episode_descriptions(task_name,[info['info']],100)
   instruction=str(np.random.choice(descriptions[0]['seen']));env.set_instruction(instruction=instruction)
   payload=capture(env,instruction,seed,info);case=J/task_name/f'valid-{len(rec["valid"]):02d}';case.mkdir(parents=True,exist_ok=False);torch.save(payload,case/'INITIAL.pt')
   row=dict(task=task_name,valid_index=len(rec['valid']),seed=seed,instruction=instruction,role=role(len(rec['valid'])),initial=str(case/'INITIAL.pt'),task_step_limit=int(env.step_lim))
   (case/'CASE.json').write_text(json.dumps(row,indent=2)+'\n');rec['valid'].append(row);env.close_env(clear_cache=True);seed+=1;save()
  return seed,0
 m.eval_policy=expert_filter;m.WebsocketClientPolicy=lambda **kwargs:None
 # Real benchmark task/environment construction, not its optional render test.
 for task in config['tasks']:
  m.main(dict(task_name=task,task_config='demo_clean',ckpt_setting='M40-eval100-expert-manifest',save_root=str(J/'official-output'),policy_name='LingBotVA-original',seed=0,port=34956,video_guidance_scale=5,action_guidance_scale=1,test_num=100))
 state['status']='COMPLETE_FIXED_SCENES_NO_POLICY_QUALITY_RESULT'
except BaseException as e:state.update(status='FAILED_OR_PARTIAL_EXPERT_PREPARATION',error=repr(e),traceback=traceback.format_exc())
finally:state.update(end=time.time(),lifecycle_seconds=time.time()-t0);save()
