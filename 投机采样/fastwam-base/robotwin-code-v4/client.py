"""Pinned FastWAM-vendored RoboTwin eval loop, observation/command audit only."""
import argparse,hashlib,json,os,pickle,signal,socket,struct,sys,time,traceback,types,random
from collections import deque
from pathlib import Path
import numpy as np
P=argparse.ArgumentParser();P.add_argument('--task',required=True);P.add_argument('--port',type=int,required=True);args=P.parse_args()
ROOT=Path(__file__).resolve().parent.parent;VENDOR=ROOT/'source/RoboTwin';OUT=ROOT/'results/episodes'/args.task;OUT.mkdir(parents=True,exist_ok=True)
assert not (OUT/'EPISODE.json').exists();start=time.time();current_seed=None
status=dict(task=args.task,status='INITIALIZING',start=start,complete=False,success=False,model_seed=42,scene_seed_start=4300000,instruction_type='unseen',config='demo_clean',invalid_attempts=[])
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('canonical task1800s cap')));signal.alarm(1800)

def write(name,row):
 p=OUT/name;q=p.with_suffix('.tmp');q.write_text(json.dumps(row,indent=2)+'\n');q.replace(p)
def read(sock,n):
 b=bytearray()
 while len(b)<n:
  x=sock.recv(n-len(b))
  if not x:raise EOFError()
  b.extend(x)
 return bytes(b)
class Policy:
 def __init__(self):self.pending=deque();self.commands=[];self.chunks=0;self.policy_seconds=0.
 def reset(self):
  assert self.chunks==0,'one valid episode per task in canonical pilot';self.pending.clear()
 def should_request_observation(self):return not self.pending
 def step(self,env,obs):
  if not self.pending:
   assert obs is not None
   small=dict(observation={c:dict(rgb=np.array(obs['observation'][c]['rgb'],copy=True)) for c in ['head_camera','left_camera','right_camera']},joint_action=dict(vector=np.array(obs['joint_action']['vector'],copy=True)))
   if self.chunks==0:
    status.update(scene_seed=current_seed,instruction=env.get_instruction(),first_rgb_sha256={c:hashlib.sha256(v['rgb'].tobytes()).hexdigest() for c,v in small['observation'].items()},initial_joint=small['joint_action']['vector'].tolist(),step_limit=env.step_lim)
   payload=pickle.dumps(dict(identity=dict(task=args.task,seed=current_seed,chunk=self.chunks,arm='T'),obs=small,instruction=env.get_instruction()),protocol=4)
   with socket.create_connection(('127.0.0.1',args.port),timeout=1200) as s:
    s.sendall(struct.pack('!I',len(payload))+payload);n=struct.unpack('!I',read(s,4))[0];reply=pickle.loads(read(s,n))
   assert reply['ok'],reply
   action=reply['action'];assert action.shape==(32,14) and np.isfinite(action).all()
   self.pending.extend(action[:24]);self.chunks+=1;self.policy_seconds+=reply['seconds']
  a=self.pending.popleft();env.take_action(a,action_type='qpos');self.commands.append(a.copy());status.update(status='RUNNING',commands=len(self.commands),chunks=self.chunks,policy_seconds=self.policy_seconds);write('STATUS.json',status)
policy=Policy()
class Audit:
 def begin(self,seed):
  global current_seed
  current_seed=int(seed);status['attempted_seed']=current_seed
  write(f'python-rng-before-seed-{current_seed}.json',dict(seed=current_seed,python_random_state=random.getstate(),source='observed without reseeding; official BaseTask seeds NumPy/Torch but not Python random'))
  assert current_seed<4300100,'100 candidate-seed cap; preserve invalid/PARTIAL'
 def finish(self,seed,success,count,limit):
  assert len(policy.commands)==count
  status.update(status='SUCCESS' if success else 'FAILURE',success=bool(success),complete=True,scene_seed=int(seed),commands=int(count),step_limit=int(limit),chunks=policy.chunks,policy_seconds=policy.policy_seconds,wall_seconds=time.time()-start,end=time.time())
  np.save(OUT/'executed-actions.npy',np.asarray(policy.commands,dtype=np.float32));write('EPISODE.json',status)
 def invalid(self,seed,reason):
  status['invalid_attempts'].append(dict(seed=int(seed),reason=reason));write('STATUS.json',status)
audit=Audit()
try:
 os.chdir(VENDOR);sys.path[:0]=[str(VENDOR),str(VENDOR/'policy'),str(VENDOR/'description/utils')]
 source=VENDOR/'script/eval_policy.py';text=source.read_text()
 # Observe existing decisions without replacing seed filtering or success rules.
 replacements={
  '    while succ_seed < test_num:\n':'    while succ_seed < test_num:\n        audit.begin(now_seed)\n',
  '        if succ:\n            TASK_ENV.suc += 1':'        audit.finish(now_seed, succ, TASK_ENV.take_action_cnt, TASK_ENV.step_lim)\n        if succ:\n            TASK_ENV.suc += 1',
  '            now_seed += 1\n            args["render_freq"] = render_freq\n            continue':'            audit.invalid(now_seed, "expert_check_failed")\n            now_seed += 1\n            args["render_freq"] = render_freq\n            continue',
 }
 for old,new in replacements.items():assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new)
 # Exceptions stay in the original loop and are retained in the task log; seed skips
 # are also listed so invalid seeds cannot silently count as policy failures.
 text=text.replace('                now_seed += 1\n','                audit.invalid(now_seed, "expert_setup_exception_logged")\n                now_seed += 1\n')
 text=text.replace('            succ_seed -= 1\n','            audit.invalid(now_seed, "rollout_setup_exception_logged")\n            succ_seed -= 1\n')
 (OUT/'INSTRUMENTED-EVAL.py').write_text(text)
 namespace=dict(__name__='fixed_robotwin_eval',__file__=str(source),audit=audit)
 exec(compile(text,str(source),'exec'),namespace)
 def decorator(policy_name,method):
  if method=='get_model':return lambda _:policy
  if method=='reset_model':return lambda model:model.reset()
  if method=='eval':return lambda env,model,obs:model.step(env,obs)
  raise ValueError(method)
 namespace['eval_function_decorator']=decorator
 generator=namespace['generate_episode_descriptions']
 def recorded_instructions(task,info,count):
  write('INSTRUCTION-GENERATION.json',dict(task=task,count=count,python_random_state=random.getstate(),info=info,instruction_type='unseen'))
  return generator(task,info,count)
 namespace['generate_episode_descriptions']=recorded_instructions
 namespace['main'](dict(task_name=args.task,task_config='demo_clean',ckpt_setting='fastwam-base-8eaceeb',policy_name='fastwam_rpc',instruction_type='unseen',eval_num_episodes=1,seed=42,eval_output_dir=str(OUT/'official'),skip_get_obs_within_replan=True))
 assert status['complete'],'official loop did not produce one valid terminal episode'
except BaseException as e:
 if not status['complete']:
  status.update(status='PARTIAL_OR_INVALID',error=repr(e),traceback=traceback.format_exc(),end=time.time(),wall_seconds=time.time()-start)
  np.save(OUT/'executed-actions.npy',np.asarray(policy.commands,dtype=np.float32));write('EPISODE.json',status)
 raise
finally:signal.alarm(0)
