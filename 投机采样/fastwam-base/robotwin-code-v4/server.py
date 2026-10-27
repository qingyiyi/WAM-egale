"""Official RoboTwin eager canonical in an isolated model process; no LIBERO transform."""
import argparse,hashlib,json,os,pickle,signal,socketserver,struct,time,traceback
from pathlib import Path
import numpy as np
import torch
from runtime import ROOT,build,write
P=argparse.ArgumentParser();P.add_argument('--assets',required=True);P.add_argument('--port',type=int,required=True);args=P.parse_args()
OUT=ROOT/'results/server-attempt01';OUT.mkdir(parents=True,exist_ok=False)
start=time.time();status=dict(status='LOADING',pid=os.getpid(),start=start)
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Robo canonical server cap')));signal.alarm(5400)
torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42)
write(OUT/'STATUS.json',status);policy,cfg=build(args.assets,OUT)
assert policy.action_horizon==32 and policy.replan_steps==24 and policy.seed==42
status.update(status='READY',calls=0,compile_action_infer=False);write(OUT/'STATUS.json',status)
failure=None

def read(sock,n):
 b=bytearray()
 while len(b)<n:
  x=sock.recv(n-len(b))
  if not x:raise EOFError()
  b.extend(x)
 return bytes(b)
class Handler(socketserver.BaseRequestHandler):
 def handle(self):
  global failure
  try:
   assert failure is None,failure
   n=struct.unpack('!I',read(self.request,4))[0];assert 0<n<16*1024*1024
   req=pickle.loads(read(self.request,n));identity=req['identity']
   d=ROOT/'results/episodes'/identity['task'];d.mkdir(parents=True,exist_ok=True);p=d/f'chunk-{identity["chunk"]:03d}'
   assert not p.with_suffix('.json').exists(),'immutable canonical chunk'
   obs=req['obs'];flat={c:obs['observation'][c]['rgb'] for c in ['head_camera','left_camera','right_camera']};flat['joint']=obs['joint_action']['vector']
   np.savez_compressed(p.with_suffix('.npz'),**flat)
   torch.cuda.synchronize();begin=time.perf_counter()
   with torch.no_grad():action=policy._infer_action_chunk(obs,req['instruction'])
   torch.cuda.synchronize();seconds=time.perf_counter()-begin
   assert action.shape==(32,14) and np.isfinite(action).all()
   ts,ds=policy.model.infer_action_scheduler.build_inference_schedule(10,device=policy.model.device,dtype=torch.bfloat16,shift_override=5.)
   row=dict(identity=identity,instruction=req['instruction'],seconds=seconds,concurrent_quality_timing=True,normalized_space='official zscore14, absolute qpos',action=action.tolist(),G=10,scheduler_t=ts.float().cpu().tolist(),scheduler_delta=ds.float().cpu().tolist(),compile_action_infer=False)
   write(p.with_suffix('.json'),row);status.update(status='SERVING',calls=status['calls']+1,updated=time.time());write(OUT/'STATUS.json',status)
   result=dict(ok=True,action=action,seconds=seconds)
  except BaseException as e:
   failure=repr(e);result=dict(ok=False,error=repr(e),traceback=traceback.format_exc());write(OUT/'FAILED.json',result)
  data=pickle.dumps(result,protocol=4);self.request.sendall(struct.pack('!I',len(data))+data)
class Server(socketserver.TCPServer):allow_reuse_address=True
with Server(('127.0.0.1',args.port),Handler) as server:server.serve_forever(poll_interval=.1)
