"""Frozen-weight GPU deployment interface audit; no threshold or weight selection."""
from pathlib import Path
import json,os,sys,time,traceback
os.sched_setaffinity(0,{18,19})
import torch
from draft import Transformer,Predictor,SPEC
base=Path('/root/cxz/robotwin/runs/tf-migration-20261021')
plan=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
torch.set_num_threads(2);torch.set_num_interop_threads(1)
start=time.time();result=dict(status='RUNNING_FIXED_INTERFACE_AUDIT',rows=0,decision_flips=0,teacher_forward=0,quality_passed=False)
def save():(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
try:
 ck=torch.load(plan['parent_checkpoint'],map_location='cpu',weights_only=True);assert ck['specification']==SPEC
 model=Transformer().float().cuda();model.load_state_dict(ck['model_state_dict']);model.eval();p=Predictor(model,ck['normalization'])
 target=torch.load(base/'target48.pt',map_location='cpu',weights_only=True)['data']['CAL']
 observed=torch.load(Path(plan['observer_audit'])/'observed18.pt',map_location='cpu',weights_only=True)['data']['CAL']
 tau=json.loads((base/'CALIBRATION.json').read_text())['tau'];cpu_rng=torch.get_rng_state();gpu_rng=torch.cuda.get_rng_state()
 result.update(condition=ck['metadata']['condition'],checkpoint=plan['parent_checkpoint'],fixed_parent_tau=tau,threshold_recalibrated=False,panels=[])
 with torch.inference_mode():
  for label,data in [('Target_CAL12',target),('CURRENT_CAL6',observed)]:
   errors=[];before=result['rows']
   for i in range(len(data['state'])):
    assert time.time()-start<240
    sig,nxt,t=data['scalars'][i].tolist();x=data['state'][i].reshape(16,2,16).bfloat16().cuda();previous=data['previous'][i].reshape_as(x).bfloat16().cuda()
    deployed=p.predict(x,previous,sig,nxt,t*1000).reshape(2,512)
    direct=p.decode_batch(dict(state=x.reshape(1,512).float(),previous=previous.reshape(1,512).float(),scalars=torch.tensor([[sig,nxt,t]],device=x.device))).bfloat16().reshape(2,512)
    assert torch.isfinite(deployed).all() and torch.equal(deployed,direct)
    teacher=data['target'][i].cuda();cost=(deployed.float()-teacher).square().sum(-1).sqrt()*abs(sig-nxt);errors.append(float(cost.min()));result['rows']+=1
   e=torch.tensor(errors);torch.save(e,out/(label+'-errors.pt'));result['panels'].append(dict(name=label,rows=len(errors),adapter_decode_bitwise=True,accepted_at_fixed_tau=int((e<=tau).sum()),median_error=float(e.median()),q90_error=float(torch.quantile(e,.9))))
 assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
 result.update(status='PASS_FINITE_GPU_ADAPTER_DECODE_NOT_CLOSED_LOOP',RNG_unchanged=True,B2_B14_teacher_parity='NOT_TESTED',checkpoint_selection=False)
except BaseException as e:result.update(status='FAILED_FIXED_INTERFACE_AUDIT',error=repr(e),traceback=traceback.format_exc())
finally:result['elapsed_seconds']=time.time()-start;save()
if result['status']!='PASS_FINITE_GPU_ADAPTER_DECODE_NOT_CLOSED_LOOP':raise SystemExit(1)
