"""Binding-only correction; original Attempt and generated arithmetic preserved."""
import inspect
from pathlib import Path
import types
import torch
from task4_direction_diagnostic_v1.observer import Observer as PreviousObserver
from task4_direction_diagnostic_v1.protocol import read,write,sha

class Observer(PreviousObserver):
    def install_action(self):
        raw=inspect.unwrap(self.s._infer_action_verifier_k2_b2)
        path=Path(__file__).resolve().parent.parent/'task4_direction_diagnostic_v1/action_observed.py'
        scope=dict(raw.__globals__)
        if scope.get('torch') is not torch:raise ValueError('unwrapped original server torch binding required')
        exec(compile(path.read_text(),str(path),'exec'),scope)
        self.s._infer_action_verifier_k2_b2=types.MethodType(scope[raw.__name__],self.s)

def load_with_existing_cal(p,arm,device,channels,out):
    from tf_usable_train_v1.adapter import load_candidate
    proof=Path(p['existing_CAL_certificate']);parity=read(proof)
    if (sha(proof)!=p['existing_CAL_sha256'] or parity['status']!='PASS_FULL_REAL_CAL_SAME_DEVICE_ADAPTER_PARITY'
            or parity['rows']!=20237 or not parity['complete_CAL'] or not parity['same_device_bitwise_exact']
            or not parity['CPU_rng_unchanged'] or not parity['CUDA_rng_unchanged']
            or parity['checkpoint_sha256']!=p['selected'][arm]['checkpoint_sha256']):
        raise ValueError('complete unchanged current-attempt CAL certificate required')
    cpu,cuda=torch.get_rng_state().clone(),torch.cuda.get_rng_state(device).clone()
    predictor=load_candidate(p['selected'][arm]['checkpoint'],'transformer64',device=device,
        expected_channels=channels,expected_sha256=p['selected'][arm]['checkpoint_sha256'])
    if not torch.equal(cpu,torch.get_rng_state()) or not torch.equal(cuda,torch.cuda.get_rng_state(device)):
        raise ValueError('unchanged candidate loader changed target RNG')
    write(out,dict(status='REUSE_EXISTING_COMPLETE_GPU_CAL_NOT_NEW_CAL_EXECUTION',
        source=str(proof),source_sha256=p['existing_CAL_sha256'],reference_rows=20237,new_CAL_rows=0,
        model_checkpoint_unchanged=True,draft_adapter_source_unchanged=True,target_RNG_unchanged=True,
        correction_scope='unwrap observation function globals only; no draft arithmetic change'))
    return predictor,parity
