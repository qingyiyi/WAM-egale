"""Independent oracle verifies real commits/cost counts at every acceptance depth."""
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from real_speculative import RealSpeculative, choose

torch.set_num_threads(2)
ts = torch.tensor([1000,980,952,920,884,832,768,680,556,358], dtype=torch.bfloat16)
ds = torch.tensor([-.021728515625,-.02587890625,-.03125,-.038818359375,-.049072265625,
                   -.06396484375,-.08740234375,-.1259765625,-.1982421875,-.357421875],dtype=torch.bfloat16)


def direction(t):
    return (t.float()/1000).to(torch.bfloat16).reshape(-1,1,1).expand(-1,32,7).clone()


def step(v,d,x):
    return x+v*d.to(x.dtype)


class Draft:
    def __init__(self, depth): self.depth=depth;self.calls=0
    def __call__(self,x,previous,t,d):
        assert previous.shape == x.shape
        offset = self.calls%3
        self.calls += 1
        out=direction(t.reshape(1)).expand(2,-1,-1).clone()
        if self.depth==0 or (self.depth==1 and offset>0): out+=10
        return out


def target(**kw):
    return direction(kw['timestep_action'])


ref=torch.zeros(1,32,7,dtype=torch.bfloat16)
for t,d in zip(ts,ds): ref=step(direction(t.reshape(1)),d,ref)
results=[]
for depth in (0,1,2):
    model=SimpleNamespace(device='cpu',infer_action_scheduler=SimpleNamespace(step=step))
    spec=RealSpeculative(model,Draft(depth),.01)
    value=spec.loop(torch.zeros_like(ref),ts,ds,torch.zeros(1,2,3),torch.ones(1,2),[],[],None,target,False)
    assert torch.equal(value,ref)
    c=spec.counts
    assert c['committed_updates']==10 and sum(r['g'] for r in spec.rounds)==10
    trees=[r for r in spec.rounds if r['kind']=='tree']
    assert all(r['depth']==depth for r in trees)
    assert c['target_state_rows']==7*c['tree_count']+c['target_B1_calls']
    assert c['candidate_expansions']==6*c['tree_count']
    assert c['draft_calls']==3*c['tree_count']
    assert all(r['step']+r['g']<=10 for r in spec.rounds)
    assert c['bootstrap_updates']==1
    results.append(dict(depth=depth,counts=dict(c),rounds=spec.rounds,oracle_final_exact=True))
# Explicit depth-first and cost tie cases: smaller invalid-prefix total cannot win.
assert choose([.2,.4,2,2,.3,.9],1.)==(2,1,0)
assert choose([.2,.4,2,2,2,2],1.)==(1,0,0)
assert choose([2,3,.2,.3,.1,.2],1.)==(0,0,0)
assert choose([.2,.2,.3,.3,.3,.3],1.)==(2,0,0)
p=Path(__file__).resolve().parent.parent/'CPU-BOUNDARIES.json'
p.write_text(json.dumps(dict(status='PASS_FINITE_ORACLE_CHECKS',cases=results,
    limitations='CPU oracle only; real target B1/B7 numerical checks required on GPU'),indent=2)+'\n')
print('PASS actual ten commits, bootstrap, tail, rejection/fallback/bonus; 3 depths and 4 selector cases')
