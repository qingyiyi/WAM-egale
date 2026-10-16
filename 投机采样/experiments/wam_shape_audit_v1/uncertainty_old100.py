"""Reproducible exploratory paired uncertainty, not a qualification test."""
import json,math,os,sys
from pathlib import Path
import numpy as np
def cp(k,n,alpha=.025):
 def cdf(j,p):return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(j+1))
 def solve(j,prob):
  lo,hi=0.,1.
  for _ in range(100):
   mid=(lo+hi)/2
   if cdf(j,mid)>prob:lo=mid
   else:hi=mid
  return (lo+hi)/2
 return (0. if k==0 else solve(k-1,1-alpha/2),1. if k==n else solve(k,alpha/2))
def main():
 os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:2]))
 root=Path(sys.argv[1]);x=json.loads((root/'OLD100-DESCRIPTIVE.json').read_text());counts=x['paired_counts'];n=len(x['pairs'])
 wins=cp(counts['spec_win'],n);losses=cp(counts['target_win'],n)
 ci=[wins[0]-losses[1],wins[1]-losses[0]]
 rng=np.random.default_rng(20261016);rep=np.zeros(50000)
 for task in range(10):
  v=np.asarray([int(p['A_OFF_success'])-int(p['T_success']) for p in x['pairs'] if p['task']==task])
  assert v.shape==(5,)
  rep+=v[rng.integers(0,5,size=(50000,5))].mean(1)/10
 boot=np.quantile(rep,[.025,.975]).tolist()
 old=json.loads((root/'OLD100-UNCERTAINTY.json').read_text())
 assert np.allclose(ci,old['iid_pairs_conservative_95_CI'],rtol=0,atol=1e-12)
 assert np.allclose(boot,old['fixed_task_stratified_paired_bootstrap_percentile95'],rtol=0,atol=1e-12)
 out=dict(status='PASS_REPRODUCED_EXPLORATORY_INTERVALS',paired_n=n,paired_counts=counts,iid_conservative95=ci,fixed_task_bootstrap95=boot,replicates=50000,seed=20261016,noninferiority_established=False,original_failed_unchanged=True)
 with (root/'OLD100-UNCERTAINTY-REPRODUCED.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
 print(json.dumps(out))
if __name__=='__main__':main()
