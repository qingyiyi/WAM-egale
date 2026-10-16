"""Sealed stage caps and RAM budget math from the executed GPU revision."""
import time

STAGES = {'kernel':0,'capture':52,'resource_probe':1,'equivalence_HC':29,'equivalence_S':29,
          'warmup_R':29,'warmup_S':29,
          **{f'OFF_{b}_{a}':29 for b in range(6) for a in ('R','S')},
          **{f'invalidation_{i:02d}':5 for i in range(1,13)},
          **{f'isolation_{a}':2 for a in ('R','HC','S')},'finish':2}
assert sum(STAGES.values()) == 585

class Ledger:
    """Bound each stage before attempted public target math, no refunds/retries."""
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.deadline=None;self.reserved=0;self.attempted=0
        self.seen=set();self.stage=None;self.stage_attempted=0;self.completed=0
    def begin(self,key):
        if key not in STAGES or key in self.seen:raise ValueError('unknown/retried stage')
        self.seen.add(key);self.reserved+=STAGES[key];self.stage=key;self.stage_attempted=0
        if self.reserved>585:raise RuntimeError('allocated budget exhausted')
    def before_forward(self):
        if self.deadline is not None and self.clock()>=self.deadline:raise TimeoutError('GPU deadline before forward')
        if self.stage not in STAGES or self.stage_attempted>=STAGES[self.stage] or self.attempted>=715:raise RuntimeError('target forward cap')
        self.stage_attempted+=1;self.attempted+=1
    def after_forward(self):self.completed+=1
    def summary(self):
        return dict(target_reserved=self.reserved,target_attempted=self.attempted,
                    target_completed=self.completed,stage=self.stage,stage_attempted=self.stage_attempted,
                    allocated_cap=585,target_hard_cap=715,extra_micro_calls=0,
                    EGL=0,complete_episodes=0,complete_policy_chunks=0,environment_steps=0)
