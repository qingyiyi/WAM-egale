"""FastWAM empirical-acceptance COST SIMULATION; no trained draft / quality claim.

Reference: fixed WAM PositionTransformer (width64, heads4, FFN256, one layer).
Changes:32 action positions, time inputs from FastWAM's actual BF16 ten-step grid,
identity affine normalizer and deterministic untrained small residual cost proxy.
Acceptance depths and branch indices are supplied by the preserved Long record.
"""
from collections import Counter
import torch
from torch import nn

class DraftCostProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(17,64)
        self.position=nn.Parameter(torch.empty(1,32,64))
        nn.init.normal_(self.position,std=.02)
        self.block=nn.TransformerEncoderLayer(64,4,256,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.head=nn.Linear(64,14)
        self.register_buffer('feature_mean',torch.zeros(17))
        self.register_buffer('feature_std',torch.ones(17))
        self.register_buffer('residual_mean',torch.zeros(7))
        self.register_buffer('residual_std',torch.full((7,),.01))
    def forward(self,state,previous,timestep,delta):
        assert state.shape==(1,32,7) and previous.shape==state.shape
        sigma=timestep.float()/1000.
        scalars=torch.stack((sigma,sigma+delta.float(),sigma)).reshape(1,1,3).expand(1,32,3)
        x=torch.cat((state.float(),previous.float(),scalars),-1)
        x=(x-self.feature_mean)/self.feature_std
        raw=self.head(self.block(self.embed(x)+self.position))
        center,spread=raw[...,:7],raw[...,7:]
        proposals=torch.stack((center+spread,center-spread),1)
        return (previous.float()[:,None]+proposals*self.residual_std+self.residual_mean).to(state.dtype)[0]

class EmpiricalReplay:
    def __init__(self,model,draft):
        self.model=model;self.draft=draft;self.trace=[];self.validate=False;self.audit_records=[]
    def set_chunk(self,trees):
        self.trace=trees;self.counts=Counter();self.rounds=[];self.used=0
    def target(self,states,timesteps,ctx,mask,cache_k,cache_v,attn,fn,compiled):
        b=states.shape[0]
        assert b in (1,7) and states.shape[1:]==(32,7)
        if compiled:torch.compiler.cudagraph_mark_step_begin()
        self.counts['target_calls']+=1;self.counts[f'target_B{b}_calls']+=1;self.counts['target_state_rows']+=b
        return fn(latents_action=states,timestep_action=timesteps,
                  context=ctx.expand(b,-1,-1),context_mask=mask.expand(b,-1),
                  video_cache_k=[x.expand(b,-1,-1) for x in cache_k],
                  video_cache_v=[x.expand(b,-1,-1) for x in cache_v],
                  action_attention_mask=attn).clone()
    def loop(self,state,timesteps,deltas,ctx,mask,cache_k,cache_v,attn,fn,compiled):
        assert len(timesteps)==len(deltas)==10 and state.shape==(1,32,7)
        step=self.model.infer_action_scheduler.step
        i=0;previous=None
        while i<10:
            remaining=10-i
            if previous is None or remaining<3 or self.used==len(self.trace):
                v=self.target(state,timesteps[i:i+1],ctx,mask,cache_k,cache_v,attn,fn,compiled)
                state=step(v,deltas[i],state);previous=v;i+=1
                self.counts['committed_updates']+=1;self.counts['ordinary_target_updates']+=1
                self.counts['bootstrap_updates' if i==1 else 'boundary_target_updates']+=1
                continue
            entry=self.trace[self.used];self.used+=1
            depth,parent,child=int(entry['depth']),int(entry['parent']),int(entry['child'])
            assert depth in (0,1,2) and parent in (0,1) and child in (0,1)
            dirs1=self.draft(state,previous,timesteps[i],deltas[i]);self.counts['draft_calls']+=1
            parents=step(dirs1,deltas[i],state.expand(2,-1,-1))
            dirs2=torch.stack([self.draft(parents[j:j+1],dirs1[j:j+1],timesteps[i+1],deltas[i+1]) for j in range(2)])
            self.counts['draft_calls']+=2
            leaves=step(dirs2.reshape(4,32,7),deltas[i+1],parents[:,None].expand(-1,2,-1,-1).reshape(4,32,7))
            packed=torch.cat((state,parents,leaves),0)
            times=torch.cat((timesteps[i:i+1],timesteps[i+1:i+2].expand(2),timesteps[i+2:i+3].expand(4)))
            versions=[x._version for x in cache_k+cache_v]
            values=self.target(packed,times,ctx,mask,cache_k,cache_v,attn,fn,compiled)
            # FP32 acceptance scores are actually computed and transferred, not skipped.
            first=((dirs1.float()-values[0:1].float()).flatten(1).norm(dim=1)*deltas[i].float().abs())
            second=((dirs2.float()-values[1:3,None].float()).flatten(2).norm(dim=2)*deltas[i+1].float().abs())
            scores=torch.cat((first,second.flatten())).tolist()
            assert all(__import__('math').isfinite(x) for x in scores)
            assert [x._version for x in cache_k+cache_v]==versions
            if self.validate and not self.audit_records:
                self.audit(packed,times,ctx,mask,cache_k,cache_v,attn,fn,compiled,values,torch.cat((deltas[i:i+1],deltas[i+1:i+2].expand(2),deltas[i+2:i+3].expand(4))))
            if depth==0:
                state=step(values[0:1],deltas[i],state);previous=values[0:1]
                self.counts['rejected_trees']+=1;self.counts['fallback_updates']+=1
            elif depth==1:
                state=step(values[1+parent:2+parent],deltas[i+1],parents[parent:parent+1]);previous=values[1+parent:2+parent]
                self.counts['draft_updates']+=1;self.counts['fallback_updates']+=1
            else:
                leaf=3+2*parent+child
                state=step(values[leaf:leaf+1],deltas[i+2],leaves[2*parent+child:2*parent+child+1]);previous=values[leaf:leaf+1]
                self.counts['draft_updates']+=2;self.counts['bonus_updates']+=1
            self.counts['tree_count']+=1;self.counts[f'depth_{depth}']+=1
            self.counts['candidate_expansions']+=6;self.counts['committed_updates']+=depth+1
            self.rounds.append(dict(step=i,depth=depth,parent=parent,child=child,g=depth+1,scores=scores))
            i+=depth+1
        assert self.counts['committed_updates']==10 and self.used==len(self.trace),(self.counts,self.used,len(self.trace))
        return state
    def audit(self,states,times,ctx,mask,ks,vs,attn,fn,compiled,batch,deltas):
        # Outside every measured policy: same seven states serial vs batch,
        # candidate isolation and read-only fixed condition/cache.
        old_counts=self.counts.copy()
        snapshots=[x.clone() for x in ks+vs]+[ctx.clone(),mask.clone()]
        serial=torch.cat([self.target(states[j:j+1],times[j:j+1],ctx,mask,ks,vs,attn,fn,compiled) for j in range(7)],0)
        diff=(batch.float()-serial.float());rel=float(diff.norm()/serial.float().norm().clamp_min(1e-6));maxdiff=float(diff.abs().max())
        limit=.05+.02*float(serial.float().abs().max())
        row=dict(check='B7_vs_serial_B1',relative_L2=rel,max_abs=maxdiff,relative_L2_limit=.01,max_abs_limit=limit,pass_all=rel<=.01 and maxdiff<=limit)
        self.audit_records.append(row);assert row['pass_all'],row
        for j in range(7):
            ref=serial[j].float();delta=(batch[j].float()-ref)
            r=float(delta.norm()/ref.norm().clamp_min(1e-6));a=float(delta.abs().max());limit=.05+.02*float(ref.abs().max())
            batched_update=self.model.infer_action_scheduler.step(batch[j],deltas[j],states[j])
            serial_update=self.model.infer_action_scheduler.step(serial[j],deltas[j],states[j])
            update_diff=float((batched_update.float()-serial_update.float()).abs().max())
            # Quantized scheduler arithmetic: one BF16 ulp envelope for multiply/add,
            # in addition to propagated observed direction discrepancy.
            update_limit=float(deltas[j].float().abs())*a+2*torch.finfo(torch.bfloat16).eps*max(1.,float(serial_update.float().abs().max()))
            per_row=dict(check='candidate_direction_and_scheduler_update',candidate=j,relative_L2=r,max_abs=a,relative_L2_limit=.01,max_abs_limit=limit,update_max_abs=update_diff,update_limit=update_limit,pass_all=r<=.01 and a<=limit and update_diff<=update_limit)
            self.audit_records.append(per_row);assert per_row['pass_all'],per_row
        mutated=states.clone();mutated[6]+=torch.tensor(.125,device=states.device,dtype=states.dtype)
        other=self.target(mutated,times,ctx,mask,ks,vs,attn,fn,compiled)
        unchanged=torch.equal(batch[:6],other[:6])
        condition_unchanged=all(torch.equal(a,b) for a,b in zip(snapshots,ks+vs+[ctx,mask]))
        row=dict(check='candidate_isolation_and_fixed_condition',other_six_exact=unchanged,condition_unchanged=condition_unchanged,pass_all=unchanged and condition_unchanged)
        self.audit_records.append(row);assert row['pass_all'],row
        self.counts=old_counts
