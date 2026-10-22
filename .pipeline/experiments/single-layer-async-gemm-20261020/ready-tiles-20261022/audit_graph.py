"""CPU审查真实graph header的任务覆盖/依赖；不代替GPU内存序或数值验证。"""
from pathlib import Path
import subprocess,json,random,heapq,re
B=Path(__file__).resolve().parent
R=Path('/rjs/cxz/论文探索/算子研究/wam_ops/optimized')
source='''#include <iostream>
#include "ready_graph.h"
int main(){for(int rotary=0;rotary<2;rotary++)for(int affine=0;affine<2;affine++)
for(int self:{96,168}){std::cout<<rotary<<" "<<affine<<" "<<self<<"\\n";
for(int n=0;n<28;n++){auto s=ready_node(n,7168,128,self,rotary,affine);std::cout<<s.dependencies<<" "<<s.tiles<<"\\n";}}
for(int p=0;p<28;p++)std::cout<<ready_priority(p)<<" ";}
'''
(B/'graph_dump.cpp').write_text(source)
subprocess.run(['g++','-std=c++17','-O2','-I'+str(R/'csrc'),str(B/'graph_dump.cpp'),'-o',str(B/'graph_dump')],check=True,timeout=30)
lines=subprocess.check_output([str(B/'graph_dump')],text=True).splitlines();priority=list(map(int,lines[-1].split()));assert sorted(priority)==list(range(28))
rows=[]
for case in range(8):
 start=case*29;rotary,affine,selftiles=map(int,lines[start].split());spec=[tuple(map(int,x.split())) for x in lines[start+1:start+29]]
 for workers in [1,4,108]:
  rng=random.Random(20261022+case+workers);nxt=[0]*28;done=[0]*28;mask=0;flags=[0]*48;pending=[];visited=set();now=0;filler=0;serial=0
  def claim():
   for n in priority:
    dep,count=spec[n]
    if mask & (1<<n) or mask & dep != dep: continue
    limit=48 if n==26 else count
    if nxt[n]<limit:
     t=nxt[n];nxt[n]+=1;return n,t
    if n==26:
     for t in range(48):
      if flags[t]==1:flags[t]=2;return n,t+48
   return None
  while mask!=(1<<28)-1:
   while len(pending)<workers:
    item=claim()
    if item is None:break
    n,t=item;dep,count=spec[n];assert mask&dep==dep and item not in visited;visited.add(item)
    if n==26 and t>=48:assert flags[t-48]==2
    if n in (15,16) and any(z[2] not in (15,16) for z in pending):filler+=1
    serial+=1;heapq.heappush(pending,(now+rng.uniform(.01,20),serial,n,t))
   assert pending,'DEADLOCK'
   now,_,n,t=heapq.heappop(pending)
   if n==26 and t<48:assert flags[t]==0;flags[t]=1
   done[n]+=1
   if done[n]==spec[n][1]:mask|=1<<n
  assert not pending and all(done[n]==spec[n][1] for n in range(28))
  assert len(visited)==sum(x[1] for x in spec) and flags==[2]*48
  rows.append(dict(rotary=bool(rotary),affine=bool(affine),self_tiles=selftiles,workers=workers,tiles=len(visited),overlap_opportunities=filler,pass_all=True))
# 直接比对抽取前后的tile数学体，不靠函数名称推断。
old=(B/'before/csrc/whole_layer.cu').read_text();new=(R/'csrc/whole_layer.cu').read_text()
p0=old.index('__device__ void point(');p1=old.index('__device__ void insert_cache',p0);frag=old[p0:p1];body=frag[frag.index('        if (op == TABLE'):frag.index('        // 此阶段只使用线程私有')];assert body in new
p0=old.index('__device__ __noinline__ void action_ffn1_wide(');p1=old.index('#include "norm_device_generated.cuh"',p0);frag=old[p0:p1];body=frag[frag.index('        const int rm'):frag.rindex('    }\n    g.sync();')];assert body in new
oldfa=(B/'before/csrc/fa2_stage.cuh').read_text();newfa=(R/'csrc/fa2_stage.cuh').read_text();p0=oldfa.index('    for (int task = 2 * int(blockIdx.x)');p1=oldfa.index('__device__ __forceinline__ void attention(',p0);frag=oldfa[p0:p1];body=frag[frag.index('        if (cross) {'):frag.rindex('    }\n    g.sync();')];assert body in newfa
# CUDA reachable tile helpers里没有grid同步；单kernel入口只有启动时一次。
ready=(R/'csrc/ready_tiles.cuh').read_text();assert ready.count('    g.sync();')==1
result=dict(status='CPU_GRAPH_AND_SOURCE_AUDIT_PASS',cases=rows,source_body_preserved=['point','ffn1','dual_attention_inner'],GPU_memory_order_validated=False,GPU_numeric_validated=False,GPU_performance_measured=False,limitations='离散事件模型只验证图和逻辑任务覆盖；不模拟CUDA内存模型/原子竞争/计算结果，也不是GPU加速预测。')
(B/'CPU-AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='cases'},ensure_ascii=False));print('schedules',len(rows),'tile_counts',sorted({r['tiles'] for r in rows}))
