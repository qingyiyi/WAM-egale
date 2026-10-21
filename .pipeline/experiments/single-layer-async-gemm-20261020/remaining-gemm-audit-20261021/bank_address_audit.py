"""Enumerate the pinned CUTLASS accumulator store mapping; compare existing NCU.
This is CPU address analysis, not a new GPU measurement or all-input proof.
"""
from pathlib import Path
import collections,json,hashlib
B=Path(__file__).resolve().parent
P=B.parent/'projection-gemm-core-20261021'
result={'scope':'SM80 measured independent context K/V, 672 CTAs, 8 MMA warps, tile256x128; STS.64. Whole-layer conflict counter not measured here.','bank_word_bytes':4,'banks':32,'ideal_transactions_per_warp_STS64':2,'details':{}}
for stride in [128,136]:
 wavefronts=0;ideal=0;examples=[]
 for warp in range(8):
  for mma_n in range(8):
   for mma_m in range(4):
    for row_offset in [0,8]:
     for half in range(2):
      banks=collections.defaultdict(set)
      for lane in range(16*half,16*half+16):
       row=64*(warp%4)+16*mma_m+row_offset+(lane>>2)
       col=64*(warp//4)+8*mma_n+2*(lane&3)
       for e in range(2):
        word=row*stride+col+e;banks[word%32].add(word)
       if warp==mma_n==mma_m==row_offset==half==0:
        examples.append({'lane':lane,'row':row,'columns':[col,col+1],'banks':[(row*stride+col)%32,(row*stride+col+1)%32]})
      wavefronts+=max(map(len,banks.values()));ideal+=1
 result['details'][str(stride)]={'per_CTA_wavefronts':wavefronts,'per_CTA_ideal':ideal,'per_projection_wavefronts':wavefronts*672,'per_projection_excess':(wavefronts-ideal)*672,'first_halfwarp':examples}
assert result['details']['128']['per_projection_wavefronts']==2752512
assert result['details']['136']['per_projection_wavefronts']==688128
assert result['details']['128']['per_projection_excess']==2064384
observed=[]
for label,path in [('before',P/'local-ncu/SUMMARY.json'),('padding',P/'pad136-local-ncu/SUMMARY.json')]:
 data=json.loads(path.read_text())
 for key,value in data.items():
  if not key.startswith('measure_own_'):continue
  metrics=value['metrics'];conflict=int(metrics['l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum']['value']);wave=int(metrics['l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum']['value'])
  expected=result['details']['128' if label=='before' else '136']
  assert conflict==expected['per_projection_excess'] and wave==expected['per_projection_wavefronts']
  observed.append({'variant':label,'kernel':key,'store_conflicts':conflict,'store_wavefronts':wave,'address_count_matches':True,'source':str(path.relative_to(B.parent))})
result['NCU_comparison']=observed
result['limitations']=['NCU store-conflict zero is specific to these K/V kernels and inputs. Shared-load conflicts remain nonzero.','An STS.64 warp has two ideal 128-byte transactions; repetitions across these transactions are not counted as bank conflicts.','Padding changes temporary physical stride, not logical output or arithmetic. Vectorized reading is a separate subsequent change.']
(B/'BANK-ADDRESS-AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:{f:v[f] for f in ['per_projection_wavefronts','per_projection_excess']} for k,v in result['details'].items()})
print(len(observed),'measured K/V rows matched')
