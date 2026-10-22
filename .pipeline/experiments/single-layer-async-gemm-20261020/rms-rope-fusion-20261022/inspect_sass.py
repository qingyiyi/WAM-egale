from pathlib import Path
import subprocess,collections,re,json
B=Path(__file__).resolve().parent
binary=next(B.glob('wam_whole_fa2_cuda*.so'))
counts=collections.Counter();functions=0
p=subprocess.Popen(['/usr/local/cuda-12.4/bin/cuobjdump','--dump-sass',str(binary)],stdout=subprocess.PIPE,text=True)
for line in p.stdout:
 if 'Function :' in line:functions+=1
 m=re.search(r'/\*[0-9a-f]+\*/\s+(?:@!?P\d+\s+)?([A-Z][A-Z0-9_.]*)',line)
 if m:counts[m.group(1).split('.')[0]]+=1
rc=p.wait();assert rc==0
selected={k:v for k,v in counts.items() if k in ['HMMA','DMMA','FADD','FMUL','FFMA','DADD','DMUL','DFMA','MUFU','SHFL','LDG','STG','LDS','STS']}
result=dict(status='STATIC_SASS_READ',binary=str(binary),functions=functions,instruction_occurrences=selected,boundary='static binary-wide occurrences, not dynamic counts, not stage runtime or utilization')
(B/'SASS-INSTRUCTION-KINDS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
