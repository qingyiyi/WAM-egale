from pathlib import Path
import os,json,statistics
B=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(B/'plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
r=json.loads((B/'replay-fix/PROFILE-1.json').read_text());row=next(x for x in r['rows'] if x['case']=='middle')
raws=row['full_timelines']
def span(r):return max(r[2048+i*4+3] for i in range(108))-min(r[2048+i*4] for i in range(108))
median=statistics.median(span(r) for r in raws);idx=min(range(len(raws)),key=lambda i:abs(span(raws[i])-median));raw=raws[idx]
f1=[raw[i*8:(i+1)*8] for i in range(112)];f2=[raw[896+i*12:896+(i+1)*12] for i in range(96)];ctas=[raw[2048+i*4:2048+(i+1)*4] for i in range(108)];t0=min(x[0] for x in ctas)
fig,ax=plt.subplots(figsize=(12,7));colors={'FFN1 compute':'#3067ad','FFN1 epilogue':'#81b1e2','FFN2 compute':'#df842c','FFN2 commit wait':'#b62d49','FFN2 epilogue':'#f4bf77','Exit barrier wait':'#dddddd'}
def bar(cta,a,b,color):ax.broken_barh([((a-t0)/1000,(b-a)/1000)],(cta-.4,.8),facecolors=color,linewidth=0)
for i,x in enumerate(ctas):bar(i,x[2],x[3],colors['Exit barrier wait'])
for x in f1:
 bar(x[5],x[1],x[2],colors['FFN1 compute']);bar(x[5],x[2],x[3],colors['FFN1 epilogue'])
for x in f2:
 bar(x[6],x[1],x[2],colors['FFN2 compute']);bar(x[6],x[2],x[3],colors['FFN2 commit wait']);bar(x[6],x[3],x[4],colors['FFN2 epilogue'])
for g in range(4):
 t=(max(x[4] for x in f1[g*28:(g+1)*28])-t0)/1000
 ax.axvline(t,color='black',linestyle=':',alpha=.4)
 if g in [1,3]:ax.text(t+2,112,f'Group {g} ready',fontsize=9)
ax.set(xlabel='Time from FFN entry (microseconds)',ylabel='Resident CTA ID',ylim=(-2,118),title='Current R2 FFN: real B14 middle-KV input (instrumented execution)')
ax.invert_yaxis();ax.legend(handles=[Patch(color=c,label=k) for k,c in colors.items()],loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,frameon=False)
fig.text(.01,.01,f'Process 1, sample {idx}: selected nearest median FFN span among 12 samples. Instrumented timeline; not formal kernel timing.',fontsize=9)
fig.tight_layout(rect=[0,.08,1,1]);fig.savefig(B/'FFN-TIMELINE.png',dpi=150);fig.savefig(B/'FFN-TIMELINE.svg')
(B/'PLOT-MANIFEST.json').write_text(json.dumps({'process':1,'case':'middle','sample_index':idx,'selection':'closest to median FFN span among all 12; no quality/performance selection'},indent=2)+'\n')
