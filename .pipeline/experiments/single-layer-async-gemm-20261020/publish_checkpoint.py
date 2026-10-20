"""Publish this completed round into main; omit runtime binaries/tensors/secrets."""
from pathlib import Path
from datetime import datetime
import hashlib,json,re
P=Path(__file__).resolve().parent;R=P.parents[2];M=R/'review_exports/WAM-egale-libero-fullscale-20261017'
files=set()
for p in P.rglob('*'):
 if not p.is_file() or p.suffix not in ('.md','.json','.py','.cu','.cuh','.csv','.log'):continue
 if p.name in ('publish_checkpoint.py','DEPLOYMENT.json','PUBLISH-CHECKPOINT.json','update_hw.py'):continue
 if p.name.endswith('-TRACE.json'):continue
 if p.stat().st_size>2*1024**2:raise RuntimeError(f'oversize:{p}')
 files.add(p)
for n in ('csrc/gemm_pipeline.cuh','pipeline_codegen.py','build_pipeline.py','whole_layer_pipeline.py','PIPELINE.md'):
 files.add(R/'算子研究/wam_ops/optimized'/n)
for n in ('INDEX.md','PROGRESS.compact.md','memory/INDEX.md','experiments/INDEX.md'):files.add(R/'.pipeline'/n)
C=R/'.pipeline/cycles/wam-research-20261016'
for n in ('PLAN.md','PROGRESS.md','EXECUTION.md','DISCUSSION-SUMMARY.md'):
 files.add(C/n);files.add(C/'history/M39-start-20261020'/n)
files.add(C/'history/M39-start-20261020/PROGRESS.compact.md')
for n in ('decision-依据profile继续单层tile流水优化.md','decision-直接在optimized迭代且不另建登记.md','requirement-算子交付明确源码入口与验证范围.md'):
 files.add(R/'.pipeline/memory/global/requirements'/n)
manifest=[]
for p in sorted(files):
 assert not p.is_symlink()
 b=p.read_bytes();assert not re.search(rb'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|\bgh[pousr]_[A-Za-z0-9]{20,}',b),str(p)
 rel=p.relative_to(R);out=M/rel;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b)
 manifest.append(dict(path=str(rel),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
section='''<!-- m39:start -->
## 10/20 M39：完整单层双缓冲GEMM

[结果](../.pipeline/experiments/single-layer-async-gemm-20261020/RESULT.md)、[源码与流水](../算子研究/wam_ops/optimized/PIPELINE.md)：九真实签名、18合法KV转移匹配旧失败whole；native对Torch9/9逐位通过，新async对Torch9/9逐位FAIL。两个独立进程432完整block样本，对同次旧whole耗时下降17.9%–36.3%，仍慢于native3.33–7.84倍。不是已超过Torch或机器人E2E。

sync同LD40对照分离padding影响；GEMM阶段快1.73–1.95倍，长KV self-attention现占35%–49%。仍一层/单kernel，新增流水仅在GEMM K块内；无跨stage ready队列，NCU流量/stall未知，原四false保持。

675block、264.838秒，六owned进程退出，两卡空。**资源审计FAILED**：书面PLAN/runner900而机器registration误留240，实际超后者435；原记录保留不改。[清理/失败](../.pipeline/experiments/single-layer-async-gemm-20261020/RESOURCE-AND-STOP-REPORT.md)。用户最新要求后续直接改optimized，不为另一agent带读另建版本/登记；更大tile仅建议尚未执行。下面M38及更早段落是历史。
<!-- m39:end -->

'''
p=M/'review/CURRENT-20261020.md';s=p.read_text();s=re.sub(r'<!-- m39:start -->.*?<!-- m39:end -->\n*','',s,flags=re.S);title,_,rest=s.partition('\n');p.write_text(title+'\n\n'+section+rest.lstrip('\n'))
p=M/'review/START_HERE.md';s=p.read_text()
if '<!-- m39-entry -->' not in s:
 title,_,rest=s.partition('\n');p.write_text(title+'\n\n<!-- m39-entry -->\n最新[M39双缓冲/资源审计](CURRENT-20261020.md)在顶部；原Torch数值NO-GO与四false不改，GPU已清理。\n\n'+rest)
record=dict(at=datetime.now().astimezone().isoformat(),files=manifest,derived=['review/CURRENT-20261020.md','review/START_HERE.md'],branch='main',scientific_gate_changes=False,exclusions=['raw Chrome traces','DEPLOYMENT/SSH','weights/tensors/KV/DSOs/toolchains'])
(M/'review/M39-PUBLICATION-MANIFEST.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
spec=R/'.tmp/publish-m39-20261020.paths';spec.write_text('\n'.join([x['path'] for x in manifest]+record['derived']+['review/M39-PUBLICATION-MANIFEST.json'])+'\n')
print(json.dumps(dict(files=len(manifest)+3,bytes=sum(x['bytes'] for x in manifest),pathspec=str(spec))))
