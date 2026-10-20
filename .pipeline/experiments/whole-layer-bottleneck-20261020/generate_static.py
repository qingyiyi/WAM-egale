"""One diagnostic ablation of task allocation; production source is frozen."""
import hashlib
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SRC=ROOT/'算子研究/wam_ops/optimized/csrc'
spec=importlib.util.spec_from_file_location('old_profile_generator',HERE.parent/'single-layer-warp-specialized-20261019/generate_profile.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
raw=(SRC/'whole_layer.cu').read_bytes()
assert hashlib.sha256(raw).hexdigest()==old.SHA
text=old.instrument(raw.decode())
text=old.replace(text,'    grid.sync(); if(grid.thread_rank()==0)',
    '    if(threadIdx.x==0)s.task=int(blockIdx.x)-int(gridDim.x);\n    grid.sync(); if(grid.thread_rank()==0)')
text=old.replace(text,'s.task=atomicAdd(p.queues+st,1);','s.task+=int(gridDim.x);')
text=text.replace('wam_whole_layer_v1','wam_whole_m38_static')
text=text.replace('wam_optimized_norm','wam_norm_m38_static').replace('"norm.cuh"','"norm_static.cuh"')
(HERE/'static.cu').write_text(text)
(HERE/'norm_static.cuh').write_text((SRC/'norm.cuh').read_text().replace('wam_optimized_norm','wam_norm_m38_static'))
(HERE/'LICENSE.pytorch').write_bytes((SRC/'LICENSE.pytorch').read_bytes())
(HERE/'SOURCE-MANIFEST.json').write_text(json.dumps(dict(original_whole_sha256=old.SHA,
    changes=['same stage globaltimer instrumentation as M34','per-stage atomic queue replaced by CTA grid stride'],
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.iterdir()
           if p.is_file() and p.suffix in ('.cu','.cuh','.py')}),indent=2)+'\n')
