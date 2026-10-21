from pathlib import Path
import hashlib,json,shutil
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
s=(R/'csrc/whole_layer.cu').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='0b7779635bfd79fc163abe44acd7682c92196d49ccef6d0527ddd5f4ce300aee'
assert not (B/'before').exists()
(B/'before').mkdir()
for name in ('whole_layer.cu','cutlass_gemm.cuh'):shutil.copy2(R/'csrc'/name,B/'before'/name)
for p in R.glob('wam_whole_fa2_cuda*.so'):shutil.copy2(p,B/'before'/p.name)
s=s.replace('#include "norm_device_generated.cuh"',(B/'candidate-functions.txt').read_text()+'#include "norm_device_generated.cuh"')
old='action_ffn_linear<64, D, F>(p, s, g, st++, bp(p, ACT), 25, bp(p, F2));'
assert s.count(old)==1
s=s.replace(old,'action_ffn2_parallel(p, s, g, st++);')
(R/'csrc/whole_layer.cu').write_text(s)
(B/'tested').mkdir()
(B/'tested/whole_layer.cu').write_text(s)
(B/'SOURCE-MANIFEST.json').write_text(json.dumps(dict(before_SHA256='0b7779635bfd79fc163abe44acd7682c92196d49ccef6d0527ddd5f4ce300aee',candidate_SHA256=hashlib.sha256(s.encode()).hexdigest(),changes='Only FFN2 parallel K in specialized stage; older failed generic-linear variant not reused as current baseline'),indent=2)+'\n')
