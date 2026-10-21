from pathlib import Path
import hashlib,json,shutil
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
s=(R/'csrc/whole_layer.cu').read_text()
assert hashlib.sha256(s.encode()).hexdigest()=='0a4db2ef8f1028d0167d9f4dae12f128a6bad24782e5d89aaa0cecdf2cbfdfeb'
assert not (B/'before').exists()
(B/'before').mkdir()
for name in ('whole_layer.cu','cutlass_gemm.cuh'):shutil.copy2(R/'csrc'/name,B/'before'/name)
for p in R.glob('wam_whole_fa2_cuda*.so'):shutil.copy2(p,B/'before'/p.name)
s=s.replace('#include "norm_device_generated.cuh"',(B/'candidate-functions.txt').read_text()+'#include "norm_device_generated.cuh"')
old='action_ffn_linear<128, F, D>(p, s, g, st++, bp(p, M3), 23, bp(p, F1));'
assert s.count(old)==1
s=s.replace(old,'action_ffn1_tail(p, s, g, st++);')
(R/'csrc/whole_layer.cu').write_text(s)
(B/'tested').mkdir()
(B/'tested/whole_layer.cu').write_text(s)
(B/'SOURCE-MANIFEST.json').write_text(json.dumps(dict(before_SHA256='0a4db2ef8f1028d0167d9f4dae12f128a6bad24782e5d89aaa0cecdf2cbfdfeb',candidate_SHA256=hashlib.sha256(s.encode()).hexdigest(),changes='Only FFN1 internal tail tile/assignment organization; no cross-stage ready; original 216 M128 tiles plus 16 M64 tail tiles, same math'),indent=2)+'\n')
