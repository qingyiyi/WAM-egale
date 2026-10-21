from pathlib import Path
import json,hashlib,shutil
B=Path(__file__).resolve().parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
expected={'csrc/whole_layer.cu': 'df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a', 'csrc/cutlass_gemm.cuh': 'cf30211ece6846416e966a2c59acab58a6cd11569ce6bed5e38338aeaee97fec', 'cutlass_gemm_codegen.py': 'b6c7b0fa799d7b8cb1ea341560196d4bfd3fb75bafc2ee4c1e58f70f31db8b6d', 'build_whole_layer.py': '19fd44bf58f1ecb7ea67216700470e0d64d5ebd8d8d7ba26ac79b9886b32b763'}
assert not (B/'before').exists()
for f,h in expected.items():
 assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h, f
(B/'before').mkdir()
for f in expected: shutil.copy2(R/f,B/'before'/Path(f).name)
for p in R.glob('wam_whole_fa2_cuda*.so'):shutil.copy2(p,B/'before'/p.name)
assert list((B/'before').glob('wam_whole_fa2_cuda*.so'))
for f in expected:shutil.copy2(B/'candidate'/Path(f).name,R/f)
(B/'SOURCE-MANIFEST.json').write_text(json.dumps(dict(before=expected,candidate={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in expected}),indent=2)+'\n')
