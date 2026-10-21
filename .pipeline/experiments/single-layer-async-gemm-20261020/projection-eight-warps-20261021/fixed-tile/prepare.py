from pathlib import Path
import json,hashlib,shutil,subprocess,os,time
B=Path(__file__).resolve().parent;P=B.parent
R=Path('/root/cxz/ops/persistent-tile-20261019/workspace/算子研究/wam_ops/optimized')
state=json.loads((P/'STATUS.json').read_text());assert state['status']=='COMPLETE_ATTEMPT'
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True);assert not any(x.startswith('GPU-c3fa8d6f') for x in apps.splitlines())
# Retain rejected large-tile binary and restore selected four-warp source/binary.
(P/'rejected-binary').mkdir()
for x in R.glob('wam_whole_fa2_cuda*.so'):shutil.copy2(x,P/'rejected-binary'/x.name)
for f in ['csrc/whole_layer.cu','csrc/cutlass_gemm.cuh','cutlass_gemm_codegen.py','build_whole_layer.py']:shutil.copy2(P/'before'/Path(f).name,R/f)
for x in (P/'before').glob('wam_whole_fa2_cuda*.so'):shutil.copy2(x,R/x.name)
expected={'csrc/whole_layer.cu': 'df2a307241928567e2b0d80a5e6c601b8e1aedaf6e4713b693d88ce247ead11a', 'csrc/cutlass_gemm.cuh': 'cf30211ece6846416e966a2c59acab58a6cd11569ce6bed5e38338aeaee97fec', 'csrc/gemm_pipeline.cuh': 'bdc8a6d1c534d9ad122e44bbb63401f5b8e52a764bed14bac800c4ee9e90345c', 'cutlass_gemm_codegen.py': 'b6c7b0fa799d7b8cb1ea341560196d4bfd3fb75bafc2ee4c1e58f70f31db8b6d', 'build_whole_layer.py': '19fd44bf58f1ecb7ea67216700470e0d64d5ebd8d8d7ba26ac79b9886b32b763'}
for f,h in expected.items():assert hashlib.sha256((R/f).read_bytes()).hexdigest()==h,f
(B/'before').mkdir()
for f in expected:shutil.copy2(R/f,B/'before'/Path(f).name)
for x in R.glob('wam_whole_fa2_cuda*.so'):shutil.copy2(x,B/'before'/x.name)
for f in expected:shutil.copy2(B/'candidate'/Path(f).name,R/f)
(B/'SOURCE-MANIFEST.json').write_text(json.dumps(dict(before=expected,candidate={f:hashlib.sha256((R/f).read_bytes()).hexdigest() for f in expected}),indent=2)+'\n')
