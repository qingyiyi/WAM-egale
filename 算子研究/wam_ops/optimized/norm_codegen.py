"""Embed pinned LayerNorm device bodies in the existing four-warp compute group.

Only logical coordinates, barrier scope, shared-storage pointer and row selection
change. Preserve the FP32 input/output arithmetic used by FP32LayerNorm.
"""
from pathlib import Path
import hashlib,json,re


def generate(output):
    source=(Path(__file__).parent/'csrc/norm.cuh').read_text()
    a=source.index('template<typename T, bool rms_norm = false>')
    b=source.index('template <typename T, typename T_ACC',a)
    stats=source[a:b].replace('compute_stats(', 'whole_compute_stats(')
    a=b
    b=source.index('template <typename T, typename T_ACC',a+1)
    forward=source[a:b].replace('vectorized_layer_norm_kernel_impl(', 'whole_norm_forward(')
    forward=forward.replace('T* Y){','T* Y, float* s_data){')
    forward=forward.replace('extern __shared__ float s_data[];', '// Shared scratch supplied by the whole-layer tile.')
    forward=forward.replace('auto i1 = blockIdx.x;', 'auto i1 = 0;')
    forward=forward.replace('compute_stats<T, rms_norm>', 'whole_compute_stats<T, rms_norm>')
    # Mean/rstd are not externally materialized in the whole-block contract.
    forward=forward.replace('mean[i1] = wd.mean;', 'if (mean) mean[i1] = wd.mean;')
    forward=forward.replace('rstd[i1] = rstd_val;', 'if (rstd) rstd[i1] = rstd_val;')
    text=stats+'\n'+forward
    text=text.replace('blockDim.x','32').replace('blockDim.y','4')
    text=text.replace('threadIdx.x','(ct() % 32)').replace('threadIdx.y','(ct() / 32)')
    text=text.replace('__syncthreads();','group_sync();')
    text='// Derived pinned Torch LayerNorm; see norm.cuh license.\n#pragma once\nusing namespace wam_optimized_norm;\n'+text
    output.mkdir(parents=True,exist_ok=True)
    (output/'norm_device_generated.cuh').write_text(text)
    (output/'NORM-SOURCE.json').write_text(json.dumps(dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),generated_sha256=hashlib.sha256(text.encode()).hexdigest(),changes='logical four-warp coordinates/barrier/shared pointer only; FP32 body preserved'),indent=2)+'\n')
