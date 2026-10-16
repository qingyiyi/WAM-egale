"""Generate a private, explicitly changed Flash kernel without editing its source."""
from pathlib import Path
import hashlib,json

OWN=Path(__file__).resolve().parent
SRC=OWN.parent/'shared_prefix_torch_v2/csrc'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=OWN/'csrc';out.mkdir(exist_ok=True)
    source=(SRC/'shared_flash_fwd_kernel.h').read_text()
    source=source.replace('// Generated from pinned original FlashAttention; see ADDRESS-ONLY.diff.\n// Original arithmetic and synchronization are byte-preserved.',
        '// Research derivative of pinned FlashAttention shared address kernel.\n// Group visibility, tile geometry and Check_inf differ. NOT bitwise certified.\n// Original licenses remain under shared_prefix_torch_v2/csrc/vendor.')
    assert source.count('#include "shared_load.h"')==1
    source=source.replace('#include "shared_load.h"','#include "group_load.h"')
    source=source.replace('compute_shared_attn_1rowblock','compute_group_attn_1rowblock')
    source=source.replace('copy_shared_kv','copy_group_kv')
    anchor='    auto thr_mma = tiled_mma.get_thread_slice(tidx);'
    assert source.count(anchor)==1
    source=source.replace(anchor,anchor+'\n    Tensor cGroup = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});\n    Tensor tGroup = thr_mma.partition_C(cGroup);')
    anchor='            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16\n        );'
    assert source.count(anchor)==2
    source=source.replace(anchor,anchor+'\n        apply_group_visibility(params, acc_s, tGroup, n_block);')
    # The last virtual tile can contain only another candidate's suffix.
    # Such a row is all -inf, so the original unmasked Check_inf=false is unsafe.
    source=source.replace('/*Check_inf=*/Is_causal || Is_local','/*Check_inf=*/true')
    source=source.replace('/*Check_inf=*/Is_local','/*Check_inf=*/true')
    (out/'group_kernel.h').write_text(source)
    inventory={str(p):sha(p) for p in [SRC/'shared_flash_fwd_kernel.h',SRC/'shared_load.h',
        SRC/'vendor/flash_attn/softmax.h',SRC/'vendor/flash_attn/kernel_traits.h',out/'group_kernel.h']}
    (OWN/'GENERATION.json').write_text(json.dumps(dict(source_sha256=inventory,
        unchanged_original_files=True,bitwise_equivalence_claim=False,
        changes=['group addresses','private suffix visibility','all-masked Check_inf','BM64'],
        flash_commit='979702c87a8713a8e0a5e9fee122b90d2ef13be5'),indent=2)+'\n')

if __name__=='__main__':main()
