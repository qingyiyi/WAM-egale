"""Copy pinned local upstream sources; generate six address-only substitutions.

No network, installations, CUDA initialization, or changes to the source lane.
"""
import difflib
import hashlib
import json
from pathlib import Path
import re
import shutil

OWN = Path(__file__).resolve().parent
ROOT = OWN.parents[2]
TILE = ROOT / 'wam_tutorial_a100/worktrees/c0-oplab-online-norm-20260922/operator_lab/wam_transformer_tile'
OBS = ROOT / '投机采样/results/runtime/p0-shared-prefix-observe-v2-20261012-182534'
EVIDENCE = ROOT / '.pipeline/research/speculative-sampling/evidence/shared-prefix-backend-observation-20261012'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p, x):
    with Path(p).open('x') as f:
        json.dump(x, f, ensure_ascii=False, indent=2)
        f.write('\n')


def generated(original):
    start = original.index('template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>\ninline __device__ void compute_attn_1rowblock(')
    end = original.index('\n////////////////////////////////////////////////////////////////////////////////////////////////////', start)
    body = original[start:end]
    body = body.replace('void compute_attn_1rowblock(', 'void compute_shared_attn_1rowblock(', 1)
    pattern = re.compile(r'FLASH_NAMESPACE::copy<([^>]+)>\(\s*gmem_tiled_copy_QKV, t([KV])g\2\(_, _, _, ([^)]+)\), t\2s\2, tKVcKV, tKVpKV(?:,\s*[^;]+?)?\s*\);')
    edits = []

    def replace(m):
        value = m[2] == 'V'
        clear = 'Clear_OOB_MN=*/true' in m[1]
        changed = (f'FLASH_NAMESPACE::copy_shared_kv<{str(value).lower()}, {str(clear).lower()}>'
                   f'(params, bidb, bidh, {m[3]}, gmem_tiled_copy_QKV, t{m[2]}s{m[2]}, tKVcKV);')
        edits.append(dict(old=m[0], new=changed))
        return changed

    body, n = pattern.subn(replace, body)
    if n != 6 or sum('true, true' in e['new'] for e in edits) != 1:
        raise ValueError('expected exactly six KV loads, including one V-tail clear')
    rebuilt = body.replace('void compute_shared_attn_1rowblock(', 'void compute_attn_1rowblock(', 1)
    for e in edits:
        rebuilt = rebuilt.replace(e['new'], e['old'], 1)
    if rebuilt != original[start:end]:
        raise ValueError('arithmetic/synchronization changed beyond KV loads')
    prefix = ('// Generated from pinned original FlashAttention; see ADDRESS-ONLY.diff.\n'
              '// Original arithmetic and synchronization are byte-preserved.\n'
              '#pragma once\n#include "shared_load.h"\nnamespace FLASH_NAMESPACE {\n')
    return prefix + body + '\n} // namespace FLASH_NAMESPACE\n', edits, original[start:end], body


def main():
    old = json.loads((OBS / 'plan.json').read_text())
    frozen = {**old['frozen_sha256'], **old['implementation_sha256']}
    for directory in (OBS, EVIDENCE, TILE):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                frozen[str(p)] = digest(p)
    for p, h in frozen.items():
        if digest(p) != h:
            raise ValueError('original source/evidence changed: ' + p)
    if (ROOT / '.pipeline/cycle.yaml').exists():
        raise ValueError('protected cycle absence changed')
    write(OWN / 'FROZEN-INPUTS.json', dict(files=frozen, protected_cycle_absent=True))
    metadata = json.loads((TILE / 'csrc/vendor/flash_attn/UPSTREAM.json').read_text())
    cutlass = json.loads((TILE / 'csrc/vendor/cutlass/UPSTREAM.json').read_text())
    for e in metadata['files']:
        p = TILE / e['path']
        if p.exists() and digest(p) != e['sha256']:
            raise ValueError('pinned Flash/CUTLASS source hash differs: ' + str(p))
    for p, h in cutlass['headers'].items():
        if digest(TILE / 'csrc/vendor/cutlass/include' / p) != h:
            raise ValueError('pinned CUTLASS source hash differs: ' + p)
    shutil.copytree(TILE / 'csrc/vendor', OWN / 'csrc/vendor')
    original = (OWN / 'csrc/vendor/flash_attn/flash_fwd_kernel.h').read_text()
    shared, edits, before, after = generated(original)
    (OWN / 'csrc/shared_flash_fwd_kernel.h').write_text(shared)
    (OWN / 'ADDRESS-ONLY.diff').write_text(''.join(difflib.unified_diff(
        before.splitlines(True), after.splitlines(True), fromfile='original FlashAttention rowblock',
        tofile='shared prefix rowblock')))
    files = {str(p.relative_to(OWN)): digest(p) for p in (OWN / 'csrc/vendor').rglob('*') if p.is_file()}
    write(OWN / 'SOURCE-PROVENANCE.json', dict(
        pytorch_version='2.9.0+cu126', pytorch_commit=metadata['pytorch_commit'],
        flash_attention_commit=metadata['flash_attention_commit'], cutlass_commit=cutlass['commit'],
        local_readonly_source=str(TILE / 'csrc/vendor'), vendor_files=files,
        address_substitutions=edits, arithmetic_and_synchronization_byte_preserved=True,
        original_standard_launch=dict(M=128, N=64, D=128, warps=4, Q_in_registers=False),
        original_compile_flags=['--use_fast_math', '-DUNFUSE_FMA'],
        source_alignment_not_identical_wheel_SASS_claim=True,
        network=False, installations=False, CUDA_initialized=False))
    print(json.dumps(dict(frozen_files=len(frozen), copied_vendor_files=len(files), address_edits=len(edits))))


if __name__ == '__main__':
    main()
