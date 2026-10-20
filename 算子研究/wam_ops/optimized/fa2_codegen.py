"""Adapt pinned FA2 device functions to the whole block's four-warp group.

The frozen vendor files are never edited. Generated functions retain FA2 math,
traversal, tile geometry, split heuristic and combine order. Only worker indices,
barriers, shared storage and logical KV address loads are adapted.
"""
from pathlib import Path
import hashlib
import json
import re


def function(source, name):
    at = source.index("void " + name + "(")
    begin = source.rfind("template<", 0, at)
    body = source.index("{", at)
    # Mask comments and string literals without changing character offsets.
    masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"',
                    lambda m: " " * len(m[0]), source, flags=re.S)
    depth = 0
    for end in range(body, len(source)):
        depth += (masked[end] == "{") - (masked[end] == "}")
        if depth == 0:
            return source[begin:end + 1]
    raise ValueError(name)


def adapt_mainloop(text, old, new, split):
    text = text.replace(old, new)
    pos = text.index(") {")
    text = text[:pos] + ", char* smem_" + text[pos:]
    text = text.replace("extern __shared__ char smem_[];", "")
    text = text.replace("const int tidx = threadIdx.x;",
                        "const int tidx = int(threadIdx.x) - 96;")
    text = text.replace("__syncthreads();", "whole_group_sync();")
    # Keep the original CuTe copy atom/layout; supply indexed logical addresses.
    pattern = re.compile(r"FLASH_NAMESPACE::copy<(?P<flags>[^;]*?)>\(\s*"
                         r"gmem_tiled_copy_QKV,\s*t(?P<kind>[KV])g[KV]"
                         r"(?P<slice>\([^;]*?\))?,\s*t[KV]s[KV],\s*tKVcKV,\s*tKVpKV"
                         r"(?:,[^;]*?)?\);", re.S)
    replacements = []
    def replace(m):
        value = m["kind"] == "V"
        flags = m["flags"]
        clear = "Clear_OOB_MN=*/true" in flags
        if split:
            # Split mainloop advances its K pointer before prefetching the next K.
            nb = "n_block" if value else "n_block - 1"
            # Prologue K is handled below by its position.
        else:
            sl = m["slice"] or ""
            nb = "n_block - 1" if "n_block - 1" in sl else "n_block"
        replacements.append((m["kind"], nb))
        if split and len(replacements) == 1:
            assert not value
            nb = "n_block"
        return (f"copy_whole_kv<{str(value).lower()}, {str(clear).lower()}, "
                f"Kernel_traits::kBlockN>(params, bidb, bidh, {nb}, "
                f"gmem_tiled_copy_QKV, t{'V' if value else 'K'}s{'V' if value else 'K'}, tKVcKV);")
    text, count = pattern.subn(replace, text)
    assert count == 6, (old, count)
    return text, replacements


def generate(output):
    root = Path(__file__).resolve().parent / "csrc"
    source = (root / "vendor/flash_attn/flash_fwd_kernel.h").read_text()
    standard, a = adapt_mainloop(function(source, "compute_attn_1rowblock"),
        "compute_attn_1rowblock", "whole_standard", False)
    # Cross-attention的K/V连续，不需要历史slot/CFG地址间接层。
    # 此变体保留原FA2加载代码，只适配warp编号、同步及已有共享空间。
    direct = function(source, "compute_attn_1rowblock").replace(
        "compute_attn_1rowblock", "whole_standard_direct")
    pos = direct.index(") {")
    direct = direct[:pos] + ", char* smem_" + direct[pos:]
    direct = direct.replace("extern __shared__ char smem_[];", "")
    direct = direct.replace("const int tidx = threadIdx.x;",
                            "const int tidx = int(threadIdx.x) - 96;")
    direct = direct.replace("__syncthreads();", "whole_group_sync();")
    split, b = adapt_mainloop(function(source, "compute_attn_1rowblock_splitkv"),
        "compute_attn_1rowblock_splitkv", "whole_split", True)
    combine = function(source, "combine_attn_seqk_parallel")
    combine = combine.replace("combine_attn_seqk_parallel", "whole_combine")
    combine = combine.replace("const Params &params) {",
                              "const Params &params, int task, char* workspace) {")
    combine = combine.replace("__shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];",
        "auto sLSE = reinterpret_cast<ElementAccum (*)[kBlockM + 1]>(workspace);")
    combine = combine.replace("const int tidx = threadIdx.x;",
                              "const int tidx = int(threadIdx.x) - 96;")
    combine = combine.replace("const int bidx = blockIdx.x;", "const int bidx = task;")
    combine = combine.replace("__syncthreads();", "whole_group_sync();")
    output.mkdir(parents=True, exist_ok=True)
    text = ("// Derived from pinned FA2; licenses in csrc/vendor/flash_attn.\n"
            "#pragma once\n#include \"fa2_load.cuh\"\nnamespace FLASH_NAMESPACE {\n"
            + standard + "\n" + direct + "\n" + split + "\n" + combine + "\n}\n")
    (output / "fa2_device_generated.cuh").write_text(text)
    (output / "FA2-SOURCE.json").write_text(json.dumps({
        "source": str(root / "vendor/flash_attn/flash_fwd_kernel.h"),
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "generated_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "standard_load_replacements": a, "split_load_replacements": b,
        "barrier": "named barrier 2, 128 participants (physical warps 3..6)",
        "math_changed": False, "baseline_vendor_edited": False,
    }, indent=2) + "\n")
    return output
