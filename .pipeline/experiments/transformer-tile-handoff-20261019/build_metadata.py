"""Build small CPU-only handoff ledgers from audited metadata and source.

This is a source-derived DAG, not a runtime profiler or new tensor replay.
No torch import, GPU context, model payload, or model-file hashing is used.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
NOW = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write(name: str, value: object) -> None:
    (BASE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


source_paths = [
    "算子研究/wam_ops/baseline/block.py",
    "算子研究/wam_ops/baseline/native.py",
    "算子研究/wam_ops/baseline/csrc/block.cu",
    "算子研究/wam_ops/baseline/csrc/gemm.h",
    "算子研究/wam_ops/optimized/csrc/ffn.cu",
    "算子研究/wam_ops/optimized/block.py",
    "投机采样/wam-p0-worktree/wan_va/modules/model.py",
]
sources = []
for name in source_paths:
    data = (ROOT / name).read_bytes()
    sources.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})

nodes = [
    {"id": "mod", "op": "FP32 table + temb", "deps": ["temb", "table"], "source": "block.cu:59-78", "scope": "per row/channel, six modulation groups"},
    {"id": "ln1", "op": "FP32 LayerNorm1", "deps": ["X"], "reduction": "full D=3072 row", "source": "block.cu:279"},
    {"id": "mod1", "op": "FP32 scale/shift -> BF16", "deps": ["ln1", "mod"], "source": "block.cu:280"},
    {"id": "sqkv", "op": "three Linear + bias -> BF16", "deps": ["mod1", "self_weights"], "reduction": "GEMM full K3072", "source": "block.cu:281-283"},
    {"id": "qk_norm", "op": "Q/K RMSNorm -> BF16", "deps": ["sqkv"], "reduction": "full D3072 before 24x128 head reshape", "source": "block.cu:281-282"},
    {"id": "rope", "op": "double arithmetic with complex64 frequencies -> BF16", "deps": ["qk_norm", "rotary"], "source": "block.cu:284"},
    {"id": "self_attn", "op": "Flash attention over legal history + private current KV", "deps": ["rope", "sqkv", "valid_KV", "row_map"], "visibility": "same CFG history; private KV remains per candidate", "source": "block.cu:286"},
    {"id": "self_out", "op": "Linear + bias -> BF16", "deps": ["self_attn", "self_out_weights"], "source": "block.cu:287"},
    {"id": "res1", "op": "FP32 gate*BF16 branch + residual -> BF16", "deps": ["X", "self_out", "mod"], "source": "block.cu:288"},
    {"id": "ln2", "op": "FP32 affine LayerNorm2 -> BF16", "deps": ["res1", "norm2_weights"], "reduction": "full D3072", "source": "block.cu:289"},
    {"id": "cross_q", "op": "Linear + Q RMSNorm", "deps": ["ln2", "cross_q_weights"], "reduction": "GEMM K3072 then full D3072 norm", "source": "block.cu:290"},
    {"id": "cross_kv", "op": "text K/V projection, K RMSNorm", "deps": ["context", "cross_kv_weights"], "cache_note": "uncached native path; strongest baseline may reuse already-admitted projections", "source": "block.cu:291-292"},
    {"id": "cross_attn", "op": "Flash attention, context length512", "deps": ["cross_q", "cross_kv"], "source": "block.cu:293"},
    {"id": "cross_out", "op": "Linear + bias -> BF16", "deps": ["cross_attn", "cross_out_weights"], "source": "block.cu:294"},
    {"id": "res2", "op": "FP32 residual sum -> BF16", "deps": ["res1", "cross_out"], "source": "block.cu:295"},
    {"id": "ln3", "op": "FP32 LayerNorm3", "deps": ["res2"], "reduction": "full D3072", "source": "block.cu:296"},
    {"id": "mod3", "op": "FP32 scale/shift -> BF16", "deps": ["ln3", "mod"], "source": "block.cu:297"},
    {"id": "ffn1", "op": "Linear 3072->14336 + bias -> BF16", "deps": ["mod3", "ffn1_weights"], "reduction": "full K3072 for each output tile", "source": "block.cu:299-304"},
    {"id": "gelu", "op": "FP32 tanh GELU on rounded BF16 input -> BF16", "deps": ["ffn1"], "source": "block.cu:299-307; ffn.cu:22-36"},
    {"id": "ffn2", "op": "Linear 14336->3072 + bias -> BF16", "deps": ["gelu", "ffn2_weights"], "reduction": "full K14336, exact arithmetic order a separate admission gate", "source": "block.cu:308"},
    {"id": "Y", "op": "FP32 gate*BF16 branch + residual -> BF16", "deps": ["res2", "ffn2", "mod"], "source": "block.cu:309"},
]
write("BLOCK-DAG.json", {
    "at": NOW, "scope": "source-derived logical DAG, not measured kernel trace",
    "reference_operator_commit": "99f7c2f3e845920fd83b22fedd19e72089f3a886",
    "source_path_for_block_cu": "算子研究/wam_ops/baseline/csrc/block.cu",
    "source_path_for_ffn_cu": "算子研究/wam_ops/optimized/csrc/ffn.cu",
    "sources": sources, "nodes": nodes,
    "existing_fused_node": ["ffn1", "gelu"],
    "proposed_scheduler_not_implemented": {
        "first_dependency_granularity": "all FFN1 columns ready for one m_tile, then FFN2 for that m_tile",
        "later_optional_granularity": "K-slice readiness only after exact accumulation semantics are proven",
        "private_KV_cross_candidate_attention_allowed": False,
        "GPU_execution_in_this_package": False,
    },
})

ledger = []
for row in json.loads((BASE / "SHAPES.json").read_text())["rows"]:
    b, q, d = row["block_args"][0]["shape"]
    context_length = row["block_args"][1]["shape"][1]
    assert d == 3072 and context_length == 512
    m = b * q
    ledger.append({
        "case": row["case"], "B": b, "Q": q, "M": m,
        "self_effective_K": row["pre_call_visible_slots"] + q,
        "history_visible_slots": row["pre_call_visible_slots"],
        "gemm_layout": "X[M,K] @ W[N,K].T, weight stored [N,K]",
        "gemm_shapes": [
            {"ops": ["self_q", "self_k", "self_v", "self_o", "cross_q", "cross_o"], "calls": 6, "M": m, "N": d, "K": d},
            {"ops": ["cross_k", "cross_v"], "calls_uncached": 2, "calls_if_admitted_cache_hit": 0, "M": b * context_length, "N": d, "K": d},
            {"ops": ["ffn1"], "calls": 1, "M": m, "N": 14336, "K": d},
            {"ops": ["ffn2"], "calls": 1, "M": m, "N": d, "K": 14336},
        ],
        "attention_per_batch_head": {"self_QK": [q, 128, row["pre_call_visible_slots"] + q], "cross_QK": [q, 128, context_length]},
        "source_fixture": row["input_path"],
    })
write("GEMM-SHAPES.json", {"at": NOW, "scope": "derived from audited logical shapes; actual library kernel tiles must be profiled separately", "rows": ledger})
print(json.dumps({"DAG_nodes": len(nodes), "shape_rows": len(ledger), "source_files": len(sources), "GPU_used": False}))
