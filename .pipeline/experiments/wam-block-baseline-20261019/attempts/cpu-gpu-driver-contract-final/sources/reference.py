"""Load the real reference from a fixed Git object without importing the server.

The class/function AST nodes below are compiled unchanged, including all cache
helpers. Import-time FlexAttention compilation, model construction and native
extension imports in the full model module are deliberately not executed.
Only the original ``attn_mode='torch'`` branch is supported by this loader.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess

import torch
from torch import nn
from torch.nn import functional as F

REVIEW_COMMIT = "ca5f37e0482c2e38e3738e3d8e71e3b185990a23"
MODEL_SOURCE = "投机采样/wam-p0-worktree/wan_va/modules/model.py"


def frozen_source(repository: Path, relative: str) -> str:
    """Read the specified commit, never the checkout's current HEAD."""
    return subprocess.check_output(
        ["git", "show", f"{REVIEW_COMMIT}:{relative}"], cwd=repository
    ).decode("utf-8")


def reference_namespace(repository: Path) -> dict:
    from diffusers.models.attention import FeedForward
    from diffusers.models.normalization import FP32LayerNorm
    from einops import rearrange

    source = frozen_source(repository, MODEL_SOURCE)
    wanted = {"custom_sdpa", "WanAttention", "WanTransformerBlock"}
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise RuntimeError("The frozen reference's class/function contract is missing")

    class UnsupportedFlexAttention(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError("FlexAttention is outside this isolated replay contract")

    def unsupported_flash(*args, **kwargs):
        raise RuntimeError("FlashAttention is outside this loader; use torch reference")

    filename = f"git:{REVIEW_COMMIT}:{MODEL_SOURCE}"
    namespace = dict(torch=torch, nn=nn, F=F, math=math, rearrange=rearrange,
                     FeedForward=FeedForward, FP32LayerNorm=FP32LayerNorm,
                     FlexAttnFunc=UnsupportedFlexAttention, flash_attn_func=unsupported_flash,
                     __name__="wam_frozen_block_reference")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    namespace["source_identity"] = {
        "commit": REVIEW_COMMIT, "path": MODEL_SOURCE,
        "sha256": hashlib.sha256(source.encode()).hexdigest(),
        "extraction": "unchanged AST nodes; explicit torch-only import environment",
        "symbols": {node.name: {"line": node.lineno, "end_line": node.end_lineno}
                    for node in nodes},
    }
    return namespace


def load_block(repository: Path, checkpoint: Path, layer: int = 0):
    """Read only one block from the original checkpoint, into CPU memory.

    No checkpoint byte hash, full-model load, dtype conversion or CUDA call.
    Meta construction avoids materializing random parameters before loading.
    Parameter names/shapes are checked by strict load_state_dict.
    """
    from safetensors import safe_open

    namespace = reference_namespace(repository)
    transformer = checkpoint / "transformer"
    config = json.loads((transformer / "config.json").read_text())
    index_path = transformer / "diffusion_pytorch_model.safetensors.index.json"
    index = json.loads(index_path.read_text())["weight_map"]
    prefix = f"blocks.{layer}."
    selected = {name: shard for name, shard in index.items() if name.startswith(prefix)}
    if not selected:
        raise RuntimeError(f"No original checkpoint parameters for {prefix}")
    state = {}
    for shard in sorted(set(selected.values())):
        with safe_open(transformer / shard, framework="pt", device="cpu") as handle:
            for name, owner in selected.items():
                if owner == shard:
                    state[name.removeprefix(prefix)] = handle.get_tensor(name)
    with torch.device("meta"):
        block = namespace["WanTransformerBlock"](
            dim=config["num_attention_heads"] * config["attention_head_dim"],
            ffn_dim=config["ffn_dim"], num_heads=config["num_attention_heads"],
            cross_attn_norm=config["cross_attn_norm"], eps=config["eps"], attn_mode="torch")
    block.load_state_dict(state, strict=True, assign=True)
    block.eval()
    block.requires_grad_(False)
    manifest = dict(namespace["source_identity"], layer=layer,
                    checkpoint=str(checkpoint.resolve()), config=config,
                    model_weight_hash="NOT_COMPUTED_USER_BOUNDARY",
                    parameters={name: {"checkpoint_key": prefix + name,
                        "shard": selected[prefix + name], "shape": list(param.shape),
                        "dtype": str(param.dtype), "stride": list(param.stride()),
                        "requires_grad": param.requires_grad}
                        for name, param in block.named_parameters()},
                    parameter_count=sum(x.numel() for x in block.parameters()),
                    parameter_bytes=sum(x.numel() * x.element_size() for x in block.parameters()))
    return block, manifest
