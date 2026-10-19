"""Read existing compact real inputs; never collect or synthesize model data.

The original capture normalized tensors with cpu().contiguous(). The layout
here is explicitly a SERIALIZED_LOGICAL_CPU contract, not a reconstruction of
unknown CUDA storage offsets/alias relationships from the production process.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

CORE_CASES = tuple((length, batch) for batch in (14, 2)
                   for length in ("short", "middle", "long"))


def decode(value, memo=None):
    if memo is None:
        memo = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, dict) and "encoding" in value:
        raw = value["value"]
        if value["encoding"] == "tensor":
            result = raw
        elif value["encoding"] == "two_CFG_groups_repeated7":
            if raw.shape[0] != 2:
                raise ValueError("CFG compact encoding must contain two group roots")
            result = raw.repeat_interleave(7, dim=0)
        else:
            raise ValueError(f"Unknown fixture encoding {value['encoding']}")
        if list(result.shape) != value["shape"]:
            raise ValueError("Decoded fixture shape does not match its manifest")
    elif isinstance(value, dict):
        result = {key: decode(item, memo) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        result = type(value)(decode(item, memo) for item in value)
    else:
        result = value
    memo[identity] = result
    return result


def tensors(value, prefix="input"):
    if isinstance(value, torch.Tensor):
        yield prefix, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from tensors(item, f"{prefix}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from tensors(item, f"{prefix}.{index}")




def visible_snapshot(cache):
    valid = cache["mask"].nonzero(as_tuple=False).squeeze(-1)
    return dict(slots=valid.clone(), k=cache["k"][:, valid].clone(),
                v=cache["v"][:, valid].clone(),
                **{key: cache[key].clone() for key in ("id", "mask", "is_pred")})


@dataclass
class BlockFixture:
    path: Path
    data: dict

    @classmethod
    def load(cls, root: Path, length: str, batch: int):
        if (length, batch) not in CORE_CASES:
            raise ValueError("Not a registered core fixture")
        path = root / f"input-{length}-B{batch}-action-layer0-first.pt"
        raw = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        data = decode(raw)
        if data["schema"] != "real-attention-replay-compact/v1":
            raise ValueError("Unsupported original fixture schema")
        if list(data["identity"]) != [length, batch, "action", "first", 0]:
            raise ValueError("Real fixture identity mismatch")
        if "block_args" not in data or "cache_metadata" not in data:
            raise ValueError("Not a complete saved block input")
        if list(data["block_args"][0].shape) != [batch, 16, 3072]:
            raise ValueError("Unexpected registered real action shape")
        if data["block_kwargs"]["update_cache"] != 0:
            raise ValueError("Core original fixtures must be read-only queries")
        return cls(path, data)

    @property
    def case_id(self):
        length, batch, *_ = self.data["identity"]
        return f"{length}-B{batch}-action-layer0-first"

    def arguments(self, update_cache=None):
        # Inputs are immutable across arms. Validator checks that explicitly;
        # mutable cache storage/metadata are allocated independently below.
        kwargs = dict(self.data["block_kwargs"])
        if update_cache is not None:
            if (update_cache not in (0, 1, 2)
                    or (self.data["identity"][1] != 2 and update_cache != 0)):
                raise ValueError("Only B2 permits registered legal-write controls")
            kwargs["update_cache"] = update_cache
        return self.data["block_args"], kwargs

    def fresh_cache(self):
        data = self.data
        metadata = {key: value.clone() for key, value in data["cache_metadata"].items()}
        slots = data["history_slots"]
        if not torch.equal(slots, metadata["mask"].nonzero(as_tuple=False).squeeze(-1)):
            raise ValueError("History slot list does not match the saved pre-call mask")
        if data["identity"][1] == 14:
            history_k, history_v = data["history_k"], data["history_v"]
        else:
            # Original dense self_k/self_v include the temporary private slots,
            # sorted by physical slot. Do not assume private rows are at the end.
            q_len = data["block_args"][0].shape[1]
            free = (~metadata["mask"]).nonzero(as_tuple=False).squeeze(-1)
            if free.numel() < q_len:
                raise ValueError("Core cache fixture unexpectedly requires eviction")
            during = metadata["mask"].clone()
            during[free[:q_len]] = True
            all_slots = during.nonzero(as_tuple=False).squeeze(-1)
            positions = torch.searchsorted(all_slots, slots)
            if data["self_k"].shape[1] != all_slots.numel():
                raise ValueError("Dense attention and logical cache slot counts disagree")
            history_k = data["self_k"][:, positions]
            history_v = data["self_v"][:, positions]
        cache = dict(k=torch.empty(data["pool_shape"], dtype=history_k.dtype,
                                   device=history_k.device),
                     v=torch.empty(data["pool_shape"], dtype=history_v.dtype,
                                   device=history_v.device), **metadata)
        cache["k"][:, slots] = history_k
        cache["v"][:, slots] = history_v
        return cache

    def install_cache(self, block):
        cache = self.fresh_cache()
        block.attn1.attn_caches = {self.data["block_kwargs"]["cache_name"]: cache}
        if block.attn2.attn_caches is not None:
            raise ValueError("Saved baseline requires original uncached cross-attention")
        return cache
