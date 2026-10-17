"""Independent suite assets and original bank math; no legacy loader or digest."""
from __future__ import annotations

import ast
import copy
import os
from pathlib import Path
import types

from .protocol import EXP, PERMISSIONS, read, require


def configure_suite_source(source, plan):
    """Only replace this process's libero config object; never edit Long files."""
    info = plan["suite_info"]
    model = Path(info["model_path"]).resolve(strict=True)
    registry_path = Path(plan["registry"]).resolve(strict=True)
    entry = read(registry_path)["suites"][info["suite"]]
    require(entry["status"] == "available" and entry["official_checkpoint"] == info["checkpoint_id"]
        and Path(entry["local_path"]).resolve(strict=True) == model, "independent suite registry/model identity differs")
    require(all((model / part).is_dir() for part in ("transformer", "vae", "text_encoder", "tokenizer")),
        "assembled suite transformer plus shared VAE/text/tokenizer required")
    original = source.VA_CONFIGS["libero"]
    config = copy.deepcopy(original)
    require(config.action_dim == 30 and tuple(config.used_action_channel_ids) == tuple(range(7))
        and config.action_norm_method == "quantiles", "original LIBERO action normalization contract differs")
    config.wan22_pretrained_model_name_or_path = str(model)
    config.libero_eval_suite = info["suite"]
    config.libero_checkpoint_id = info["checkpoint_id"]
    config.libero_checkpoint_registry = str(registry_path)
    config.norm_stat = copy.deepcopy(info["norm_stat"])
    source.VA_CONFIGS = {**source.VA_CONFIGS, "libero": config}
    os.environ.update(LINGBOT_VA_LIBERO_SUITE=info["suite"], LINGBOT_VA_LIBERO_CHECKPOINT=str(model),
        LINGBOT_VA_LIBERO_CHECKPOINT_ID=info["checkpoint_id"], LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY=str(registry_path))
    return config, dict(schema="tf-suite-process-config/v1", suite=info["suite"],
        community_checkpoint_id=info["checkpoint_id"], publisher_revision=info["publisher_revision"],
        model_path=str(model), independent_registry=str(registry_path), norm_stat=config.norm_stat,
        changed_fields=["wan22_pretrained_model_name_or_path", "libero_eval_suite", "libero_checkpoint_id",
            "libero_checkpoint_registry", "norm_stat"], original_config_object_unchanged=True,
        original_run_and_model_math_unchanged=True, permissions=PERMISSIONS)


def load_predictor(plan, server):
    if all(r["arm"] == "T" for r in plan["rows"]):
        path = str(Path(plan["job"]) / "UNUSED_T_ONLY_NO_SUITE_DRAFT")
        def unavailable(*_args, **_keywords):
            raise RuntimeError("T-only pipeline must never request a draft prediction")
        predictor = types.SimpleNamespace(checkpoint_path=path, checkpoint_condition="UNUSED_T_ONLY",
            predict=unavailable, inline_observer=None)
        return predictor, dict(schema="tf-suite-no-draft-T-only/v1", suite=plan["suite"],
            no_draft_loaded=True, no_Long_draft_substitution=True, original_target_only=True, permissions=PERMISSIONS)
    from .adapter import load_checkpoint, install_inline_observer
    selected = plan["selected"]
    predictor = load_checkpoint(selected["path"], architecture="transformer64", device=server.device,
        expected_channels=range(7), suite=plan["suite"].removeprefix("libero_"),
        condition=selected["condition"], expected_metadata=selected["metadata"])
    predictor.inline_observer = install_inline_observer(predictor, max_audited_queries=32)
    return predictor, dict(schema="tf-suite-draft-deployment/v1", checkpoint_path=predictor.checkpoint_path,
        suite=plan["suite"], condition=selected["condition"], parameter_count=53070,
        metadata=selected["metadata"], no_SHA=True, no_old_CAL_inheritance=True,
        inline_decode_audit="first32 actual predict calls; no extra teacher/model query; not CPU-GPU equivalence",
        permissions=PERMISSIONS)


def load_text_bank(plan, server):
    import torch
    path = Path(plan["bank"]).resolve(strict=True)
    require(path.stat().st_size <= 64 * 2**20, "bounded suite text bank required")
    data = torch.load(path, map_location="cpu", weights_only=True)
    info = plan["suite_info"]
    require(data["schema"] == "libero-suite-original-T5-bank/v1" and data["suite"] == info["suite"]
        and data["target_checkpoint_id"] == info["checkpoint_id"]
        and Path(data["checkpoint_path"]).resolve(strict=True) == Path(info["model_path"]).resolve(strict=True)
        and data["sequence_length"] == 512 and data["dtype"] == "bfloat16"
        and data["output_shape"] == [1, 512, 4096] and data["torch_version"] == str(torch.__version__)
        and data["public_task_prompts"] == info["public_task_prompts"], "suite bank identity/full512/runtime differs")
    embeddings = data["embeddings"]
    require(set(embeddings) == set(info["public_task_prompts"].values()) | {""} and len(embeddings) == 11,
        "exact ten suite prompts plus empty negative required")
    for value in embeddings.values():
        require(isinstance(value, torch.Tensor) and value.device.type == "cpu" and value.dtype == torch.bfloat16
            and tuple(value.shape) == (1, 512, 4096) and bool(torch.isfinite(value).all()), "suite bank tensor differs")
    source = EXP / "reset_text_bank_v1/bank.py"
    tree = ast.parse(source.read_bytes(), filename=str(source))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "TextBank"]
    require(len(classes) == 1, "one original TextBank class required")
    gets = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == "get"]
    installs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "install"]
    require(len(gets) == len(installs) == 1, "original get/install source required")
    bank = types.SimpleNamespace(path=path, data=data, embeddings=embeddings, hits=0)
    namespace = dict(torch=torch, Path=Path, os=os, types=types, TextBank=object,
        __name__="tf_suite_pipeline_v1.original_bank_math")
    exec(compile(ast.Module(body=gets + installs, type_ignores=[]), str(source), "exec"), namespace)
    bank.get = types.MethodType(namespace["get"], bank)
    restore = namespace["install"](server, bank)
    return bank, dict(schema="tf-suite-text-bank-deployment/v1", suite=info["suite"], path=str(path),
        target_checkpoint_id=info["checkpoint_id"], sequence_length=512, embeddings=11, dtype="bfloat16",
        original_get_install_AST_unchanged=True, no_SHA=True, no_T5_forward=True, permissions=PERMISSIONS), restore
