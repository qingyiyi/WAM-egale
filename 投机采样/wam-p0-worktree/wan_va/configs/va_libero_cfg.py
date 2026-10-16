# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os
from pathlib import Path

from easydict import EasyDict

from .shared_config import va_shared_cfg

va_libero_cfg = EasyDict(__name__='Config: VA libero')
va_libero_cfg.update(va_shared_cfg)
va_shared_cfg.infer_mode = 'server'

# Formal LIBERO evaluation must be explicitly checkpoint-aligned by the
# launcher.  Do not provide a Long checkpoint fallback here: this config is
# imported for other server modes as well, so validation happens only when the
# ``libero`` server is selected.
va_libero_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "LINGBOT_VA_LIBERO_CHECKPOINT", ""
)
va_libero_cfg.libero_eval_suite = os.getenv("LINGBOT_VA_LIBERO_SUITE", "")
va_libero_cfg.libero_checkpoint_id = os.getenv("LINGBOT_VA_LIBERO_CHECKPOINT_ID", "")
va_libero_cfg.libero_checkpoint_registry = os.getenv(
    "LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY", ""
)


def validate_libero_checkpoint_environment(config=va_libero_cfg):
    """Fail before model construction unless registry and checkpoint agree.

    Checking only that a checkpoint directory exists is insufficient: a caller
    could label the LIBERO-Long directory as Goal/Spatial/Object.  Every formal
    server therefore validates the launcher environment against the same
    fail-closed registry consumed by ``p0_resolve_libero_checkpoint.py``.
    """
    missing = [
        name for name, value in (
            ("LINGBOT_VA_LIBERO_SUITE", config.libero_eval_suite),
            ("LINGBOT_VA_LIBERO_CHECKPOINT", config.wan22_pretrained_model_name_or_path),
            ("LINGBOT_VA_LIBERO_CHECKPOINT_ID", config.libero_checkpoint_id),
            ("LINGBOT_VA_LIBERO_CHECKPOINT_REGISTRY", config.libero_checkpoint_registry),
        ) if not value
    ]
    if missing:
        raise RuntimeError(
            "checkpoint-aligned LIBERO evaluation requires " + ", ".join(missing)
            + ". Start it through scripts/p0_resolve_libero_checkpoint.py; "
            "there is intentionally no fallback to the LIBERO-Long checkpoint."
        )
    checkpoint = Path(config.wan22_pretrained_model_name_or_path)
    registry_path = Path(config.libero_checkpoint_registry)
    if not checkpoint.is_dir():
        raise RuntimeError(
            f"configured LIBERO checkpoint path does not exist: {checkpoint}. "
            "Do not substitute a checkpoint from another suite."
        )
    if not registry_path.is_file():
        raise RuntimeError(
            f"configured LIBERO checkpoint registry does not exist: {registry_path}. "
            "Do not bypass the suite-to-checkpoint resolver."
        )
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        suite_entry = registry["suites"][config.libero_eval_suite]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"invalid LIBERO checkpoint registry/suite for {config.libero_eval_suite!r}: {exc}"
        ) from exc
    if suite_entry.get("status") != "available":
        raise RuntimeError(
            f"LIBERO suite {config.libero_eval_suite!r} is not available for formal "
            "evaluation in the checkpoint registry; do not substitute another suite's checkpoint."
        )
    expected_path = Path(suite_entry.get("local_path", "")).resolve()
    expected_id = suite_entry.get("official_checkpoint")
    if checkpoint.resolve() != expected_path or config.libero_checkpoint_id != expected_id:
        raise RuntimeError(
            "LIBERO suite/checkpoint mismatch: registry requires "
            f"suite={config.libero_eval_suite!r}, checkpoint={expected_path}, "
            f"id={expected_id!r}; received checkpoint={checkpoint.resolve()}, "
            f"id={config.libero_checkpoint_id!r}. Do not substitute a checkpoint from another suite."
        )

# Keep the text encoder on CPU: disabling all offload does not fit A100 40GB.
# Observation encoding is on the worker's GPU, independently of text offload.
# WAM_VAE_DEVICE=cpu restores the CPU-VAE control; both timing arms must match.
va_libero_cfg.enable_offload = True
va_libero_cfg.vae_device = 'cuda'

va_libero_cfg.attn_window = 30
va_libero_cfg.frame_chunk_size = 4
va_libero_cfg.env_type = 'none'

va_libero_cfg.height = 128
va_libero_cfg.width = 128
va_libero_cfg.action_dim = 30
va_libero_cfg.action_per_frame = 4
va_libero_cfg.obs_cam_keys = [
    'observation.images.agentview_rgb', 'observation.images.eye_in_hand_rgb'
]
va_libero_cfg.guidance_scale = 5
va_libero_cfg.action_guidance_scale = 1

va_libero_cfg.num_inference_steps = 20
va_libero_cfg.video_exec_step = -1
va_libero_cfg.action_num_inference_steps = 50

va_libero_cfg.snr_shift = 5.0
va_libero_cfg.action_snr_shift = 0.05

va_libero_cfg.used_action_channel_ids = list(range(0, 7))
inverse_used_action_channel_ids = [len(va_libero_cfg.used_action_channel_ids)
                                   ] * va_libero_cfg.action_dim
for i, j in enumerate(va_libero_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_libero_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_libero_cfg.action_norm_method = 'quantiles'
va_libero_cfg.norm_stat = {
    "q01": [
        -0.6589285731315613,
        -0.84375,
        -0.9375,
        -0.12107142806053162,
        -0.15964286029338837,
        -0.26571428775787354,
        -1.0
    ] + [0.] * 23,
    "q99": [
        0.8999999761581421,
        0.8544642925262451,
        0.9375,
        0.17142857611179352,
        0.1842857152223587,
        0.34392857551574707,
        1.0
    ] + [0.] * 23,
}
