"""CPU-exact frozen TF draft with explicit copies at the live CUDA boundary.

This is a diagnostic collector, never a draft latency benchmark. The original
CPU predictor and its normalization/math are reused; production code is not
patched and the model must remain on CPU.
"""
import hashlib
from pathlib import Path
import torch
from paths import EXP, load

MODEL_SOURCE = EXP / 'draft_architecture_v1/models.py'


class CandidatePredictor:
    def __init__(self, checkpoint, expected_sha, *, allow_device_transfer=False):
        path = Path(checkpoint).resolve()
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
            raise ValueError('candidate checkpoint hash differs')
        module = load('onpolicy_frozen_architecture_models', MODEL_SOURCE)
        # Loading initializes modules before state_dict loading; do not advance
        # the policy's global CPU RNG while constructing this private predictor.
        with torch.random.fork_rng(devices=[]):
            self.predictor = module.Predictor.load(path)
        if self.predictor.architecture != 'transformer64':
            raise ValueError('only frozen transformer64 data generator allowed')
        self.model = self.predictor.model
        self.checkpoint_path = str(path)
        self.checkpoint_sha256 = expected_sha
        self.allow_device_transfer = bool(allow_device_transfer)
        self._layout_server = None

    def validate_layout(self):
        """Read geometry established by the original reset; never create it."""
        server = self._layout_server
        if server is None:
            return {'status': 'STANDALONE_CPU_PREDICTOR'}
        if not hasattr(server, 'action_per_frame'):
            raise RuntimeError('candidate prediction requires original server reset first')
        if (server._p0_action_draft_mode != 'verifier_k2_b2_bonus' or
                tuple(server.job_config.used_action_channel_ids) != tuple(range(7)) or
                server.action_per_frame != server.job_config.action_per_frame or
                server.action_per_frame * server.job_config.frame_chunk_size != 16):
            raise ValueError('original reset must establish frozen K2/B2 compact geometry')
        return dict(status='VALIDATED_ORIGINAL_RESET_GEOMETRY',
                    action_per_frame=int(server.action_per_frame),
                    frame_chunk_size=int(server.job_config.frame_chunk_size))

    def layout_status(self):
        if self._layout_server is not None and not hasattr(self._layout_server, 'action_per_frame'):
            return {'status': 'AWAITING_ORIGINAL_RESET_NO_GEOMETRY_CLAIM'}
        return self.validate_layout()

    @torch.no_grad()
    def predict(self, current_state, previous_direction, sigma, next_sigma, timestep):
        self.validate_layout()
        if (current_state.device != previous_direction.device or
                current_state.shape != previous_direction.shape or current_state.numel() != 112):
            raise ValueError('matching compact 112 input tensors required')
        if any(p.device.type != 'cpu' for p in self.model.parameters()):
            raise ValueError('frozen diagnostic predictor must remain on CPU')
        if current_state.device.type != 'cpu' and not self.allow_device_transfer:
            raise ValueError('live input transfer requires explicit collector configuration')
        result = self.predictor.predict(current_state.detach().cpu(), previous_direction.detach().cpu(),
                                        sigma, next_sigma, timestep)
        return result.to(device=current_state.device, dtype=current_state.dtype)


def install(server, plan, *, allow_device_transfer):
    if (server._p0_action_draft_mode != 'verifier_k2_b2_bonus' or
            tuple(server.job_config.used_action_channel_ids) != tuple(range(7))):
        raise ValueError('frozen K2/B2 bonus action channels required')
    if server._p0_action_draft_b2_predictor is not None:
        raise ValueError('candidate must be installed before lazy MLP loading')
    predictor = CandidatePredictor(plan['new_checkpoint'], plan['new_checkpoint_sha256'],
                                   allow_device_transfer=allow_device_transfer)
    predictor._layout_server = server
    if hasattr(server, 'action_per_frame'):
        predictor.validate_layout()
    server._p0_action_draft_b2_checkpoint = plan['new_checkpoint']
    server._p0_action_draft_b2_predictor = predictor
    return predictor
