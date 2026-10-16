"""Isolated, CPU-testable two-proposal architectures; production sources unchanged."""
from __future__ import annotations

import math
from pathlib import Path
import torch
from torch import nn

SPECS = {
    'mlp51': {'kind': 'mlp', 'hidden': 51},
    'mlp96': {'kind': 'mlp', 'hidden': 96},
    'transformer64': {'kind': 'transformer', 'width': 64, 'heads': 4, 'ffn': 256,
                      'layers': 1, 'dropout': 0., 'tokens': 16, 'activation': 'gelu', 'norm_first': True},
}


class ResidualMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(227, hidden), nn.SiLU(),
                                   nn.Linear(hidden, hidden), nn.SiLU())
        self.head = nn.Linear(hidden, 224)

    def forward(self, x):
        center, delta = self.head(self.trunk(x)).reshape(-1, 2, 112).unbind(1)
        return torch.stack((center + delta, center - delta), 1)


def position_tokens(x):
    """Canonical [channel, frame, subaction] flattening -> 16 position tokens."""
    if x.ndim != 2 or x.shape[1] != 227:
        raise ValueError('expected [batch, 227] normalized features')
    state = x[:, :112].reshape(-1, 7, 16).transpose(1, 2)
    previous = x[:, 112:224].reshape(-1, 7, 16).transpose(1, 2)
    scalars = x[:, 224:].unsqueeze(1).expand(-1, 16, -1)
    return torch.cat((state, previous, scalars), -1)


class PositionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(17, 64)
        self.position = nn.Parameter(torch.empty(1, 16, 64))
        nn.init.normal_(self.position, std=.02)
        self.block = nn.TransformerEncoderLayer(64, 4, 256, dropout=0., activation='gelu',
                                                batch_first=True, norm_first=True)
        self.head = nn.Linear(64, 14)

    def forward(self, x):
        y = self.head(self.block(self.embed(position_tokens(x)) + self.position))
        center = y[:, :, :7].transpose(1, 2).reshape(-1, 112)
        delta = y[:, :, 7:].transpose(1, 2).reshape(-1, 112)
        return torch.stack((center + delta, center - delta), 1)


def make_model(name):
    spec = SPECS[name]
    return ResidualMLP(spec['hidden']) if spec['kind'] == 'mlp' else PositionTransformer()


class Predictor:
    def __init__(self, architecture, normalization):
        self.architecture = architecture
        self.model = make_model(architecture).float()
        for key in ('feature_mean', 'feature_std', 'residual_mean', 'residual_std'):
            setattr(self, key, normalization[key].detach().clone().float())
        if self.feature_mean.shape != (227,) or self.residual_mean.shape != (112,):
            raise ValueError('normalizer shape mismatch')
        if any(not bool(torch.isfinite(getattr(self, k)).all()) for k in normalization):
            raise ValueError('nonfinite normalizer')
        if bool((self.feature_std <= 0).any()) or bool((self.residual_std <= 0).any()):
            raise ValueError('normalizer must be positive')

    def decode_batch(self, data):
        features = torch.cat((data['state'], data['previous'], data['scalars']), 1)
        normalized = (features - self.feature_mean) / self.feature_std
        raw = self.model(normalized)
        return (data['previous'][:, None] + raw * self.residual_std[None, None]
                + self.residual_mean[None, None])

    @torch.no_grad()
    def predict(self, current_state, previous_direction, sigma, next_sigma, timestep):
        if current_state.shape != previous_direction.shape or current_state.numel() != 112:
            raise ValueError('two-head predictor requires matching compact 112-element tensors')
        if current_state.device.type != 'cpu' or previous_direction.device.type != 'cpu':
            raise ValueError('this experiment only authorizes CPU prediction')
        if not all(math.isfinite(float(v)) for v in (sigma, next_sigma, timestep)):
            raise ValueError('nonfinite time scalar')
        if not bool(torch.isfinite(current_state).all()) or not bool(torch.isfinite(previous_direction).all()):
            raise ValueError('nonfinite input')
        self.model.eval()
        data = {'state': current_state.detach().reshape(1, 112).float(),
                'previous': previous_direction.detach().reshape(1, 112).float(),
                'scalars': torch.tensor([[sigma, next_sigma, timestep / 1000.]], dtype=torch.float32)}
        value = self.decode_batch(data).reshape(2, *current_state.shape).to(current_state.dtype)
        if not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite output')
        return value

    def save(self, path, metadata):
        torch.save({'schema': 'draft-architecture-cpu/v1', 'architecture': self.architecture,
                    'specification': SPECS[self.architecture], 'metadata': metadata,
                    'model_state_dict': self.model.state_dict(),
                    'normalization': {k: getattr(self, k) for k in
                                      ('feature_mean', 'feature_std', 'residual_mean', 'residual_std')}}, path)

    @classmethod
    def load(cls, path):
        data = torch.load(Path(path), map_location='cpu', weights_only=False)
        if data['schema'] != 'draft-architecture-cpu/v1' or data['specification'] != SPECS[data['architecture']]:
            raise ValueError('checkpoint architecture mismatch')
        result = cls(data['architecture'], data['normalization'])
        result.model.load_state_dict(data['model_state_dict'], strict=True)
        result.model.eval()
        return result
