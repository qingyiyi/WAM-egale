"""Same small residual-direction transformer, adapted to RoboTwin 32x14.

FIT-only statistics and a newly trained checkpoint are mandatory. No inherited
LIBERO normalizer, direction checkpoint, or acceptance threshold is accepted.
"""
import torch
from torch import nn


class DirectionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(31, 64)
        self.position = nn.Parameter(torch.empty(1, 32, 64))
        nn.init.normal_(self.position, std=.02)
        self.block = nn.TransformerEncoderLayer(64, 4, 256, dropout=0., activation='gelu',
                                               batch_first=True, norm_first=True)
        self.head = nn.Linear(64, 28)
        for name, n in [('feature', 899), ('residual', 448)]:
            self.register_buffer(name + '_mean', torch.zeros(n))
            self.register_buffer(name + '_std', torch.ones(n))

    def decode(self, state, previous, scalars):
        b = state.shape[0]
        assert state.shape == previous.shape == (b, 32, 14)
        assert scalars.shape == (b, 3)
        state, previous = state.float().reshape(b, 448), previous.float().reshape(b, 448)
        features = torch.cat((state, previous, scalars.float()), 1)
        x = (features - self.feature_mean) / self.feature_std
        x = torch.cat((x[:, :448].reshape(b, 32, 14), x[:, 448:896].reshape(b, 32, 14),
                       x[:, 896:, None].transpose(1, 2).expand(-1, 32, -1)), -1)
        y = self.head(self.block(self.embed(x) + self.position))
        center, spread = y[..., :14], y[..., 14:]
        residual = torch.stack((center + spread, center - spread), 1).reshape(b, 2, 448)
        return (previous[:, None] + residual * self.residual_std + self.residual_mean).reshape(b, 2, 32, 14)

    def forward(self, state, previous, timestep, delta):
        assert state.shape == previous.shape == (1, 32, 14)
        sigma = timestep.float() / 1000.
        scalars = torch.stack((sigma, sigma + delta.float(), sigma)).reshape(1, 3)
        return self.decode(state, previous, scalars).to(state.dtype)[0]


def load(path, device='cuda'):
    data = torch.load(path, map_location='cpu', weights_only=True)
    assert data['schema'] == 'fastwam-robotwin-direction-v1' and data['step'] == 3000
    assert data['action_dim'] == 14 and data['action_horizon'] == 32
    model = DirectionPredictor().to(device)
    model.load_state_dict(data['state_dict'], strict=True)
    return model.eval().requires_grad_(False)
