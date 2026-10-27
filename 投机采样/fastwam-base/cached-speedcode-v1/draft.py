"""FastWAM adaptation of the frozen WAM PositionTransformer residual predictor.

32 time-major action positions; new FIT normalization; no old checkpoint/tau.
Target direction is decoded as previous_direction + predicted residual.
"""
import torch
from torch import nn


class DirectionPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(17, 64)
        self.position = nn.Parameter(torch.empty(1, 32, 64))
        nn.init.normal_(self.position, std=.02)
        self.block = nn.TransformerEncoderLayer(64, 4, 256, dropout=0., activation='gelu',
                                               batch_first=True, norm_first=True)
        self.head = nn.Linear(64, 14)
        for name, n in [('feature', 451), ('residual', 224)]:
            self.register_buffer(name + '_mean', torch.zeros(n))
            self.register_buffer(name + '_std', torch.ones(n))

    def decode(self, state, previous, scalars):
        b = state.shape[0]
        state, previous = state.float().reshape(b, 224), previous.float().reshape(b, 224)
        features = torch.cat((state, previous, scalars.float()), 1)
        x = (features - self.feature_mean) / self.feature_std
        x = torch.cat((x[:, :224].reshape(b, 32, 7), x[:, 224:448].reshape(b, 32, 7),
                       x[:, 448:, None].transpose(1, 2).expand(-1, 32, -1)), -1)
        y = self.head(self.block(self.embed(x) + self.position))
        center, spread = y[..., :7], y[..., 7:]
        residual = torch.stack((center + spread, center - spread), 1).reshape(b, 2, 224)
        return (previous[:, None] + residual * self.residual_std + self.residual_mean).reshape(b, 2, 32, 7)

    def forward(self, state, previous, timestep, delta):
        assert state.shape == previous.shape == (1, 32, 7)
        sigma = timestep.float() / 1000.
        scalars = torch.stack((sigma, sigma + delta.float(), sigma)).reshape(1, 3)
        return self.decode(state, previous, scalars).to(state.dtype)[0]


def load(path, device='cuda'):
    data = torch.load(path, map_location='cpu', weights_only=True)
    assert data['schema'] == 'fastwam-direction-v1' and data['step'] == 3000
    model = DirectionPredictor().to(device)
    model.load_state_dict(data['state_dict'], strict=True)
    model.eval().requires_grad_(False)
    return model
