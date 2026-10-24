"""RoboTwin TF64: original single pre-norm block, 32 positions × 16 channels.

Only representation-dependent input/output projections and positions differ from
LIBERO. Predict compact active channels in the config order; caller scatters into
full 30-channel latent without changing the target noise allocation.
"""
import torch
from torch import nn

CHANNELS, TOKENS, SIZE = 16, 32, 512
SPEC = dict(width=64, heads=4, ffn=256, layers=1, dropout=0., tokens=TOKENS,
            channels=CHANNELS, norm_first=True, activation='gelu')

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(35, 64)
        self.position = nn.Parameter(torch.empty(1, TOKENS, 64))
        nn.init.normal_(self.position, std=.02)
        self.block = nn.TransformerEncoderLayer(64, 4, 256, dropout=0., activation='gelu',
                                                batch_first=True, norm_first=True)
        self.head = nn.Linear(64, 32)

    def forward(self, features):
        state = features[:, :SIZE].reshape(-1, CHANNELS, TOKENS).transpose(1, 2)
        previous = features[:, SIZE:2*SIZE].reshape(-1, CHANNELS, TOKENS).transpose(1, 2)
        times = features[:, 2*SIZE:].unsqueeze(1).expand(-1, TOKENS, -1)
        hidden = self.embed(torch.cat((state, previous, times), -1)) + self.position
        raw = self.head(self.block(hidden))
        center = raw[:, :, :CHANNELS].transpose(1, 2).reshape(-1, SIZE)
        delta = raw[:, :, CHANNELS:].transpose(1, 2).reshape(-1, SIZE)
        return torch.stack((center + delta, center - delta), 1)

class Predictor:
    def __init__(self, model, norm):
        self.model = model
        for key, value in norm.items():
            setattr(self, key, value.to(next(model.parameters()).device))

    def decode_batch(self, data):
        features = torch.cat((data['state'], data['previous'], data['scalars']), 1)
        raw = self.model((features-self.feature_mean)/self.feature_std)
        return (data['previous'][:, None] + raw*self.residual_std[None, None]
                + self.residual_mean[None, None])

    @torch.no_grad()
    def predict(self, state, previous, sigma, next_sigma, timestep):
        assert state.shape == previous.shape and state.numel() == SIZE
        device = next(self.model.parameters()).device
        assert state.device == device
        data = dict(state=state.reshape(1, SIZE).float(), previous=previous.reshape(1, SIZE).float(),
                    scalars=torch.tensor([[sigma, next_sigma, timestep/1000.]], device=device))
        return self.decode_batch(data).reshape(2, *state.shape).to(state.dtype)
