"""RoboTwin TF64 geometry; LIBERO architecture, loaders and weights are untouched."""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import nn

CHANNEL_IDS=(0,1,2,3,4,5,6,28,7,8,9,10,11,12,13,29)
TOKENS,CHANNELS,DIM=32,16,512
FEATURES=2*DIM+3
SCHEMA='robotwin-tf64-draft/v1'
SPEC=dict(kind='transformer',width=64,heads=4,ffn=256,layers=1,dropout=0.,
          tokens=TOKENS,channels=CHANNELS,channel_ids=list(CHANNEL_IDS),norm_first=True,activation='gelu')
PERMISSIONS=dict(engineering_quality_screen=False,independent_test_complete=False,
                 noninferiority_established=False,formal_performance_permission=False)

class PositionTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Linear(35,64)
        self.position=nn.Parameter(torch.empty(1,TOKENS,64))
        nn.init.normal_(self.position,std=.02)
        self.block=nn.TransformerEncoderLayer(64,4,256,dropout=0.,activation='gelu',
                                             batch_first=True,norm_first=True)
        self.head=nn.Linear(64,32)
        self.output_dim=DIM

    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=FEATURES:raise ValueError('RoboTwin requires [N,1027]')
        state=x[:,:DIM].reshape(-1,CHANNELS,TOKENS).transpose(1,2)
        previous=x[:,DIM:2*DIM].reshape(-1,CHANNELS,TOKENS).transpose(1,2)
        scalars=x[:,2*DIM:].unsqueeze(1).expand(-1,TOKENS,-1)
        y=self.head(self.block(self.embed(torch.cat((state,previous,scalars),-1))+self.position))
        center=y[:,:,:CHANNELS].transpose(1,2).reshape(-1,DIM)
        delta=y[:,:,CHANNELS:].transpose(1,2).reshape(-1,DIM)
        return torch.stack((center+delta,center-delta),1)

class Predictor:
    """Same FP32 normalization/model/decode math, returned in original state dtype."""
    def __init__(self,model,normalization,device='cpu',checkpoint_path=None):
        self.model=model.eval().to(device=device,dtype=torch.float32)
        self.device=next(model.parameters()).device
        self.normalization={}
        for key,shape in [('feature_mean',(FEATURES,)),('feature_std',(FEATURES,)),
                          ('residual_mean',(DIM,)),('residual_std',(DIM,))]:
            value=normalization[key]
            if value.dtype!=torch.float32 or tuple(value.shape)!=shape or not bool(torch.isfinite(value).all()):raise ValueError('normalizer geometry/dtype: '+key)
            if key.endswith('std') and not bool((value>0).all()):raise ValueError('positive std required')
            self.normalization[key]=value.detach().clone().to(self.device)
        self.output_dim,self.heads=DIM,2
        self.checkpoint_path=str(checkpoint_path) if checkpoint_path else None

    def decode_batch(self,data):
        previous=data['previous'].float()
        features=torch.cat((data['state'].float(),previous,data['scalars'].float()),1)
        n=self.normalization
        raw=self.model((features-n['feature_mean'])/n['feature_std'])
        return previous[:,None]+raw*n['residual_std'][None,None]+n['residual_mean'][None,None]

    @torch.no_grad()
    def predict(self,current_state,previous_direction,sigma,next_sigma,timestep):
        if current_state.shape!=previous_direction.shape or current_state.numel()!=DIM or current_state.shape[0]!=CHANNELS:raise ValueError('matching ordered 16-channel/512-element compact state required')
        if current_state.device!=self.device or previous_direction.device!=self.device:raise ValueError('predictor/state device mismatch')
        if not all(math.isfinite(float(v)) for v in (sigma,next_sigma,timestep)):raise ValueError('finite scheduler scalars required')
        data=dict(state=current_state.detach().reshape(1,DIM).float(),previous=previous_direction.detach().reshape(1,DIM).float(),
                  scalars=torch.tensor([[sigma,next_sigma,timestep/1000.]],device=self.device,dtype=torch.float32))
        out=self.decode_batch(data).reshape(2,*current_state.shape).to(current_state.dtype)
        if not bool(torch.isfinite(out).all()):raise ValueError('nonfinite draft')
        return out


def load_checkpoint(path,*,device,expected_channels):
    if tuple(expected_channels)!=CHANNEL_IDS:raise ValueError('RoboTwin ordered channel mapping mismatch')
    path=Path(path).resolve(strict=True)
    if path.stat().st_size>2*2**20:raise ValueError('small draft checkpoint required')
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d.get('schema')!=SCHEMA or d.get('specification')!=SPEC or d.get('permissions')!=PERMISSIONS:raise ValueError('RoboTwin checkpoint identity mismatch')
    with torch.random.fork_rng(devices=[]):model=PositionTransformer().float()
    if sum(p.numel() for p in model.parameters())!=56416:raise ValueError('fixed TF64-RT parameter count')
    model.load_state_dict(d['model_state_dict'],strict=True)
    predictor=Predictor(model,d['normalization'],device=device,checkpoint_path=path)
    predictor.metadata=d['metadata']
    return predictor
