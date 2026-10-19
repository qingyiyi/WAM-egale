"""只替换FFN第一层Linear+GELU；FFN2保持原Torch。

供静态真实fixture研究显式使用，不进入install_model默认路径。
"""
from contextlib import contextmanager
import torch

from .prepared import extension


class FusedFFN1:
    def __init__(self, module, shape):
        self.module = module
        self.shape = tuple(shape)
        self.weight, self.bias = module.proj.weight, module.proj.bias
        if self.shape not in ((2,16,3072),(14,16,3072)) or module.approximate != 'tanh':
            raise ValueError('只接受固定真实FFN1形状与原tanh GELU')
        self.output = torch.empty((*self.shape[:2],14336),device=self.weight.device,dtype=torch.bfloat16)
        self.native = extension('wam_ffn_cuda')

    def forward(self, x):
        if tuple(x.shape) != self.shape or not x.is_contiguous():
            raise ValueError('静态FFN1形状/布局发生变化')
        self.native.ffn1_gelu_out(x.flatten(0,1),self.weight,self.bias,self.output.flatten(0,1))
        return self.output

    @contextmanager
    def install(self):
        if 'forward' in self.module.__dict__:
            raise ValueError('不与已有forward覆盖混装')
        self.module.forward = self.forward
        try:
            yield
        finally:
            del self.module.forward
