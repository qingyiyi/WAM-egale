"""Independent whole-layer pipeline entry; experimental, not production admitted."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import torch
from .whole_layer import WholeLayerBlock

@lru_cache(None)
def extension(root,kind):
    if kind not in ('sync','async','profile'):raise ValueError(kind)
    if torch.version.git_version!='0fabc3ba44823f257e70ce397d989c8de5e362c1':
        raise RuntimeError('pinned Torch2.9 required')
    name='wam_whole_pipeline_'+kind;files=list(Path(root).glob(name+'*.so'))
    if len(files)!=1:raise RuntimeError('build isolated optimized/build_pipeline.py first')
    spec=importlib.util.spec_from_file_location(name,files[0]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m

class PipelineBlock(WholeLayerBlock):
    production_admitted=False
    def __init__(self,block,extension_root,kind='async'):
        super().__init__(block);self.impl=extension(str(Path(extension_root).resolve()),kind)
    def __call__(self,hidden_states,encoder_hidden_states,temb,rotary_emb,
                 update_cache=0,cache_name='pos',candidate_cache_rows=None,debug=False,audit=True):
        if torch.is_grad_enabled():raise RuntimeError('inference_mode required')
        cache=self.block.attn1.attn_caches.get(cache_name)
        values=[] if cache is None else [cache[k] for k in ('k','v','id','mask','is_pred')]
        result=self.impl.forward(self.weights,hidden_states,encoder_hidden_states,temb,rotary_emb,
            values,candidate_cache_rows,update_cache,self.eps,audit)
        return result if debug else result['output']
