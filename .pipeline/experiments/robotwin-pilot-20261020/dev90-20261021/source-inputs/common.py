"""Pinned model/config/bank setup for independent RoboTwin adapters."""
from pathlib import Path
import os,sys,types,copy
import torch
R=Path('/root/cxz/robotwin');L=R/'source/lingbot-va-pinned'
def setup(output):
    for p in [L,L/'wan_va']:sys.path.insert(0,str(p))
    import wan_va_server as source
    # Source helper is bound before importing the extracted speculative math.
    mesh=types.ModuleType('mesh_helper');mesh.get_mesh_id=source.get_mesh_id;sys.modules['mesh_helper']=mesh
    source.init_logger();cfg=copy.deepcopy(source.VA_CONFIGS['robotwin'])
    cfg.wan22_pretrained_model_name_or_path=str(R/'models/target');cfg.save_root=str(Path(output)/'debug-directories')
    cfg.rank=cfg.local_rank=0;cfg.world_size=1
    assert cfg.num_inference_steps==25 and cfg.action_num_inference_steps==50
    assert cfg.frame_chunk_size==2 and cfg.action_per_frame==16 and cfg.action_snr_shift==1.
    source.init_distributed(1,0,0);server=source.VA_Server(cfg)
    bank=torch.load(R/'runs/target-first6-3/server/text-bank.pt',map_location='cpu',weights_only=False)
    assert bank['source']=='7c6ffa9bfc4b83582cafc860fab4c82cc7deeeeb' and bank['target_revision']=='8c9dea8abbc5c91cc9e18bc3264b8915083bbe70'
    entries=bank['embeddings'];assert all(x.shape==(1,512,4096) and x.dtype==torch.bfloat16 for x in entries.values())
    server.text_encoder.to('cpu');torch.cuda.empty_cache()
    def lookup(_s,prompt=None,num_videos_per_prompt=1,max_sequence_length=512,device=None,dtype=None):
        if isinstance(prompt,list) and len(prompt)==1:prompt=prompt[0]
        assert prompt in entries and num_videos_per_prompt==1 and max_sequence_length==512
        assert dtype in (None,torch.bfloat16)
        return entries[prompt].to(device=device or _s.device,dtype=torch.bfloat16).clone()
    server._get_t5_prompt_embeds=types.MethodType(lookup,server)
    source.save_async=lambda *a,**k:None
    return source,server
