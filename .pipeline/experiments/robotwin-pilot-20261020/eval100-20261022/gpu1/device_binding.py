"""Bind the original SAPIEN/PhysX path explicitly to remote GPU1.

CUDA_VISIBLE_DEVICES alone does not specify a Vulkan physical device. This
changes resource selection only; shader, denoiser, physics and cameras stay
at the pinned RoboTwin defaults.
"""
import json,os
from pathlib import Path

def install(out):
    import sapien
    import sapien.core as core
    from sapien.wrapper.scene import Scene
    from sapien.pysapien.render import SapienRenderer as CppRenderer,RenderSystem
    from sapien.pysapien.physx import PhysxCpuSystem,set_scene_config
    pci=os.environ['WAM_RENDER_PCI'];device=sapien.Device('pci:'+pci)
    info=dict(alias='pci:'+pci,pci=device.pci_string,name=device.name,
              can_render=device.can_render(),is_cuda=device.is_cuda(),cuda_id=device.cuda_id)
    Path(out).write_text(json.dumps(info,indent=2)+'\n')
    if str(device.pci_string).lower()[-12:]!=pci.lower()[-12:] or not device.can_render():
        raise RuntimeError('requested GPU1 Vulkan rendering capability/binding unavailable')
    Original=core.SapienRenderer
    class BoundRenderer(Original):
        def __init__(self,**kwargs):
            if kwargs:raise ValueError('unexpected renderer override')
            CppRenderer.__init__(self,device)
    def create_scene(engine,config=None):
        if config is None:
            from sapien.pysapien.physx import PhysxSceneConfig
            config=PhysxSceneConfig()
        set_scene_config(config)
        return Scene([PhysxCpuSystem(),RenderSystem(device)])
    core.SapienRenderer=BoundRenderer
    core.Engine.create_scene=create_scene
    return info
