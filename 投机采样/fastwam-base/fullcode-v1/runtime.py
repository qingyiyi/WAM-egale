"""Pinned official factory/preprocessing imported without training/simulator dependencies."""
import gc
import ast
import inspect
import json
import math
import os
import sys
import time
import textwrap
import zipfile
import resource
from pathlib import Path
from typing import Any, Optional
import numpy as np
from PIL import Image
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from adapter import install, source_function

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'source/FastWAM'
sys.path.insert(0, str(SOURCE / 'src'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def build(assets, out, device='cuda'):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.update(DIFFSYNTH_MODEL_BASE_PATH=str(assets), DIFFSYNTH_SKIP_DOWNLOAD='true',
                      HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    for n, f in [('eval', eval), ('max', lambda x: max(x)), ('split', lambda s, i: s.split('/')[int(i)])]:
        if not OmegaConf.has_resolver(n): OmegaConf.register_new_resolver(n, f)
    with initialize_config_dir(config_dir=str(SOURCE / 'configs'), version_base=None):
        cfg = compose(config_name='sim_libero', overrides=['EVALUATION.sigma_shift=5.0',
                'model.redirect_common_files=false', 'EVALUATION.task_suite_name='+os.environ.get('FASTWAM_SUITE', 'libero_10'), 'seed=42'])
    cfg.EVALUATION.output_dir = str(out)
    cfg.ckpt = str(Path(assets) / 'yuanty/fastwam/libero_uncond_2cam224.pt')
    stats = Path(assets) / 'yuanty/fastwam/libero_uncond_2cam224_dataset_stats.json'
    cfg.EVALUATION.dataset_stats_path = str(stats)
    assert int(cfg.EVALUATION.num_inference_steps) == 10
    assert int(cfg.EVALUATION.num_steps_wait) == 30 and int(cfg.EVALUATION.replan_steps) == 10
    assert not cfg.EVALUATION.visualize_future_video
    OmegaConf.save(cfg, out / 'RESOLVED-CONFIG.yaml', resolve=True)
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    processor = instantiate(cfg.data.train.processor)
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(str(stats)))
    processor.eval()
    ns = dict(torch=torch, np=np, Image=Image, math=math, DictConfig=DictConfig,
              FastWAMProcessor=FastWAMProcessor, Optional=Optional, Any=Any)
    for n in ['get_libero_image', 'quat2axisangle', 'invert_gripper_action']:
        source_function(SOURCE / 'experiments/libero/libero_utils.py', n, ns)
    for n in ['_center_crop_resize', '_normalize_proprio', '_denormalize_action',
              '_extract_sim_state', '_obs_to_model_input']:
        source_function(SOURCE / 'experiments/libero/eval_libero_single.py', n, ns)
    factory = source_function(SOURCE / 'src/fastwam/runtime.py', 'create_fastwam',
                              dict(torch=torch, DictConfig=DictConfig, OmegaConf=OmegaConf, __package__='fastwam'))
    loads = []
    load_files = []
    original = torch.nn.Module.load_state_dict
    original_torch_load = torch.load
    from fastwam.models.wan22.helpers import loader as component_loader
    original_component_load = component_loader._load_registered_model
    loader_tree = ast.parse(textwrap.dedent(inspect.getsource(original_component_load)))
    loader_fn = loader_tree.body[0]
    loader_fn.name = '_load_registered_model_low_memory'
    new_body = []
    replacements = 0
    for node in loader_fn.body:
        if ast.unparse(node) == 'model = model_class(**model_kwargs)':
            new_body.extend(ast.parse('''if model_name == "wan_video_text_encoder":
    with torch.device("meta"):
        model = model_class(**model_kwargs)
    assert not list(model.named_buffers()), "T5 meta construction requires no non-checkpoint buffers"
else:
    model = model_class(**model_kwargs)
print("COMPONENT_LOAD", model_name, flush=True)
''').body)
            replacements += 1
        elif ast.unparse(node) == 'model.load_state_dict(state_dict, strict=False)':
            new_body.extend(ast.parse('model.load_state_dict(state_dict, strict=False, assign=(model_name == "wan_video_text_encoder"))').body)
            replacements += 1
        else:
            new_body.append(node)
    assert replacements == 2
    loader_fn.body = new_body
    loader_tree = ast.fix_missing_locations(loader_tree)
    (out/'LOW-MEMORY-LOADER-SOURCE.py').write_text(ast.unparse(loader_tree)+'\n')
    loader_namespace = dict(original_component_load.__globals__)
    exec(compile(loader_tree, str(out/'LOW-MEMORY-LOADER-SOURCE.py'), 'exec'), loader_namespace)
    component_loader._load_registered_model = loader_namespace[loader_fn.name]
    def mapped_load(path, *a, **kw):
        mapped = isinstance(path, (str, os.PathLike)) and zipfile.is_zipfile(path)
        if mapped: kw.setdefault('mmap', True)
        load_files.append(dict(path=str(path), mmap=bool(kw.get('mmap', False))))
        print('CHECKPOINT_LOAD', str(path), 'mmap', mapped, flush=True)
        return original_torch_load(path, *a, **kw)
    torch.load = mapped_load
    def strict(module, state, *a, **kw):
        result = original(module, state, *a, **kw)
        row = dict(module=type(module).__name__, keys=len(state), missing=list(result.missing_keys),
                   unexpected=list(result.unexpected_keys))
        loads.append(row)
        assert not row['missing'] and not row['unexpected'], row
        return result
    torch.nn.Module.load_state_dict = strict
    start = time.perf_counter()
    try:
        print('MODEL_FACTORY_BEGIN', flush=True)
        mc = OmegaConf.to_container(cfg.model, resolve=True)
        mc.pop('_target_')
        model = factory(**mc, model_dtype=torch.bfloat16, device=device)
        payload = model.load_checkpoint(cfg.ckpt)
        del payload
        gc.collect()
    finally:
        torch.nn.Module.load_state_dict = original
        torch.load = original_torch_load
        component_loader._load_registered_model = original_component_load
    model.eval().requires_grad_(False)
    install(model, SOURCE, out / 'ADAPTED-INFER-SOURCE.py')
    torch.cuda.synchronize()
    write(out / 'MODEL-LOAD.json', dict(seconds=time.perf_counter()-start, strict_inventory=loads,
           torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
           allocated_bytes=torch.cuda.memory_allocated(), model_paths=model.model_paths,
           low_memory_loading=dict(T5_meta_assign=True, mmap_files=load_files,
                                   peak_host_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                                   scope='loading only; all checkpoint keys required; T5 has no non-checkpoint buffers'),
           tf32_matmul=torch.backends.cuda.matmul.allow_tf32, tf32_cudnn=torch.backends.cudnn.allow_tf32))
    return model, processor, cfg, ns


def inputs(obs, task, model, processor, cfg, ns):
    image, proprio, _ = ns['_obs_to_model_input'](obs, cfg=cfg, processor=processor,
                                               width=448, height=224, device='cuda', dtype=model.torch_dtype)
    return dict(prompt="A video recorded from a robot's point of view executing the following instruction: " + task,
                input_image=image, proprio=proprio, action_horizon=32, num_inference_steps=10,
                sigma_shift=5., seed=42, rand_device='cpu', tiled=False, text_cfg_scale=1., negative_prompt='')


def postprocess(action, processor, ns):
    a = ns['_denormalize_action'](action, processor)[0]
    a[..., -1] = a[..., -1] * 2 - 1
    a = ns['invert_gripper_action'](a)
    a[..., -1] = np.sign(a[..., -1])
    assert a.shape == (32, 7) and np.isfinite(a).all()
    return a
