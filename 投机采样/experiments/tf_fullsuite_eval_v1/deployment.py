"""Unhashed deployment of the sealed TF_ON artifact and existing text bank.

Importing this module performs no torch/model/CUDA operation.  Old checkpoint
loaders and TextBank.__init__ are never invoked: those paths calculate hashes.
The original predictor class and bank get/install bodies are compiled without
changes from source bytes compared directly with a local sealed reference.
No file identity is inferred from a size match or from a recorded digest.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import os
from pathlib import Path
import types

EXP = Path(__file__).resolve().parent.parent
NORMALIZATION = {'feature_mean': (227,), 'feature_std': (227,),
    'residual_mean': (112,), 'residual_std': (112,)}
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _path(plan, name, *, fallback=None):
    value = plan.get(name, fallback)
    if isinstance(value, dict):
        value = value.get('remote_path') or value.get('path')
    _require(isinstance(value, (str, Path)) and str(value), 'explicit asset path required: ' + name)
    return Path(value).resolve(strict=True)


def _equal(a, b, message):
    from tf_fullsuite_eval_v1.state import compare_raw
    result = compare_raw(a, b)
    _require(result['equal'], message + ': ' + str(result['mismatches'][:4]))


def _reference(plan):
    import torch
    _require(plan.get('frozen_permissions') == PERMISSIONS, 'old quality permissions must stay false')
    path = _path(plan, 'deployment_reference')
    _require(path.is_file() and path.stat().st_size <= 2 * 2**20, 'bounded locally sealed reference required')
    data = torch.load(path, map_location='cpu', weights_only=True)
    # This campaign reuses the already sealed Long asset unchanged. Its
    # provenance schema remains the original PF2 seal, not a new certificate.
    _require(data.get('schema') == 'tf-pf2-local-sealed-reference/v1'
        and data.get('condition') == 'TF_ON', 'wrong sealed reference condition/schema')
    _require(data.get('torch_version') == str(torch.__version__), 'sealed local/remote torch runtime differs')
    return path, data


def _source(reference, relative, field):
    path = EXP / relative
    data = path.read_bytes()
    _require(isinstance(reference.get(field), bytes) and data == reference[field],
        'raw source bytes differ from local seal: ' + relative)
    return path, data


def _predictor_class(path, raw):
    import torch
    nodes = [n for n in ast.parse(raw, filename=str(path)).body
        if isinstance(n, ast.ClassDef) and n.name == 'ActionPredictor']
    _require(len(nodes) == 1, 'one unchanged original ActionPredictor class required')
    _require({n.name for n in nodes[0].body if isinstance(n, ast.FunctionDef)} == {'__init__', 'predict'},
        'predictor method inventory changed')
    namespace = dict(torch=torch, math=math, NORMALIZATION=NORMALIZATION,
        __name__='tf_fullsuite_eval_v1._original_predictor_math')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['ActionPredictor']


def load_predictor(plan, server):
    """Return (original ActionPredictor, no-hash deployment receipt).

    Requires plan.deployment_reference and selected.remote_path/path.  The seal
    contains original metadata, specification, normalization, model_state_dict,
    original selected_path, torch_version, and the three raw source byte fields.
    Both C/P arms retain checkpoint condition TF_ON; arm is only a schedule.
    """
    import torch
    _require(plan.get('frozen_permissions') == PERMISSIONS, 'old quality permissions must stay false')
    _require(tuple(server.job_config.used_action_channel_ids) == tuple(range(7))
        and int(server.job_config.action_per_frame) == 4
        and int(server.job_config.frame_chunk_size) == 4
        and server.dtype == torch.bfloat16, 'original BF16 compact seven-channel action API required')
    selected = plan['selected']
    path = _path(plan, 'selected')
    _require(path.is_file() and path.stat().st_size == selected['bytes'] <= 2 * 2**20,
        'bounded selected checkpoint byte count differs')
    seal_path, reference = _reference(plan)
    payload = torch.load(path, map_location='cpu', weights_only=True)
    _require(payload.get('schema') == 'tf-effect-draft-3000/v1'
        and payload.get('architecture') == 'transformer64', 'fixed TF_ON checkpoint schema differs')
    _equal(payload.get('metadata'), selected['metadata'], 'checkpoint metadata differs from frozen plan')
    _equal(payload['metadata'], reference['metadata'], 'checkpoint metadata differs from local seal')
    _require(payload['metadata']['condition'] == 'TF_ON'
        and payload['metadata']['permissions'] == PERMISSIONS
        and payload['metadata']['robot_quality_validated'] is False
        and payload['metadata']['deployed'] is False, 'TF_ON provenance/quality flags differ')
    for group in ('specification', 'normalization', 'model_state_dict'):
        _equal(payload[group], reference[group], 'checkpoint payload differs from local seal: ' + group)
    norm = payload['normalization']
    _require(set(norm) == set(NORMALIZATION), 'complete normalizer inventory required')
    for key, shape in NORMALIZATION.items():
        value = norm[key]
        _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == torch.float32 and tuple(value.shape) == shape
            and bool(torch.isfinite(value).all())
            and (not key.endswith('std') or bool((value > 0).all())), 'invalid original normalizer: ' + key)
    predictor_path, predictor_raw = _source(reference,
        'target_trace_draft_train_v1/adapter.py', 'predictor_source_bytes')
    model_path, _ = _source(reference, 'draft_architecture_v1/models.py', 'model_source_bytes')
    spec = importlib.util.spec_from_file_location('tf_fullsuite_eval_v1._original_models', model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _equal(payload['specification'], module.SPECS['transformer64'], 'fixed TF64 specification changed')
    # CPU-only construction must not consume the already initialized target RNG.
    before_cpu = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = module.make_model('transformer64').float()
    _require(torch.equal(before_cpu, torch.get_rng_state()), 'small-model construction consumed target CPU RNG')
    state = payload['model_state_dict']
    _require(set(state) == set(model.state_dict()), 'model parameter inventory differs')
    for key, old in model.state_dict().items():
        value = state[key]
        _require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
            and value.shape == old.shape and bool(torch.isfinite(value).all()), 'model tensor differs: ' + key)
    model.load_state_dict(state, strict=True)
    _require(sum(p.numel() for p in model.parameters()) == 53070, 'frozen TF64 parameter count changed')
    cls = _predictor_class(predictor_path, predictor_raw)
    # recorded digest is provenance text only: no code calculates/validates it.
    predictor = cls(model, norm, architecture='transformer64', checkpoint_path=path,
        checkpoint_sha256=selected.get('expected_sha256', 'UNKNOWN_NO_REMOTE_HASH'), device=server.device)
    predictor.checkpoint_condition = 'TF_ON'
    receipt = dict(schema='tf-fullsuite-unhashed-predictor-deployment/v1', condition='TF_ON',
        checkpoint_path=str(path), local_sealed_selected_path=reference.get('selected_path'),
        checkpoint_bytes=path.stat().st_size, reference_path=str(seal_path),
        metadata_exact=True, normalization_typed_bytes_exact=True, model_typed_bytes_exact=True,
        model_and_predictor_source_raw_bytes_exact=True, parameter_count=53070,
        original_ActionPredictor_class_AST_reused=True, original_decode_math_unchanged=True,
        source_digest_executed=False, checkpoint_digest_executed=False,
        remote_checkpoint_file_byte_verification='UNKNOWN_NO_HASH',
        seal_origin='copied locally sealed reference; no remote cryptographic identity claim',
        runtime_torch_version=str(torch.__version__), no_model_forward=True, permissions=PERMISSIONS)
    predictor.deployment_receipt = receipt
    return predictor, receipt


def load_text_bank(plan, server):
    """Return (bank, receipt, restore); use original get/install math, no hash.

    The local seal contains text_bank_metadata (all original top-level fields
    except embeddings) and text_bank_source_bytes.  Mapping checkpoint_path in
    a new in-memory bank object is explicit; the original bank file is untouched.
    """
    import torch
    seal_path, reference = _reference(plan)
    path = _path(plan, 'bank')
    _require(path.is_file() and path.stat().st_size <= 64 * 2**20, 'bounded original text bank required')
    original = torch.load(path, map_location='cpu', weights_only=True)
    metadata = {k: v for k, v in original.items() if k != 'embeddings'}
    _equal(metadata, reference['text_bank_metadata'], 'bank metadata differs from local seal')
    _require(original.get('schema') == 'long10-original-T5-bank/v1'
        and original.get('sequence_length') == 512 and original.get('dtype') == 'bfloat16'
        and original.get('output_shape') == [1, 512, 4096]
        and original.get('torch_version') == str(torch.__version__), 'original text-bank runtime/schema changed')
    _require(original.get('target_checkpoint_id') == server.job_config.libero_checkpoint_id,
        'bank and actual target checkpoint identities differ')
    prompts = original['public_task_prompts']
    embeddings = original['embeddings']
    _require(set(prompts) == {str(i) for i in range(10)}
        and set(embeddings) == set(prompts.values()) | {''} and len(embeddings) == 11,
        'exact ten Long10 plus empty-negative prompt coverage required')
    for value in embeddings.values():
        _require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == torch.bfloat16 and tuple(value.shape) == (1, 512, 4096)
            and bool(torch.isfinite(value).all()), 'invalid original text embedding')
    bank_source, bank_raw = _source(reference, 'reset_text_bank_v1/bank.py', 'text_bank_source_bytes')
    tree = ast.parse(bank_raw, filename=str(bank_source))
    classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TextBank']
    _require(len(classes) == 1, 'one original bank class required')
    gets = [n for n in classes[0].body if isinstance(n, ast.FunctionDef) and n.name == 'get']
    installs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install']
    _require(len(gets) == len(installs) == 1, 'one unchanged bank get/install pair required')
    bank = types.SimpleNamespace(path=path, data=dict(original), embeddings=embeddings, hits=0)
    original_path = str(original['checkpoint_path'])
    mapped = str(Path(server.job_config.wan22_pretrained_model_name_or_path).resolve(strict=True))
    bank.data['checkpoint_path'] = mapped
    namespace = dict(torch=torch, os=os, Path=Path, types=types, TextBank=object,
        __name__='tf_fullsuite_eval_v1._original_bank_math')
    exec(compile(ast.Module(body=gets + installs, type_ignores=[]), str(bank_source), 'exec'), namespace)
    bank.get = types.MethodType(namespace['get'], bank)
    restore = namespace['install'](server, bank)
    receipt = dict(schema='tf-fullsuite-unhashed-text-bank-deployment/v1', path=str(path), reference_path=str(seal_path),
        original_checkpoint_path=original_path, explicitly_mapped_checkpoint_path=mapped,
        target_checkpoint_id=original['target_checkpoint_id'], metadata_exact_before_mapping=True,
        original_get_install_AST_reused=True, source_raw_bytes_exact=True,
        sequence_length=512, embeddings=11, dtype='bfloat16', runtime_torch_version=str(torch.__version__),
        source_digest_executed=False, remote_bank_file_byte_verification='UNKNOWN_NO_HASH',
        embedding_payload_origin='original copied bank; shape/dtype/finite checked, local payload bytes not independently compared',
        no_T5_forward=True, permissions=PERMISSIONS)
    return bank, receipt, restore
