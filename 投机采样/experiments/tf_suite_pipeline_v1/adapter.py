"""Suite-specific TF64 artifacts and original production prediction, without hashes.

No torch import, model construction, prediction, or device operation runs at
module import. Numerical classes are extracted from frozen project sources;
legacy checkpoint loaders, preparation functions and synthetic tests are not.
"""
from __future__ import annotations

import ast
import math
from pathlib import Path
import types

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
MODELS = EXP/'draft_architecture_v1/models.py'
PRODUCTION = EXP/'target_trace_draft_train_v1/adapter.py'
SCHEMA = 'tf-suite-draft-checkpoint/v1'
PLAN_SCHEMA = 'tf-suite-fixed-stage-training-plan/v1'
DATA_SCHEMA = 'tf-suite-draft-columns/v1'
INDEX_SCHEMA = 'tf-suite-draft-columns-index/v1'
SEED = 20260923
TAU = 0.007839680526450805
ARCHITECTURE = 'transformer64'
PARAMETERS = 53070
CONDITIONS = ('PARENT2000', 'CHILD2500', 'CONT3000', 'ON3000')
UPDATES = dict(PARENT2000=2000, CHILD2500=2500, CONT3000=3000, ON3000=3000)
PARENTS = dict(CHILD2500='PARENT2000', CONT3000='CHILD2500', ON3000='CHILD2500')
PERMISSIONS = dict(engineering_quality_screen=False, independent_test_complete=False,
    noninferiority_established=False, formal_performance_permission=False)
NORMALIZATION = dict(feature_mean=(227,), feature_std=(227,),
    residual_mean=(112,), residual_std=(112,))
SPECIFICATION = dict(kind='transformer', width=64, heads=4, ffn=256, layers=1,
    dropout=0., tokens=16, activation='gelu', norm_first=True)
_MODELS = _PREDICTOR = None


def require(value, message):
    if not value:
        raise ValueError(message)


def suite_short(value):
    value = str(value)
    result = value[7:] if value.startswith('libero_') else value
    require(result in ('object', 'goal', 'spatial'), 'registered new suite required')
    return result


def model_source():
    """Compile original architecture definitions; do not import a train runner."""
    global _MODELS
    if _MODELS is not None:
        return _MODELS
    import torch
    wanted = {'ResidualMLP', 'position_tokens', 'PositionTransformer', 'make_model', 'Predictor'}
    tree = ast.parse(MODELS.read_bytes(), filename=str(MODELS))
    nodes = [n for n in tree.body if
        isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in wanted or
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SPECS' for t in n.targets)]
    require({n.name for n in nodes if isinstance(n, (ast.ClassDef, ast.FunctionDef))} == wanted,
        'original architecture definition inventory changed')
    namespace = dict(torch=torch, nn=torch.nn, math=math, Path=Path,
        __name__='tf_suite_pipeline_v1._original_architecture')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(MODELS), 'exec'), namespace)
    require(namespace['SPECS'][ARCHITECTURE] == SPECIFICATION, 'fixed TF64 specification changed')
    _MODELS = types.SimpleNamespace(**namespace)
    return _MODELS


def predictor_class():
    """Retain the full original ActionPredictor class, including predict math."""
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    import torch
    nodes = [n for n in ast.parse(PRODUCTION.read_bytes()).body
        if isinstance(n, ast.ClassDef) and n.name == 'ActionPredictor']
    require(len(nodes) == 1 and {n.name for n in nodes[0].body if isinstance(n, ast.FunctionDef)}
        == {'__init__', 'predict'}, 'original production predictor inventory changed')
    namespace = dict(torch=torch, math=math, NORMALIZATION=NORMALIZATION,
        __name__='tf_suite_pipeline_v1._original_predictor')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PRODUCTION), 'exec'), namespace)
    _PREDICTOR = namespace['ActionPredictor']
    return _PREDICTOR


def typed_equal(left, right):
    import torch
    return (isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
        and left.dtype == right.dtype and left.shape == right.shape
        and torch.equal(left.detach().contiguous().reshape(-1).view(torch.uint8),
            right.detach().to(left.device).contiguous().reshape(-1).view(torch.uint8)))


def validate_normalization(norm):
    import torch
    require(set(norm) == set(NORMALIZATION), 'complete original normalizer inventory required')
    for key, shape in NORMALIZATION.items():
        x = norm[key]
        require(isinstance(x, torch.Tensor) and x.device.type == 'cpu'
            and x.dtype == torch.float32 and tuple(x.shape) == shape and bool(torch.isfinite(x).all())
            and (not key.endswith('std') or bool((x > 0).all())), 'invalid normalizer: ' + key)


def validate_metadata(metadata, *, suite, condition=None, expected_metadata=None):
    short = suite_short(suite)
    require(isinstance(metadata, dict) and metadata.get('suite_short') == short
        and metadata.get('suite') == 'libero_' + short, 'checkpoint suite identity differs')
    condition = condition or metadata.get('condition')
    require(condition in CONDITIONS and metadata.get('condition') == condition
        and metadata.get('seed') == SEED and metadata.get('total_updates') == UPDATES[condition]
        and metadata.get('architecture') == ARCHITECTURE and metadata.get('parameter_count') == PARAMETERS
        and metadata.get('batch') == 512 and metadata.get('weight_decay') == .01
        and metadata.get('optimizer_state_reset') is True
        and metadata.get('normalization_refit') is False
        and metadata.get('normalization_source') == 'TARGET200.FIT180'
        and metadata.get('selection') == 'fixed_final_update_no_CAL_or_seed_selection'
        and metadata.get('permissions') == PERMISSIONS
        and metadata.get('training_updates_complete') is True
        and metadata.get('normalization_fit_once') is (condition == 'PARENT2000')
        and isinstance(metadata.get('train_job'), str) and metadata['train_job']
        and metadata.get('robot_quality_validated') is False and metadata.get('deployed') is False,
        'fixed suite training metadata/quality flags differ')
    if condition == 'PARENT2000':
        require(metadata.get('initial_total_updates') == 0 and metadata.get('updates') == 2000
            and metadata.get('learning_rate') == 1e-4 and metadata.get('target_half') == 512
            and metadata.get('observer_half') == 0 and metadata.get('parent_checkpoint') is None,
            'pure-target parent2000 recipe differs')
    else:
        require(metadata.get('initial_total_updates') == UPDATES[PARENTS[condition]]
            and metadata.get('parent_condition') == PARENTS[condition]
            and metadata.get('updates') == 500 and metadata.get('learning_rate') == 3e-5
            and metadata.get('target_half') == metadata.get('observer_half') == 256
            and isinstance(metadata.get('parent_checkpoint'), str), 'fixed mixed500 recipe differs')
    require(isinstance(metadata.get('target_checkpoint_id'), str) and metadata['target_checkpoint_id']
        and isinstance(metadata.get('target_checkpoint_path'), str) and metadata['target_checkpoint_path'],
        'actual suite-specific target checkpoint provenance required')
    if expected_metadata is not None:
        require(metadata == expected_metadata, 'checkpoint metadata differs from selected deployment manifest')
    return condition


def load_checkpoint(path, *, architecture=ARCHITECTURE, device, expected_channels=range(7),
                    suite, condition=None, expected_metadata=None):
    """Load a fixed suite artifact; no old loader, digest or extra forward."""
    import torch
    channels = tuple(range(expected_channels)) if isinstance(expected_channels, int) else tuple(expected_channels)
    require(architecture == ARCHITECTURE and channels == tuple(range(7)), 'fixed TF64/seven ordered channels required')
    path = Path(path).resolve(strict=True)
    require(path.is_file() and path.stat().st_size <= 2*2**20, 'bounded small draft checkpoint required')
    data = torch.load(path, map_location='cpu', weights_only=True)
    require(data.get('schema') == SCHEMA and data.get('architecture') == ARCHITECTURE
        and data.get('specification') == SPECIFICATION, 'suite checkpoint schema/specification differs')
    condition = validate_metadata(data.get('metadata'), suite=suite, condition=condition,
        expected_metadata=expected_metadata)
    norm = data.get('normalization', {})
    validate_normalization(norm)
    source = model_source()
    with torch.random.fork_rng(devices=[]):
        model = source.make_model(ARCHITECTURE).float()
    require(sum(p.numel() for p in model.parameters()) == PARAMETERS, 'fixed TF64 capacity differs')
    weights = data.get('model_state_dict', {})
    require(set(weights) == set(model.state_dict()), 'complete TF parameter inventory required')
    for key, ref in model.state_dict().items():
        value = weights[key]
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
            and value.dtype == torch.float32 and value.shape == ref.shape
            and bool(torch.isfinite(value).all()), 'invalid TF model tensor: ' + key)
    model.load_state_dict(weights, strict=True)
    device = torch.device(device)
    require(device.type in ('cpu', 'cuda'), 'CPU or assigned CUDA predictor required')
    result = predictor_class()(model, norm, architecture=ARCHITECTURE,
        checkpoint_path=path, checkpoint_sha256='UNKNOWN_USER_NO_HASH', device=device)
    result.checkpoint_condition = condition
    result.suite_short = suite_short(suite)
    result.suite = 'libero_' + result.suite_short
    result.metadata = data['metadata']
    result.deployment_receipt = dict(schema='tf-suite-no-hash-draft-deployment/v1',
        checkpoint=str(path), checkpoint_bytes=path.stat().st_size, suite=result.suite,
        condition=condition, architecture=ARCHITECTURE, parameters=PARAMETERS,
        original_ActionPredictor_class_AST_reused=True, original_predict_math_unchanged=True,
        source_and_checkpoint_SHA_executed=False, checkpoint_file_byte_identity='UNKNOWN_NO_HASH',
        model_forward_executed_by_loader=False, old_CAL_certificate_inherited=False,
        permissions=PERMISSIONS, robot_quality_validated=False)
    return result


class InlineObserver:
    """Check original arithmetic using one actual forward; never create queries.

    This is a bounded observed input/decode contract, not CPU/GPU model parity
    or a CAL/robot certificate. Observation errors do not replace returned actions.
    """
    def __init__(self, predictor, *, max_audited_queries=32):
        require(type(max_audited_queries) is int and 0 <= max_audited_queries <= 256,
            'bounded real-query inline observation required')
        self.predictor, self.limit = predictor, max_audited_queries
        self.total_calls, self.records = 0, []
        self.original = predictor.predict
        self.closed = False
        self.had_predict, self.previous_predict = 'predict' in vars(predictor), vars(predictor).get('predict')
        self.bound_predict = types.MethodType(self.predict, predictor)
        predictor.predict = self.bound_predict

    def predict(self, _predictor, state, previous, sigma, next_sigma, timestep):
        import torch
        import time
        self.total_calls += 1
        if len(self.records) >= self.limit:
            return self.original(state, previous, sigma, next_sigma, timestep)
        started, handles, observed = time.monotonic(), [], dict(forwards=0)
        record = dict(actual_call=self.total_calls, status='UNKNOWN', extra_teacher_forwards=0,
            extra_draft_forwards=0, same_device_arithmetic_only=True, source_and_checkpoint_SHA_executed=False)
        output, error = None, None
        try:
            before_cpu = torch.get_rng_state().clone()
            before_cuda = (torch.cuda.get_rng_state(self.predictor.device).clone()
                if self.predictor.device.type == 'cuda' else None)

            def pre(_model, args):
                observed['forwards'] += 1
                observed['normalized_input'] = args[0].detach()

            def post(_model, _args, result):
                if result is not None:
                    observed['raw'] = result.detach()

            handles.append(self.predictor.model.register_forward_pre_hook(pre))
            handles.append(self.predictor.model.register_forward_hook(post, always_call=True))
        except Exception as exc:
            record['observer_setup_error'] = repr(exc)
        try:
            output = self.original(state, previous, sigma, next_sigma, timestep)
        except BaseException as exc:
            error = exc
            record.update(status='ORIGINAL_PREDICT_FAILED', original_error=repr(exc))
            raise
        finally:
            for handle in handles:
                try:
                    handle.remove()
                except Exception as exc:
                    record['observer_remove_error'] = repr(exc)
            if error is not None:
                record['observed_model_forwards'] = observed['forwards']
                self.records.append(record)
        try:
            require('observer_setup_error' not in record, 'inline observer setup unavailable')
            require('observer_remove_error' not in record, 'inline observer removal unavailable')
            require(observed['forwards'] == 1 and 'raw' in observed, 'one actual draft forward required')
            x = state.detach().reshape(1, 112).float()
            p = previous.detach().reshape(1, 112).float()
            times = torch.tensor([[sigma, next_sigma, timestep/1000.]], device=state.device, dtype=torch.float32)
            expected_input = (torch.cat((x, p, times), 1)-self.predictor.feature_mean)/self.predictor.feature_std
            expected = (p[:, None] + observed['raw']*self.predictor.residual_std[None, None]
                + self.predictor.residual_mean[None, None]).reshape(2, *state.shape).to(state.dtype)
            normalized_equal = typed_equal(observed['normalized_input'], expected_input)
            decoded_equal = typed_equal(output, expected)
            rng_equal = torch.equal(before_cpu, torch.get_rng_state()) and (
                before_cuda is None or torch.equal(before_cuda, torch.cuda.get_rng_state(self.predictor.device)))
            dtype_ok = (state.dtype in (torch.bfloat16,torch.float32)
                and previous.dtype in (torch.bfloat16,torch.float32) and output.dtype == state.dtype)
            shape_ok = tuple(output.shape) == (2, *state.shape)
            params_ok = all(p.dtype == torch.float32 for p in self.predictor.model.parameters())
            record.update(status='PASS_OBSERVED_REAL_QUERY_CONTRACT' if
                normalized_equal and decoded_equal and rng_equal and dtype_ok and shape_ok and params_ok else
                'INVALID_OBSERVED_REAL_QUERY_CONTRACT', input_normalization_typed_bytes_exact=normalized_equal,
                decoded_output_typed_bytes_exact=decoded_equal, Torch_CPU_CUDA_RNG_unchanged=rng_equal,
                actual_state_dtype=str(state.dtype),actual_previous_dtype=str(previous.dtype),
                actual_output_dtype=str(output.dtype),original_input_dtype_contract=dtype_ok,
                original_output_preserves_state_dtype=output.dtype == state.dtype,
                output_shape_valid=shape_ok, FP32_draft_parameters=params_ok,
                observed_model_forwards=observed['forwards'], device=str(state.device),
                CPU_GPU_model_direction_parity='NOT_TESTED', elapsed_seconds=time.monotonic()-started)
        except Exception as exc:
            record.update(status='UNKNOWN_INLINE_OBSERVATION_ERROR', observer_error=repr(exc),
                observed_model_forwards=observed['forwards'])
        self.records.append(record)
        return output

    def summary(self):
        return dict(schema='tf-suite-inline-real-query-contract/v1', actual_predict_calls=self.total_calls,
            observed_queries=len(self.records), observation_limit=self.limit,
            records=list(self.records), extra_teacher_forwards=0, extra_draft_forwards=0,
            no_independent_CAL_or_synthetic_test=True, CPU_GPU_model_parity='UNKNOWN_NOT_TESTED',
            old_CAL_certificate_inherited=False, robot_quality_validated=False, permissions=PERMISSIONS)

    def close(self):
        if self.closed:
            return
        require(self.predictor.predict == self.bound_predict, 'restore inline observers in reverse order')
        if self.had_predict:
            self.predictor.predict = self.previous_predict
        else:
            del self.predictor.predict
        self.closed = True


def install_inline_observer(predictor, *, max_audited_queries=32):
    return InlineObserver(predictor, max_audited_queries=max_audited_queries)
