"""Bounded infrastructure continuation; no additional CONT optimizer step.

The original v1 source is read and transformed in memory, never changed. Its
non-CONT training AST and production checkpoint/adapter schema remain intact.
No job, model call, CUDA context, or training occurs at import.
"""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys
import time

OWN = Path(__file__).resolve().parent
EXP = OWN.parent
OLD_SOURCE = EXP / 'tf_effect_train_v1/train.py'
OLD_ADAPTER = EXP / 'tf_effect_train_v1/adapter.py'
if str(EXP) not in sys.path:
    sys.path.append(str(EXP))


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _call_name(node, name):
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == name


def _statement_call(node, name):
    return isinstance(node, ast.Expr) and _call_name(node.value, name)


def _write_to(node, filename):
    return (_statement_call(node, 'write') and node.value.args
        and isinstance(node.value.args[0], ast.BinOp)
        and isinstance(node.value.args[0].op, ast.Div)
        and isinstance(node.value.args[0].right, ast.Constant)
        and node.value.args[0].right.value == filename)


def _torch_save_to(node, filename):
    return (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute) and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == 'torch' and node.value.func.attr == 'save'
        and len(node.value.args) == 2 and isinstance(node.value.args[1], ast.BinOp)
        and isinstance(node.value.args[1].op, ast.Div)
        and isinstance(node.value.args[1].right, ast.Constant)
        and node.value.args[1].right.value == filename)


def _injected(code):
    return ast.parse(code).body


def transform():
    """Build+reverse review without importing torch or running any function.

    All inserted infrastructure calls use a unique __reuse_v2_ prefix. The
    sole original-code replacement wraps the unchanged training statements in
    the non-CONT else branch; inverse transformation must recover exact ASTs.
    """
    parsed = ast.parse(OLD_SOURCE.read_text(), filename=str(OLD_SOURCE))
    original = {node.name: node for node in parsed.body
        if isinstance(node, ast.FunctionDef) and node.name in ('prepare', 'run')}
    require(set(original) == {'prepare', 'run'}, 'one frozen prepare/run pair required')
    modified = copy.deepcopy(original)

    class Prepare(ast.NodeTransformer):
        def visit_Expr(self, node):
            if _write_to(node, 'PREREGISTRATION.json'):
                return _injected('__reuse_v2_register(registration, args)') + [node]
            if _write_to(node, 'PLAN.json'):
                return _injected('__reuse_v2_plan(job, plan, args)') + [node]
            if _write_to(node, 'PREPARED.json'):
                return _injected('__reuse_v2_prepared(record, plan)') + [node]
            return self.generic_visit(node)

    modified['prepare'] = Prepare().visit(modified['prepare'])

    class Run(ast.NodeTransformer):
        def visit_For(self, node):
            node = self.generic_visit(node)
            # Only the fixed condition loop contains the original model.train,
            # optimizer construction, log init, and update loop consecutively.
            hits = [index for index, value in enumerate(node.body)
                if isinstance(value, ast.Expr) and ast.unparse(value) == 'predictor.model.train()']
            if not hits:
                return node
            require(len(hits) == 1, 'one training sequence required')
            index = hits[0]
            sequence = node.body[index:index+4]
            require(len(sequence) == 4 and isinstance(sequence[1], ast.Assign)
                and isinstance(sequence[2], ast.Assign) and isinstance(sequence[3], ast.For)
                and ast.unparse(sequence[1].targets[0]) == 'optimizer'
                and ast.unparse(sequence[2].targets[0]) == 'logs'
                and ast.unparse(sequence[3].target) == '(update, (target_ids, same_ids))',
                'frozen optimizer/log/update block differs')
            branch = ast.If(test=ast.parse("condition == 'TF_CONT'", mode='eval').body,
                body=_injected('''
__reuse_v2_restore(predictor, plan)
active_updates = 500
logs = []
'''), orelse=sequence)
            node.body[index:index+4] = [branch]
            return node

        def visit_Expr(self, node):
            if _torch_save_to(node, 'draws.pt'):
                return _injected('__reuse_v2_draws(draws, plan)') + [node]
            if _write_to(node, 'RESULT.json'):
                return _injected('__reuse_v2_record(record, condition, plan)') + [node]
            if _write_to(node, 'FINAL-RESULT.json'):
                return _injected('__reuse_v2_final(final, plan)') + [node]
            return self.generic_visit(node)

    modified['run'] = Run().visit(modified['run'])

    class Reverse(ast.NodeTransformer):
        def visit_Expr(self, node):
            if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                    and node.value.func.id.startswith('__reuse_v2_')):
                return None
            return self.generic_visit(node)

        def visit_If(self, node):
            if (node.body and _statement_call(node.body[0], '__reuse_v2_restore')):
                return node.orelse
            return self.generic_visit(node)

    proof = {}
    for name, node in modified.items():
        restored = Reverse().visit(copy.deepcopy(node))
        reference = ast.dump(original[name], include_attributes=False)
        actual = ast.dump(restored, include_attributes=False)
        require(reference == actual, 'inverse transformation changed original ' + name)
        proof[name] = dict(original_ast_sha256=hashlib.sha256(reference.encode()).hexdigest(),
            reversed_ast_sha256=hashlib.sha256(actual.encode()).hexdigest(), exact_inverse=True)
    # Assert precisely one CONT bypass and each infrastructure receipt hook.
    combined = ast.Module(body=list(modified.values()), type_ignores=[])
    counts = {name: sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        for n in ast.walk(combined)) for name in ('__reuse_v2_register', '__reuse_v2_plan',
            '__reuse_v2_prepared', '__reuse_v2_draws', '__reuse_v2_restore', '__reuse_v2_record', '__reuse_v2_final')}
    require(all(value == 1 for value in counts.values()), 'unique infrastructure injection inventory differs')
    return ast.fix_missing_locations(combined), dict(schema='tf-effect-reuse-ast-review/v2',
        source=str(OLD_SOURCE), source_sha256=sha(OLD_SOURCE), functions=proof,
        injection_counts=counts, non_CONT_optimizer_training_AST_unchanged=True,
        no_batch_size_dtype_or_acceptance_change=True)


def closed_predecessor(path):
    """No CUDA/model call; reject partial ON work before admitting a retry."""
    path = Path(path).resolve(strict=True)
    failed = read(path / 'FAILED.json')
    started = read(path / 'STARTED.json')
    require(not Path('/proc', str(started['pid'])).exists(), 'predecessor PID must be fully exited before continuation')
    require(failed.get('status') == 'FAILED_FIXED_TF_EFFECT_TRAINING'
        and ((failed.get('condition') == 'TF_CONT' and failed.get('updates_completed') == 500)
            or (failed.get('condition') == 'TF_ON' and failed.get('updates_completed') == 0))
        and failed.get('complete_runs') in (0, 1),
        'only complete500 CONT and zero ON/MLP updates can be continued without repeated training')
    require((path / 'STOP_REQUESTED').exists(), 'explicit stop prevents hidden partially trained ON')
    require(not any((path / f'{name}-seed20260923').exists() for name in ('TF_ON', 'MLP_ON')),
        'ON/MLP artifact exists: cannot claim first500 updates')
    require((path / 'TF_CONT-seed20260923/checkpoint.pt').is_file(), 'actual completed500 CONT checkpoint required')
    return path, failed


def register(registration, args):
    prior, failed = closed_predecessor(args.prior_job)
    require(args.wall_cap_seconds == 1800 and args.storage_cap_mib == 256 and args.include_mlp_on,
        'fixed new1800sec/256MiB and the same three conditions required')
    require(args.master_deadline_epoch > time.time(), 'master/user absolute deadline already expired')
    registration['deadline_epoch'] = min(registration['deadline_epoch'], args.master_deadline_epoch)
    registration.update(infrastructure_continuation=True, original_failed_job=str(prior),
        original_FAILED_sha256=sha(prior / 'FAILED.json'), original_budget_not_extended=True,
        predecessor_elapsed_seconds=failed['elapsed_since_registration'], added_wall_budget_seconds=1800,
        master_deadline_epoch=args.master_deadline_epoch, inference_CPU_threads=1, training_CPU_threads=2,
        new_optimizer_updates=dict(TF_CONT=0, TF_ON=500, MLP_ON=500),
        completed_prior_optimizer_updates=dict(TF_CONT=500, TF_ON=0, MLP_ON=0),
        terminal_status='REGISTERED_INFRASTRUCTURE_CONTINUATION_NOT_PREPARED')


def augment_plan(job, plan, args):
    import torch
    from tf_effect_train_v1.adapter import (SCHEMA, GENERATOR_SHA, PARENT_SHAS, PERMISSIONS,
        CONDITIONS, verify_plan, model_source)
    from tf_effect_train_v1.train import OLD_CONT, write
    prior, failed = closed_predecessor(args.prior_job)
    old_plan = verify_plan(prior / 'PLAN.json')
    require(old_plan['conditions'] == plan['conditions'] == ['TF_CONT', 'TF_ON', 'MLP_ON']
        and old_plan['new_manifest_sha256'] == plan['new_manifest_sha256']
        and old_plan['input_stat'] == plan['input_stat'], 'same fixed data/split/draw inputs required')
    for key in ('seed', 'initial_total_updates', 'continuation_updates', 'total_updates', 'batch',
            'target_half', 'same_query_half', 'learning_rate', 'weight_decay', 'normalization_refit', 'optimizer_state_reset', 'cells'):
        require(old_plan[key] == plan[key], 'continuation changes scientific recipe: ' + key)
    borrowed = prior / 'TF_CONT-seed20260923/checkpoint.pt'
    payload = torch.load(borrowed, map_location='cpu', weights_only=True)
    known = torch.load(OLD_CONT, map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    require(sha(OLD_CONT) == GENERATOR_SHA and payload['schema'] == SCHEMA
        and payload['architecture'] == 'transformer64'
        and payload['specification'] == model_source().SPECS['transformer64']
        and metadata['condition'] == 'TF_CONT' and metadata['seed'] == 20260923
        and metadata['initial_total_updates'] == 2500 and metadata['continuation_updates'] == 500
        and metadata['total_updates'] == 3000 and metadata['parent_checkpoint_sha256'] == PARENT_SHAS['transformer64']
        and metadata['plan_sha256'] == sha(prior / 'PLAN.json')
        and metadata['permissions'] == PERMISSIONS, 'borrowed completed500 checkpoint identity differs')
    for group in ('model_state_dict', 'normalization'):
        require(set(payload[group]) == set(known[group]), 'borrowed model/norm inventory differs')
        for key, value in payload[group].items():
            reference = known[group][key]
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float32
                and value.shape == reference.shape and bool(torch.isfinite(value).all())
                and torch.equal(value, reference), 'borrowed completed500 tensor differs: ' + group + '/' + key)
    require(torch.equal(torch.load(prior / 'draws.pt', map_location='cpu', weights_only=True)['target'],
        torch.load(Path(old_plan['cells']['TF_CONT']['parent_checkpoint']).parent.parent / 'draws.pt',
            map_location='cpu', weights_only=True)['20260923']['target']), 'predecessor target draws differ')
    tree, review = transform()
    write(job / 'AST-INVERSE-REVIEW.json', review)
    lineage = dict(schema='tf-effect-completed-CONT-artifact-reuse/v2', prior_job=str(prior),
        checkpoint=str(borrowed), checkpoint_sha256=sha(borrowed), original_plan_sha256=sha(prior / 'PLAN.json'),
        original_FAILED_sha256=sha(prior / 'FAILED.json'), existing_complete_CONT_checkpoint=str(OLD_CONT),
        existing_complete_CONT_sha256=GENERATOR_SHA, all_model_and_normalization_tensors_bitwise_exact=True,
        optimizer_updates_already_completed=500, optimizer_updates_this_job=0,
        original_FAILED_retained=True, original_budget_not_extended=True,
        new_CPU_validation_threads=1, previous_CPU_validation_threads=2,
        previous_CPU2_output_bitwise_parity_inherited=False, production_GPU_CAL_unchanged=True)
    write(job / 'ARTIFACT-REUSE-PROOF.json', lineage)
    new_files = (OWN / 'train.py', OWN / 'README.md')
    plan['source_sha256'].update({str(path): sha(path) for path in new_files})
    evidence = (prior / 'PREREGISTRATION.json', prior / 'PLAN.json', prior / 'STARTED.json',
        prior / 'FAILED.json', prior / 'STOP_REQUESTED', prior / 'draws.pt', borrowed,
        job / 'AST-INVERSE-REVIEW.json', job / 'ARTIFACT-REUSE-PROOF.json')
    plan['evidence_sha256'].update({str(path): sha(path) for path in evidence})
    plan.update(infrastructure_continuation_schema='tf-effect-cpu-reuse/v2', artifact_reuse=lineage,
        training_CPU_threads=2, validation_CPU_threads=1, validation_batch_size=1,
        old_CPU2_parity_inherited=False, original_training_job=str(prior),
        added_wall_budget_seconds=1800, original_elapsed_seconds=failed['elapsed_since_registration'],
        master_deadline_epoch=args.master_deadline_epoch,
        new_optimizer_updates=dict(TF_CONT=0, TF_ON=500, MLP_ON=500),
        before_old_CAL_CPU_reference_threads=2, after_old_CAL_CPU_reference_threads=1,
        old_CAL_before_after_thread_mismatch_is_descriptive_only=True)


def restore(predictor, plan):
    import torch
    lineage = plan['artifact_reuse']
    require(sha(lineage['checkpoint']) == lineage['checkpoint_sha256'], 'borrowed source bytes changed')
    payload = torch.load(lineage['checkpoint'], map_location='cpu', weights_only=True)
    for key, value in payload['normalization'].items():
        require(torch.equal(value, getattr(predictor, key)), 'borrowed normalizer differs from actual child2500')
    predictor.model.load_state_dict(payload['model_state_dict'], strict=True)
    predictor.model.eval()
    require(all(torch.equal(value, payload['model_state_dict'][key])
        for key, value in predictor.model.state_dict().items()), 'CONT restore serialization differs')


def prepared(record, plan):
    record['source_files_original'] = record['source_files']
    record['source_files'] = len(plan['source_sha256'])
    record.update(infrastructure_continuation=True, original_failed_job=plan['original_training_job'],
        training_CPU_threads=2, validation_CPU_threads=1, original_FAILED_retained=True)


def draws(draws, plan):
    import torch
    original = torch.load(Path(plan['original_training_job']) / 'draws.pt', map_location='cpu', weights_only=True)
    require(set(draws) == set(original) == {'target', 'old_same', 'new_same'}, 'draw inventory changed')
    for key, value in draws.items():
        require(value.shape == (500, 256) and value.dtype == torch.int64 and torch.equal(value, original[key]),
            'all500 original draws changed: ' + key)


@contextmanager
def validation_threads():
    import torch
    old = torch.get_num_threads()
    require(old == 2, 'training engine must be CPU2 before a CPU1 validation context')
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def record(record, condition, plan):
    record.update(infrastructure_continuation_schema=plan['infrastructure_continuation_schema'],
        training_CPU_threads=2, validation_CPU_threads=1, validation_batch_size=1,
        actual_optimizer_updates_this_job=plan['new_optimizer_updates'][condition],
        model_total_continuation_updates=500, original_failed_job=plan['original_training_job'],
        CONT_reuse_proof=plan['artifact_reuse'] if condition == 'TF_CONT' else None,
        CPU2_output_bitwise_parity_inherited=False,
        full_CPU1_adapter_vs_original_decode_bitwise_exact=True,
        old_CAL_before_after_thread_mismatch_is_descriptive_only=True,
        old_CAL_before_CPU_reference_threads=2, after_CPU_reference_threads=1,
        production_GPU_CAL_still_required=True)


def final(final, plan):
    final.update(infrastructure_continuation=True, original_failed_job=plan['original_training_job'],
        original_elapsed_seconds=plan['original_elapsed_seconds'], added_wall_budget_seconds=1800,
        cumulative_old_and_new_elapsed_seconds=plan['original_elapsed_seconds'] + final['elapsed_since_registration'],
        new_optimizer_updates=plan['new_optimizer_updates'],
        unique_optimizer_updates_by_condition=dict(TF_CONT=500, TF_ON=500, MLP_ON=500),
        validation_CPU_threads=1, training_CPU_threads=2, old_CPU2_parity_inherited=False,
        production_GPU_CAL_unchanged=True)


def executable_functions():
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU-only continuation must hide CUDA')
    from tf_effect_train_v1 import train as previous
    module, review = transform()
    namespace = dict(previous.__dict__)
    namespace.update(__reuse_v2_register=register, __reuse_v2_plan=augment_plan,
        __reuse_v2_prepared=prepared, __reuse_v2_draws=draws,
        __reuse_v2_restore=restore, __reuse_v2_record=record, __reuse_v2_final=final)
    original_math = previous.frozen_functions
    original_reference = previous.reference_cpu

    def frozen_math():
        funcs = original_math()
        predict = funcs['predict_all']

        def one_thread(*args, **kwargs):
            with validation_threads():
                return predict(*args, **kwargs)

        funcs['predict_all'] = one_thread
        return funcs

    def reference(*args, **kwargs):
        with validation_threads():
            return original_reference(*args, **kwargs)

    namespace['frozen_functions'] = frozen_math
    namespace['reference_cpu'] = reference
    exec(compile(module, str(OWN / 'train.py') + ':reviewed-original-AST', 'exec'), namespace)
    return namespace['prepare'], namespace['run'], review


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare', action='store_true'); choice.add_argument('--run', action='store_true')
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--prior-job', type=Path)
    parser.add_argument('--master-deadline-epoch', type=float)
    parser.add_argument('--onpolicy', type=Path); parser.add_argument('--data-audit', type=Path)
    parser.add_argument('--trigger', type=Path); parser.add_argument('--include-mlp-on', action='store_true')
    parser.add_argument('--wall-cap-seconds', type=int, default=1800)
    parser.add_argument('--storage-cap-mib', type=int, default=256)
    args = parser.parse_args()
    if args.prepare:
        if any(value is None for value in (args.prior_job, args.master_deadline_epoch, args.onpolicy, args.data_audit, args.trigger)):
            parser.error('--prepare requires --prior-job --master-deadline-epoch --onpolicy --data-audit --trigger')
        closed_predecessor(args.prior_job)
    else:
        if any(value is not None for value in (args.prior_job, args.master_deadline_epoch, args.onpolicy, args.data_audit, args.trigger)):
            parser.error('--run uses only frozen PLAN and registration')
    prepare, run, review = executable_functions()
    prepare(args) if args.prepare else run(args.job)


if __name__ == '__main__':
    main()
