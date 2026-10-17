"""Independent O3 source admission using copied raw source, never a digest.

The legacy admission has one explicit source-SHA guard. Its AST is reused
with exactly that statement replaced by a raw source byte comparison. All
class/code/layout/device/ownership guards and numerical functions remain.
The frozen legacy modules/classes/globals are not edited or monkeypatched.
"""
from __future__ import annotations
import ast
import inspect
from pathlib import Path
import textwrap


def install_o3(server, *, original, source_reference):
    template = original.TemplateSharedPrefix
    old = template.admit
    tree = ast.parse(textwrap.dedent(inspect.getsource(old)))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != 'admit':
        raise ValueError('one unchanged legacy source admission function required')
    function = functions[0]
    matches = [n for n in function.body if isinstance(n, ast.If) and
        any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='digest'
            for c in ast.walk(n.test))]
    expected = ast.parse('digest(MODEL_SOURCE) != self.expected_source_sha256',mode='eval').body
    if len(matches)!=1 or ast.dump(matches[0].test)!=ast.dump(expected):
        raise ValueError('legacy source digest guard shape changed; do not guess')
    reference_path=Path(source_reference).resolve(strict=True)
    reference=reference_path.read_bytes()
    if not 1024 <= len(reference) <= 2*2**20:
        raise ValueError('bounded original target source reference required')
    original_path=Path(old.__globals__['MODEL_SOURCE'])

    def require_raw_source():
        if original_path.read_bytes()!=reference:
            raise ValueError('actual target source raw bytes differ from copied fixed reference')

    replacement=ast.Expr(value=ast.Call(func=ast.Name(id='_require_raw_target_source',ctx=ast.Load()),args=[],keywords=[]))
    function.body[function.body.index(matches[0])]=ast.copy_location(replacement,matches[0])
    if any(isinstance(c,ast.Call) and isinstance(c.func,ast.Name) and c.func.id=='digest' for c in ast.walk(function)):
        raise ValueError('remaining digest call in independent admission')
    namespace={**old.__globals__,'_require_raw_target_source':require_raw_source,
        '__name__':'tf_fullsuite_eval_v1._raw_source_admission'}
    module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(module,str(reference_path)+':independent-admit','exec'),namespace)
    admission=namespace['admit']

    class RawSourceMeasuredO3(original.MeasuredO3):
        admit=admission

    class RawSourceOnlineO3(original.OnlineO3):
        _base_install=original.clone_function(original.OnlineO3._base_install,SharedPrefix=RawSourceMeasuredO3)

    control=original.clone_function(original.control,OnlineO3=RawSourceOnlineO3)
    result=control(server,'O3',allow_cuda=True)
    result.no_digest_admission_provenance=dict(
        legacy_source_guard='replaced by independent raw source byte comparison',
        reference_path=str(reference_path),actual_source_path=str(original_path),
        all_other_legacy_admission_AST_statements_retained=True,
        legacy_sources_and_class_globals_unchanged=True,
        source_SHA_executed=False,remote_reference_cryptographic_identity='UNKNOWN_NOT_HASHED')
    return result
