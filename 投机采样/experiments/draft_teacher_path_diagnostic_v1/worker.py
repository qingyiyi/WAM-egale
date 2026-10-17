"""Two complete recorded chunks and one frozen real-query causal comparison.

No simulator, training, production integration, formal performance or smoke.
"""
import argparse
import copy
import os
from pathlib import Path
import signal
import sys
import traceback
import types

OWN = Path(__file__).resolve().parent
sys.path.insert(0, str(OWN.parent))
from draft_teacher_path_diagnostic_v1.protocol import ROOT, load, read, write, require


def run(s, source, helpers, job, p):
    import numpy as np
    import torch
    from fixed_load_cost import snapshot as snap
    from reset_text_bank_v1.bank import TextBank, install as install_bank
    from task4_direction_completion_v1.observer import Observer, load_with_existing_cal
    from template import TemplateSharedPrefix

    validate_server, RamAccounting, modules = helpers
    validate_server(s)
    require(s._p0_action_draft_mode == 'off', 'initial ordinary target path required')
    require(Path(source.__file__).resolve() == ROOT / '投机采样/wam-p0-worktree/wan_va/wan_va_server.py',
            'original pinned model source required')
    initial = torch.load(p['arms']['T']['rng'], map_location='cpu', weights_only=True)
    require(torch.equal(torch.get_rng_state(), initial['CPU']) and
            torch.equal(torch.cuda.get_rng_state(s.device), initial['CUDA']), 'actual target initialization RNG differs')
    s.action_per_frame = int(s.job_config.action_per_frame)
    install_bank(s, TextBank(p['reference_plan']['bank']))
    def no_text(*a, **kw):
        raise RuntimeError('validated complete text bank required')
    s._get_t5_prompt_embeds = types.MethodType(no_text, s)
    accounting = RamAccounting(s, cap=p['target_forward_cap'])
    current = dict(arm=None, action_index=0, query=None, batch=2, pending_capture=None)
    captures, stages = {}, {}
    handles = []

    def cpu(value):
        return snap.cpu(value)

    def numbers(a, b):
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape or a.dtype != b.dtype:
                return dict(bitwise=False, shape_A=list(a.shape), shape_B=list(b.shape),
                    dtype_A=str(a.dtype), dtype_B=str(b.dtype), incomparable=True)
            aa = torch.view_as_real(a) if a.is_complex() else a
            bb = torch.view_as_real(b) if b.is_complex() else b
            d = (aa.double() - bb.double()).abs()
            return dict(bitwise=torch.equal(aa, bb), different=int(torch.count_nonzero(d)),
                max_abs=float(d.max()) if d.numel() else 0., elements=d.numel(),
                finite=bool(torch.isfinite(aa).all() and torch.isfinite(bb).all()))
        if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
            children = [numbers(x, y) for x, y in zip(a, b)]
            return dict(bitwise=len(a) == len(b) and all(x['bitwise'] for x in children), children=children)
        if isinstance(a, dict) and isinstance(b, dict):
            children = {k: numbers(a[k], b[k]) for k in a.keys() & b.keys()}
            return dict(bitwise=a.keys() == b.keys() and all(x['bitwise'] for x in children.values()), children=children)
        return dict(bitwise=a == b)

    def select(value):
        if isinstance(value, torch.Tensor):
            b = current['batch']
            ids = [0, b//2]
            if value.ndim and value.shape[0] == b:
                value = value[ids]
            elif value.ndim and value.shape[0] == b*16:
                value = value.reshape(b, 16, *value.shape[1:])[ids]
            elif value.ndim > 1 and value.shape[:2] == (1, b*16):
                value = value.reshape(b, 16, *value.shape[2:])[ids]
            return cpu(value)
        if isinstance(value, (list, tuple)):
            return type(value)(select(v) for v in value)
        if isinstance(value, dict):
            return {k: select(v) for k, v in value.items()}
        return value

    def record(name, value):
        if current['query'] is not None:
            stages.setdefault(current['query'], []).append((name, select(value)))

    for name, module in s.transformer.named_modules():
        selected = (name in ('action_embedder', 'condition_embedder.text_embedder',
                    'condition_embedder_action', 'rope', 'norm_out', 'action_proj_out') or
                    (name.startswith('blocks.0.') and (not list(module.children()) or
                     name in ('blocks.0.attn1', 'blocks.0.attn2', 'blocks.0.ffn'))) or
                    (name.startswith('blocks.') and name.count('.') == 1))
        if selected:
            handles.append(module.register_forward_hook(lambda m, a, y, name=name: record(name, y)))
    handles.append(s.transformer.blocks[0].register_forward_pre_hook(
        lambda m, a, kw: record('block0.full_inputs', dict(args=a,
            kwargs={k: v for k, v in kw.items() if k != 'candidate_cache_rows'})), with_kwargs=True))

    def cache_snapshot():
        return [snap.capture_cache(b.attn1.attn_caches[s.cache_name]) for b in s.transformer.blocks]

    def restore_caches(caches):
        for block, saved in zip(s.transformer.blocks, caches):
            snap.restore_cache(block.attn1.attn_caches[s.cache_name], saved)

    def observe_pre(_model, args, kw):
        inp = args[0] if args else kw['input_dict']
        probe = False
        if current['arm'] == 'T' and kw.get('action_mode') and kw.get('update_cache', 0) == 0:
            index = current['action_index']
            current['action_index'] += 1
            probe = index == 1
        elif current['arm'] == 'TF_CONT' and kw.get('candidate_cache_rows') is not None:
            probe = 'TF_CONT' not in captures
        if not probe:
            return
        name = current['arm']
        captures[name] = dict(input=cpu(inp), kwargs=cpu(kw), caches=cache_snapshot(),
            CPU_rng=cpu(torch.get_rng_state()), CUDA_rng=cpu(torch.cuda.get_rng_state(s.device)))
        current.update(query=name, batch=inp['noisy_latents'].shape[0], pending_capture=name)

    def observe_post(_model, args, kw, output):
        name = current['pending_capture']
        if name is not None:
            captures[name]['output'] = cpu(output)
            current.update(query=None, pending_capture=None)
    handles.append(s.transformer.register_forward_pre_hook(observe_pre, with_kwargs=True))
    handles.append(s.transformer.register_forward_hook(observe_post, with_kwargs=True, always_call=True))

    chunks = []
    proxy = online = None
    methods = (s.infer, s.postprocess_action, s.action_scheduler.step)
    try:
        for arm in ('T', 'TF_CONT'):
            row = p['arms'][arm]
            expected = torch.load(row['trace'], map_location='cpu', weights_only=False)
            current.update(arm=arm, action_index=0)
            if arm == 'TF_CONT':
                cp, parity = load_with_existing_cal(p['reference_plan'], arm, s.device, tuple(range(7)), job/'CAL-REUSE.json')
                s._p0_action_draft_b2_predictor = cp
                s._p0_action_draft_b2_checkpoint = cp.checkpoint_path
                s._p0_action_draft_mode = p['reference_plan']['mode']
                proxy = modules['selection'].install(s, 'prefix_then_cost')
            observer = Observer(s, job, arm)
            if arm != 'T':
                observer.install_action()
                modules['buffers'].prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
                online = modules['measured_online'].control(s, 'O3', allow_cuda=True)
            observer.install_io(accounting)
            torch.set_rng_state(initial['CPU'])
            torch.cuda.set_rng_state(initial['CUDA'], s.device)
            s.save_root = str(job / 'recorded-chunks' / arm)
            context = dict(expected['context'])
            reset_context = {k: v for k, v in context.items() if k not in
                ('rollout_chunk', 'env_steps_before', 'simulator_timestep_before')}
            s.infer(dict(reset=True, prompt=context['prompt'], p0_trace_context=reset_context))
            with np.load(row['observations'], allow_pickle=False) as z:
                obs = {k: z[k] for k in z.files}
            result = s.infer(dict(obs=obs, prompt=context['prompt'], p0_trace_context=context))
            source.save_async.__globals__['executor'].submit(lambda: None).result(timeout=60)
            folder = Path(s.exp_save_root)
            for prefix in ('actions', 'latents'):
                actual = torch.load(folder / (prefix + '_0.pt'), map_location='cpu', weights_only=True)
                original = torch.load(row[prefix], map_location='cpu', weights_only=True)
                require(torch.equal(actual, original), arm + ' original complete chunk ' + prefix + ' differs')
            recorded = torch.load(job/'episodes'/context['parallel_worker_id']/'direction_trace/chunk-000.pt',
                map_location='cpu', weights_only=False)
            require(len(recorded['commits']) == len(expected['commits']) == 50, 'all50 scheduler commits required')
            for x, y in zip(recorded['commits'], expected['commits']):
                require(x['step'] == y['step'] and x['source'] == y['source'] and x['bonus'] == y['bonus'] and
                    torch.equal(x['state'], y['state']) and torch.equal(x['direction'], y['direction']),
                    arm + ' committed path/direction reproduction differs')
            require(np.array_equal(result['action'], expected['submitted_server_action']), 'decoded command differs')
            chunks.append(dict(arm=arm, full_policy_chunk=True, all50_commits_exact=True,
                actions_video_and_decoded_exact=True, not_closed_loop_episode=True))
            if online:
                online.close()
                online = None
            s.infer, s.postprocess_action, s.action_scheduler.step = methods
        require(set(captures) == {'T', 'TF_CONT'}, 'both actual first-query captures required')
        current.update(arm=None, query=None)
        write(job/'TWO-CHUNK-REPRODUCTION.json', dict(status='PASS_COMPLETE_RECORDED_POLICY_CHUNKS', chunks=chunks))
        a, b = captures['T'], captures['TF_CONT']
        root_input = {k: v[[0, 7]] for k, v in b['input'].items()}
        inputs = numbers(a['input'], root_input)
        caches_equal = all(snap.fingerprint(x) == snap.fingerprint(y) for x, y in zip(a['caches'], b['caches']))
        require(inputs['bitwise'] and caches_equal, 'full actual inputs/valid KV differ before first teacher discrepancy')
        require(torch.equal(a['CPU_rng'], b['CPU_rng']) and torch.equal(a['CUDA_rng'], b['CUDA_rng']),
                'actual query RNG differs')
        torch.save(dict(schema='one-real-task4-root-query/v1', input_B2=a['input'], input_B14=b['input'],
            kwargs_B2=a['kwargs'], kwargs_B14=b['kwargs'], shared_valid_history=a['caches'],
            CPU_rng=a['CPU_rng'], CUDA_rng=a['CUDA_rng'], no_cross_candidate_attention=True), job/'REAL-QUERY.pt')

        def dev(value):
            if isinstance(value, torch.Tensor):
                return value.to(s.device)
            if isinstance(value, dict):
                return {k: dev(v) for k, v in value.items()}
            return value

        comparisons = {}
        outputs = {'T': a['output'], 'TF_CONT': b['output'][[0, 7]]}
        configs = [('B2_repeat', a['input'], None, False),
                   ('B2_readonly', root_input, torch.tensor([0, 1], dtype=torch.long), False),
                   ('B14_dense', b['input'], b['kwargs']['candidate_cache_rows'], False),
                   ('B14_shared_repeat', b['input'], b['kwargs']['candidate_cache_rows'], True)]
        for name, inp, rows, shared in configs:
            restore_caches(a['caches'])
            torch.set_rng_state(a['CPU_rng'])
            torch.cuda.set_rng_state(a['CUDA_rng'], s.device)
            current.update(query=name, batch=inp['noisy_latents'].shape[0])
            ctl = None
            try:
                kw = dict(update_cache=0, cache_name=s.cache_name, action_mode=True)
                if shared:
                    ctl = TemplateSharedPrefix(s.transformer, enabled=True, allow_cuda=True).install()
                    with ctl.lease(request='draft-quality-root-replay', arm='S', cache_name=s.cache_name, device=s.device) as token:
                        y = s.transformer(dev(inp), **kw, candidate_cache_rows=token.rows)
                else:
                    if rows is not None:
                        kw['candidate_cache_rows'] = rows.to(s.device)
                    y = s.transformer(dev(inp), **kw)
                outputs[name] = select(y)
                after = cache_snapshot()
                # Ordinary temporary staging may write invalid id/is_pred slots;
                # all valid history bytes and masks must remain unchanged.
                for old, new in zip(a['caches'], after):
                    require(torch.equal(old['mask'], new['mask']) and torch.equal(old['k_valid'], new['k_valid']) and
                        torch.equal(old['v_valid'], new['v_valid']), 'replay mutated valid history')
                require(torch.equal(torch.get_rng_state(), a['CPU_rng']) and
                    torch.equal(torch.cuda.get_rng_state(s.device), a['CUDA_rng']), 'query replay consumed RNG')
                comparisons[name] = numbers(a['output'], outputs[name])
            finally:
                current['query'] = None
                if ctl:
                    ctl.close()
        require(comparisons['B2_repeat']['bitwise'], 'same ordinary query is not bitwise reproducible')
        require(numbers(outputs['TF_CONT'], outputs['B14_shared_repeat'])['bitwise'],
                'production shared verifier repeat differs')
        stage_results = {}
        baseline = stages['T']
        for name, items in stages.items():
            if name == 'T':
                continue
            rows = []
            require([x[0] for x in items] == [x[0] for x in baseline], 'stage topology differs: ' + name)
            for (label, x), (_, y) in zip(baseline, items):
                rows.append(dict(stage=label, **numbers(x, y)))
            stage_results[name] = dict(first_difference=next((x['stage'] for x in rows if not x['bitwise']), None), stages=rows)
        torch.save(dict(outputs=outputs, stages=stages), job/'REAL-STAGES.pt')
        closed = accounting.close()
        write(job/'WORKER-RESULT.json', dict(status='COMPLETE_SAME_REAL_QUERY_TEACHER_PATH_DIAGNOSIS',
            chunks=chunks, all_actual_input_fields_exact=True, all30_actual_valid_KV_exact=True,
            actual_query_RNG_exact=True, comparisons_to_ordinary_B2=comparisons,
            comparisons_to_actual_shared_B14={k: numbers(outputs['TF_CONT'], v) for k, v in outputs.items()},
            stage_results=stage_results, target_accounting=closed, query_scheduler_index=1,
            root_cause_of_episode_failure='UNKNOWN', training_trigger=False, permissions=p['permissions'],
            no_environment_episode=True, no_training=True, performance_claim=False))
    finally:
        for handle in handles:
            handle.remove()
        if online:
            online.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--execute-reviewed', action='store_true', required=True)
    args = parser.parse_args()
    p = load(args.job)
    require(os.getenv('CUDA_VISIBLE_DEVICES') == '0' and (args.job/'RUN-STARTED.json').exists(), 'bounded supervisor required')
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    def stop(*a):
        raise TimeoutError('numeric diagnosis active timeout')
    signal.signal(signal.SIGALRM, stop)
    signal.alarm(1000)
    try:
        from tf_usable_quality_v1.legacy import server_helpers
        helpers = server_helpers()
        from wan_va import wan_va_server as source
        def serve(s, rank, host, port):
            require(rank == 0, 'one rank required')
            with torch.no_grad():
                run(s, source, helpers, args.job, p)
        source.run_async_server_mode = serve
        sys.argv = [str(__file__), '--config-name', 'libero', '--port', '19658', '--save_root', str(args.job/'unused')]
        source.init_logger()
        source.main()
        source.save_async.__globals__['executor'].shutdown(wait=True)
    except BaseException as exc:
        if not (args.job/'WORKER-FAILED.json').exists():
            write(args.job/'WORKER-FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc(), training_trigger=False))
        raise
    finally:
        signal.alarm(0)
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
