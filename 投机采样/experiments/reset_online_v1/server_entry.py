"""Instrument original reset in a persistent, one-client, complete-task server."""
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback
import types

from protocol import EXP, load, write


def fingerprint(value):
    import torch
    if isinstance(value, torch.Tensor):
        x = value.detach().contiguous().cpu()
        return dict(shape=list(x.shape), dtype=str(x.dtype),
                    bytes_sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [fingerprint(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError('unsupported reset-state field ' + str(type(value)))


def snapshot(s, *, empty):
    import torch
    layers = []
    for block in s.transformer.blocks:
        cache = block.attn1.attn_caches[s.cache_name]
        mask = cache['mask']
        if empty and (bool(mask.any()) or bool(cache['is_pred'].any())
                      or not bool((cache['id'] == -1).all())):
            raise ValueError('original reset did not empty every cache layer')
        # Unwritten torch.empty K/V storage has no semantics. Only valid slots
        # are compared; full mask/id/is_pred metadata is always compared.
        layers.append(fingerprint(dict(mask=mask, id=cache['id'], is_pred=cache['is_pred'],
                                       k=cache['k'][:, mask], v=cache['v'][:, mask])))
    if empty and (s.frame_st_id != 0 or s.init_latent is not None
                  or any(v is not None for v in s.streaming_vae.feat_cache)):
        raise ValueError('original reset left streaming/frame state')
    return dict(frame_st_id=int(s.frame_st_id), use_cfg=bool(s.use_cfg),
                init_latent=fingerprint(s.init_latent),
                prompt=fingerprint(s.prompt_embeds), negative=fingerprint(s.negative_prompt_embeds),
                action_mask=fingerprint(s.action_mask), q01=fingerprint(s.actions_q01),
                q99=fingerprint(s.actions_q99), action_norm_method=s.action_norm_method,
                KV=layers, VAE=fingerprint(s.streaming_vae.feat_cache),
                cpu_rng=fingerprint(torch.get_rng_state()),
                cuda_rng=fingerprint(torch.cuda.get_rng_state(s.device)))


def main():
    job = Path(os.environ['RESET_ONLINE_JOB']).resolve()
    p = load(job)
    arm = os.environ['RESET_ONLINE_ARM']
    if arm not in p['arms'] or os.getenv('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('one assigned local model required')
    sys.path.insert(0, str(EXP))
    sys.path.insert(0, str(EXP / 'formal_e2e_speed_v1'))
    from formal_e2e_speed_v1.server_entry import _paths, validate_server, RamAccounting
    _paths()
    from wan_va import wan_va_server as source
    from measured_online import control
    from selection import install as install_selector
    import buffers
    from reset_text_bank_v1.bank import TextBank, install

    original_run = source.run_async_server_mode

    def serve(s, local_rank, host, port):
        validate_server(s)
        if local_rank != 0:
            raise ValueError('single rank')
        s.action_per_frame = int(s.job_config.action_per_frame)
        proxy = online = None
        if arm == 'A_OFF':
            s._load_p0_action_draft_b2_predictor().model.to(s.device).eval()
            proxy = install_selector(s, 'prefix_then_cost')
            buffers.prepare(s._infer_action_verifier_k2_b2, s._load_p0_shadow_api())
            online = control(s, 'O3', allow_cuda=True)
        accounting = RamAccounting(s, cap=20000)
        bank = TextBank(p['bank'])
        original_encode = s.encode_prompt
        restore_bank = install(s, bank)
        cached_encode = s.encode_prompt
        restore_bank()
        original_reset, original_infer = s._reset, s.infer
        initial_cpu = source.torch.get_rng_state().clone()
        initial_cuda = source.torch.cuda.get_rng_state(s.device).clone()
        active = dict(key=None, row=None, phases={}, helper_calls=[], request_start=0,
                      tree_start=0, forward_start=0)
        gate = threading.RLock()

        def phase(obj, name, label):
            original = getattr(obj, name)
            def wrapped(*args, **kwargs):
                start = time.monotonic()
                result = original(*args, **kwargs)
                active['phases'][label] = active['phases'].get(label, 0.) + time.monotonic() - start
                return result
            setattr(obj, name, wrapped)

        phase(s.transformer, 'clear_cache', 'transformer_clear')
        phase(s.transformer, 'create_empty_cache', 'transformer_create')
        phase(s.streaming_vae, 'clear_cache', 'streaming_vae_clear')
        get_text = s._get_t5_prompt_embeds
        def timed_text(_s, *args, **kwargs):
            start = time.monotonic()
            result = get_text(*args, **kwargs)
            active['helper_calls'].append(dict(prompt=kwargs.get('prompt'),
                                               wall_sec=time.monotonic() - start))
            return result
        s._get_t5_prompt_embeds = types.MethodType(timed_text, s)

        def reset(_s, prompt=None):
            row = active['row']
            selected = cached_encode if row['text_bank'] else original_encode
            def timed_encode(*args, **kwargs):
                start = time.monotonic()
                result = selected(*args, **kwargs)
                active['phases']['text_encode'] = time.monotonic() - start
                return result
            s.encode_prompt = timed_encode
            source.torch.cuda.synchronize(s.device)
            before_cpu, before_cuda = source.torch.get_rng_state(), source.torch.cuda.get_rng_state(s.device)
            start = time.monotonic()
            result = original_reset(prompt=prompt)
            source.torch.cuda.synchronize(s.device)
            duration = time.monotonic() - start
            if not source.torch.equal(before_cpu, source.torch.get_rng_state()) or not source.torch.equal(before_cuda, source.torch.cuda.get_rng_state(s.device)):
                raise ValueError('reset changed model RNG')
            for text, value in ((prompt, s.prompt_embeds), ('', s.negative_prompt_embeds)):
                expected = bank.get(text, device=s.device, dtype=s.dtype, sequence_length=512)
                if not source.torch.equal(value, expected):
                    raise ValueError('online original/cached text differs from complete bank')
            out = job / 'episodes' / row['key']
            write(out / 'RESET.json', dict(status='PASS_RESET_TEXT_AND_EMPTY_STATE',
                row=row, reset_wall_sec=duration, phases=active['phases'],
                text_helper_calls=active['helper_calls'], RNG_unchanged=True,
                state=snapshot(s, empty=True), subphase_semantics='Host wall; reset total synchronizes CUDA; snapshots outside reset timer',
                performance_claim=False))
            return result
        s._reset = types.MethodType(reset, s)

        def infer(_s, request):
            with gate:
                context = request.get('p0_trace_context') or {}
                key = context.get('parallel_worker_id')
                if request.get('reset'):
                    rows = [r for r in p['cases'] if r['key'] == key and r['arm'] == arm]
                    if len(rows) != 1 or active['key'] is not None:
                        raise ValueError('one assigned reset after completed episode required')
                    row = rows[0]
                    if context.get('task_index') != row['task'] or context.get('episode_index') != row['init']:
                        raise ValueError('actual task/init differs')
                    source.torch.set_rng_state(initial_cpu)
                    source.torch.cuda.set_rng_state(initial_cuda, s.device)
                    active.update(key=key, row=row, phases={}, helper_calls=[],
                                  request_start=len(accounting.requests), forward_start=accounting.attempted,
                                  tree_start=0 if proxy is None else len(proxy.records))
                    s.save_root = str(job / 'episodes' / key / 'server_output')
                elif key != active['key']:
                    raise ValueError('request does not belong to active complete episode')
                return original_infer(request)
        s.infer = types.MethodType(infer, s)

        def finish_monitor():
            try:
                while True:
                    time.sleep(.2)
                    with gate:
                        key = active['key']
                        if key is None:
                            continue
                        out = job / 'episodes' / key
                        if not (out / 'FINISH_REQUESTED').exists():
                            continue
                        source.save_async.__globals__['executor'].submit(lambda: None).result(timeout=90)
                        source.torch.cuda.synchronize(s.device)
                        if accounting.failed or accounting.active or (proxy is not None and proxy.pending is not None):
                            raise ValueError('unfinished model accounting/selection')
                        write(out / 'FINAL-SERVER-STATE.json', snapshot(s, empty=False))
                        write(out / 'COUNTER.json', dict(arm=arm, key=key,
                            target_forwards=accounting.attempted - active['forward_start'],
                            requests=accounting.requests[active['request_start']:],
                            acceptance=[] if proxy is None else proxy.records[active['tree_start']:],
                            bank_hits=bank.hits, persistent_model=True))
                        active['key'] = None
                        write(out / 'FINISH_READY.json', dict(status='PASS', async_drained=True))
            except BaseException as exc:
                write(job / f'{arm}-MONITOR-FAILED.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        threading.Thread(target=finish_monitor, daemon=True).start()
        write(job / f'{arm}-READY.json', dict(status='READY', arm=arm, pid=os.getpid(),
              all_original_and_bank_embeddings_checked_at_actual_reset=True,
              model_persistent=True, performance_claim=False, port=port))
        return original_run(s, local_rank, host, port)

    source.run_async_server_mode = serve
    source.init_logger()
    source.main()


if __name__ == '__main__':
    main()
