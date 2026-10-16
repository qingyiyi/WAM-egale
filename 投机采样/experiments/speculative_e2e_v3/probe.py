"""Actual complete chunk D/S outputs, all30 cache and RNG comparison; no EGL."""
import copy
import importlib.util
import os
from pathlib import Path
import sys
import traceback

from task_contract import OWN, WORKTREE, worker_entry, write, digest

def snapshot_module():
    path = OWN.parent/'fixed_load_cost/snapshot.py'
    spec = importlib.util.spec_from_file_location('e2e_original_snapshot',path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def run(s, source, job, out, p):
    import torch
    from online_shared import OnlineShared
    from runtime import Runtime
    from server_entry import validate_server
    validate_server(s)
    snap = snapshot_module()
    payload = torch.load(p['snapshot_reference']['path'],map_location='cpu',weights_only=False)
    if (snap.fingerprint(payload['state']) != p['snapshot_reference']['state_hash']
            or snap.fingerprint(payload['request']) != p['snapshot_reference']['request_hash']):
        raise ValueError('real fixed fixture content hash changed')
    if any(c['slots'].numel() != 144 for c in payload['state']['caches']):
        raise ValueError('original short fixture history mismatch')
    s.infer(dict(reset=True,prompt=None))
    predictor = s._load_p0_action_draft_b2_predictor()
    predictor.model.to(s.device).eval()
    if predictor.model.output_dim != 112:
        raise ValueError('frozen A_off geometry')
    s._load_p0_shadow_api()
    rt = Runtime(s,out,p['probe_forward_cap'],allow_cuda=True)
    saved = {}
    def save(value, path):
        name = Path(path).name
        if name.startswith('latents_'): saved['video'] = value
        elif name.startswith('actions_'): saved['normalized_action'] = value
        elif name.startswith('obs_data_'): saved['observation'] = value
        else: raise ValueError('unexpected original async artifact')
    source.save_async = save
    originals = [b.attn1.forward for b in s.transformer.blocks]
    results = {}
    for arm in ('D','S'):
        snap.restore(s,payload['state'])
        if snap.fingerprint(snap.capture(s)) != payload['state_hash']:
            raise ValueError('all30 original cache/RNG starting restore mismatch')
        saved.clear()
        control = OnlineShared(s,enabled=arm=='S',allow_cuda=True).install()
        try:
            count = rt.attempted
            returned = s.infer(copy.deepcopy(payload['request']))
            torch.cuda.synchronize()
            if set(saved) != {'video','normalized_action'}:
                raise ValueError('complete original action/video outputs missing')
            if saved['normalized_action'].shape[1] != 30:
                raise ValueError('full30-channel normalized action required')
            data = dict(saved=snap.cpu(saved),returned=snap.cpu(returned),
                        final_state=snap.capture(s))
            detail = dict(output_hash=snap.fingerprint(data['saved']),
                          returned_hash=snap.fingerprint(data['returned']),
                          final_cache_scheduler_VAE_RNG_hash=snap.fingerprint(data['final_state']),
                          forward_intents=rt.attempted-count,shared=control.summary())
            torch.save(data,out/(arm+'-complete-chunk.pt'))
            results[arm] = detail
        finally:
            control.close()
        if any(b.attn1.forward != old for b,old in zip(s.transformer.blocks,originals)):
            raise ValueError('per-chunk attention cleanup failed')
    for key in ('output_hash','returned_hash','final_cache_scheduler_VAE_RNG_hash'):
        if results['D'][key] != results['S'][key]:
            raise ValueError('actual full-chunk D/S bitwise mismatch: '+key)
    if results['D']['forward_intents'] != results['S']['forward_intents']:
        raise ValueError('D/S scheduler/route forward count differs')
    if not results['S']['shared']['candidates'] or not results['S']['shared']['last_request_clean']:
        raise ValueError('actual shared path not used/clean')
    rt.close()
    write(out/'PROBE-RESULT.json',dict(status='PASS_FULL_CHUNK_BITWISE',results=results,
          all30_original_cache_scheduler_VAE_RNG_bitwise=True,
          complete30_channel_normalized_action_and_video_bitwise=True,
          returned_physical_actions_bitwise=True,production_source_modified=False,
          target_intents=rt.attempted,EGL=0,full_episodes=0),exclusive=True,durable=True)

def main():
    job,out,p = worker_entry()
    sys.path.insert(0,str(WORKTREE))
    from wan_va import wan_va_server as source
    def execute(s,rank,host,port):
        try: run(s,source,job,out,p)
        except BaseException as exc:
            write(out/'PROBE-FAILED.json',dict(error=repr(exc),traceback=traceback.format_exc()),
                  exclusive=True,durable=True)
            raise
    source.run_async_server_mode = execute
    source.init_logger()
    source.main()

if __name__ == '__main__':
    main()
