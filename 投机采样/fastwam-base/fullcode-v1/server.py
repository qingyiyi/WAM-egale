import argparse
import json
import os
import pickle
import signal
import socket
import socketserver
import struct
import threading
import time
import traceback
from pathlib import Path
import numpy as np
import torch
from runtime import ROOT, build, inputs, postprocess, write
from real_speculative import CanonicalCollector, RealSpeculative
from draft import load as load_draft
from policy_schedule import policy_role

P = argparse.ArgumentParser()
P.add_argument('--port', type=int, required=True)
P.add_argument('--worker', required=True)
P.add_argument('--assets', required=True)
args = P.parse_args()
OUT = ROOT / 'results' / args.worker
OUT.mkdir(parents=True, exist_ok=False)
reg = json.loads((ROOT / 'REGISTRATION.json').read_text())
start = time.time()
deadline = min(reg['deadline_epoch'] - 300, start + 8*3600)
status = dict(phase='LOADING', start=start, deadline=deadline, pid=os.getpid(),
              birth_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]), calls=0)
write(OUT / 'STATUS.json', status)
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('bounded server lifecycle')))
signal.alarm(max(1, int(deadline-time.time())))
torch.set_num_threads(2)
torch.manual_seed(42)
np.random.seed(42)
lock = threading.Lock()
model, processor, cfg, ns = build(args.assets, OUT)
server_identity = dict(host=socket.gethostname(), worker=args.worker,
                       gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                       visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES', 'all'),
                       cuda=torch.version.cuda, model_load=str(OUT/'MODEL-LOAD.json'))
collector = CanonicalCollector(model)
spec = None
first = True
failure_latch = None


def handle(request):
    global spec, first
    if time.time() > deadline: raise TimeoutError('server deadline')
    if failure_latch is not None:
        raise RuntimeError('server stopped accepting predictions after failure: '+failure_latch)
    if request['op'] == 'ping': return dict(ready=True, calls=status['calls'])
    if request['op'] != 'predict': raise ValueError('unknown operation')
    requested_arm, identity = request['arm'], request['identity']
    arm = policy_role(requested_arm, int(identity['chunk']))
    episode = ROOT / 'results/episodes' / identity['episode']
    episode.mkdir(parents=True, exist_ok=True)
    n = int(identity['chunk'])
    base = episode / f'chunk-{n:03d}'
    if base.with_suffix('.json').exists(): raise ValueError('immutable chunk already exists')
    # Save exact raw observations outside timing, allowing condition reconstruction.
    np.savez_compressed(base.with_suffix('.npz'), **request['obs'])
    write(episode / 'IDENTITY.json', dict(identity, task_description=request['task'], arm=requested_arm))
    kwargs = inputs(request['obs'], request['task'], model, processor, cfg, ns)
    if first:
        t = time.perf_counter()
        with torch.no_grad():
            ref = model.infer_action(**kwargs, compile_action_infer=False)['action'].float()
            model.empirical_replay = collector
            adapted = model.infer_action_replay(**kwargs, compile_action_infer=False)['action'].float()
            assert torch.equal(ref, adapted), 'canonical collector changed scheduler result'
            compiled = model.infer_action(**kwargs, compile_action_infer=True)['action'].float()
        diff = compiled-ref
        rel = float(diff.norm()/ref.norm().clamp_min(1e-6))
        maxerr, lim = float(diff.abs().max()), .05+.02*float(ref.abs().max())
        assert rel <= .01 and maxerr <= lim, (rel, maxerr, lim)
        write(OUT / 'CANONICAL-PARITY.json', dict(collector_eager_exact=True, compiled_relative_L2=rel,
              max_abs=maxerr, max_abs_limit=lim, relative_L2_limit=.01, cold_seconds=time.perf_counter()-t,
              cold_validation_policy_calls=3, cold_validation_target_calls=30,
              cold_validation_target_state_rows=30, input_observation=str(base.with_suffix('.npz')),
              scheduler=dict(updates=10, sigma_shift=5.0, operational_dtype='torch.bfloat16',
                             timestep=collector.columns['timestep'].tolist(),
                             delta_sigma=collector.columns['delta'].tolist(),
                             source='actual canonical GPU scheduler passed to collector'),
              server_identity=server_identity))
        first = False
    collect = arm == 'T' and identity['split'] in ('FIT', 'CAL')
    if arm == 'S' and spec is None:
        selected = json.loads((ROOT / 'training/SELECTED.json').read_text())
        spec = RealSpeculative(model, load_draft(ROOT / 'training/checkpoint-3000.pt'), selected['tau'])
        model.empirical_replay = spec
        spec.validate = True
        audit_begin = time.perf_counter()
        with torch.no_grad(): model.infer_action_replay(**kwargs, compile_action_infer=True)
        write(OUT / 'B7-AUDIT.json', spec.audit_records)
        torch.save(spec.audit_inputs, OUT/'B7-AUDIT-INPUT.pt')
        write(OUT / 'SPEC-COLD-VALIDATION.json', dict(seconds=time.perf_counter()-audit_begin,
              scheduler_policy_counts=dict(spec.counts), extra_audit_target_calls=8,
              extra_audit_target_state_rows=14, input_observation=str(base.with_suffix('.npz')),
              fixture=str(OUT/'B7-AUDIT-INPUT.pt'), server_identity=server_identity))
        spec.validate = False
    model.empirical_replay = spec if arm == 'S' else collector
    torch.cuda.synchronize()
    t = time.perf_counter()
    kwargs = inputs(request['obs'], request['task'], model, processor, cfg, ns)
    with torch.no_grad():
        result = (model.infer_action_replay if arm == 'S' or collect else model.infer_action)(
            **kwargs, compile_action_infer=True)
        action = postprocess(result['action'], processor, ns)
    torch.cuda.synchronize()
    seconds = time.perf_counter()-t
    if collect:
        torch.save(dict(schema='fastwam-canonical-teacher/v1', identity=identity,
                        conditions=str(base.with_suffix('.npz')), columns=collector.columns), base.with_suffix('.pt'))
    counts = dict(spec.counts) if arm == 'S' else dict(committed_updates=10, target_calls=10, target_B1_calls=10, target_state_rows=10)
    counts['canonical_chunks' if arm == 'T' else 'speculative_chunks'] = 1
    counts['PF_refresh_chunks'] = int(requested_arm == 'P' and arm == 'T')
    row = dict(identity=identity, arm=requested_arm, policy_role=arm, seconds=seconds, concurrent_quality_timing=True,
               server_identity=server_identity,
               collector_io_in_policy=collect, counts=counts, rounds=spec.rounds if arm == 'S' else [],
               normalized_action=result['action'].float().cpu().tolist(), action=action.tolist())
    write(base.with_suffix('.json'), row)
    status.update(phase='SERVING', updated=time.time(), calls=status['calls']+1, last_identity=identity)
    write(OUT / 'STATUS.json', status)
    return dict(action=action, seconds=seconds, counts=counts, server_identity=server_identity)


def receive(sock, count):
    chunks = bytearray()
    while len(chunks) < count:
        data = sock.recv(count-len(chunks))
        if not data: raise EOFError('socket closed')
        chunks.extend(data)
    return bytes(chunks)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        global failure_latch
        try:
            size = struct.unpack('!I', receive(self.request, 4))[0]
            assert 0 < size < 4*1024*1024
            req = pickle.loads(receive(self.request, size))
            with lock:
                try:
                    value = dict(ok=True, result=handle(req))
                except BaseException as exc:
                    # A failed cold audit or inference must not let a queued
                    # client continue with a partially initialized sampler.
                    if failure_latch is None: failure_latch = repr(exc)
                    status.update(phase='FAILED_LATCHED', error=failure_latch, updated=time.time())
                    write(OUT/'STATUS.json', status)
                    raise
        except BaseException as exc:
            value = dict(ok=False, error=repr(exc), traceback=traceback.format_exc())
            write(OUT / f'ERROR-{time.time_ns()}.json', value)
        data = pickle.dumps(value, protocol=4)
        self.request.sendall(struct.pack('!I', len(data))+data)


class Server(socketserver.TCPServer):
    # The two independent environment clients queue on one model thread.
    # PyTorch CUDA Graph trees use thread-local managers; creating a new
    # inference thread per request would repeatedly create cold graph state.
    allow_reuse_address = False


try:
    with Server(('127.0.0.1', args.port), Handler) as server:
        status.update(phase='READY', updated=time.time())
        write(OUT / 'STATUS.json', status)
        server.serve_forever(poll_interval=1.)
except BaseException as exc:
    status.update(phase='STOPPED_OR_FAILED', end=time.time(), error=repr(exc), traceback=traceback.format_exc())
    write(OUT / 'STATUS.json', status)
    raise
