"""Prepared 14D teacher/speculative RPC server; requires a reviewed Robo panel.

Two isolated environment processes queue on this single model thread. The
registration must explicitly choose common eager/compile behavior for all arms.
"""
import argparse
import hashlib
import json
import os
import pickle
import signal
import socket
import socketserver
import struct
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from adapter import install
from draft import load as load_draft
from policy_bridge import install_policy_bridge, postprocess, prepare
from policy_schedule import policy_role
from real_speculative import CanonicalCollector, RealSpeculative
from runtime import ROOT, SOURCE, build, write


def receive(sock, count):
    data = bytearray()
    while len(data) < count:
        block = sock.recv(count - len(data))
        if not block:
            raise EOFError('client disconnected')
        data.extend(block)
    return bytes(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--worker', required=True)
    parser.add_argument('--assets', required=True)
    args = parser.parse_args()
    reg = json.loads((ROOT / 'REGISTRATION.json').read_text())
    panel = json.loads((ROOT / 'PANEL.json').read_text())
    review = ROOT / 'CANONICAL-REVIEW.json'
    assert panel['canonical_review_passed'] and panel['action_dim'] == 14
    assert reg['canonical_review_sha256'] == hashlib.sha256(review.read_bytes()).hexdigest()
    assert json.loads(review.read_text())['passed']
    assert reg['checkpoint_revision'] == '8eaceeb24c3cc92ff2a9c9a9d266a4941b836705'
    assert type(reg['compile_action_infer']) is bool
    compiled = reg['compile_action_infer']
    out = ROOT / 'results' / args.worker
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    deadline = min(reg['deadline_epoch'] - 300, start + reg['server_cap_seconds'])
    assert deadline > start
    status = dict(status='LOADING', start=start, deadline=deadline, pid=os.getpid(), calls=0)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('registered server cap')))
    signal.alarm(max(1, int(deadline - start)))
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    write(out / 'STATUS.json', status)
    policy, cfg = build(args.assets, out)
    model = policy.model
    install(model, SOURCE, out / 'ADAPTED-INFER-SOURCE.py')
    install_policy_bridge(policy, out / 'ADAPTED-POLICY-PREFIX.py')
    collector = CanonicalCollector(model)
    spec = None
    first = True
    failure = None
    server_identity = dict(host=socket.gethostname(), worker=args.worker,
        gpu=torch.cuda.get_device_name(), visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        torch=torch.__version__, cuda=torch.version.cuda, compile_action_infer=compiled,
        model_load=str(out / 'MODEL-LOAD.json'))

    @torch.no_grad()
    def predict(request):
        nonlocal spec, first
        assert failure is None and time.time() < deadline
        if request['op'] == 'ping':
            return dict(ready=True, calls=status['calls'])
        assert request['op'] == 'predict'
        identity = request['identity']
        episode = ROOT / 'results/episodes' / identity['episode']
        assert episode.parent == ROOT / 'results/episodes' and episode.name == identity['episode']
        episode.mkdir(parents=True, exist_ok=True)
        chunk = int(identity['chunk'])
        assert chunk >= 0 and identity['split'] in ('FIT', 'CAL', 'DEV', 'FULL')
        base = episode / f'chunk-{chunk:03d}'
        assert not base.with_suffix('.json').exists(), 'immutable chunk'
        requested_arm = request['arm']
        arm = policy_role(requested_arm, chunk)
        collect = identity['split'] in ('FIT', 'CAL')
        assert not collect or requested_arm == 'T'
        obs = request['obs']
        raw = {c: np.asarray(obs['observation'][c]['rgb'])
               for c in ('head_camera', 'left_camera', 'right_camera')}
        raw['joint'] = np.asarray(obs['joint_action']['vector'])
        np.savez_compressed(base.with_suffix('.npz'), **raw)
        write(episode / 'IDENTITY.json', dict(identity, arm=requested_arm, instruction=request['instruction']))
        kwargs = prepare(policy, obs, request['instruction'])
        if first:
            begin = time.perf_counter()
            official = policy._infer_action_chunk(obs, request['instruction'])
            eager = model.infer_action(**kwargs, compile_action_infer=False)
            assert np.array_equal(official, postprocess(policy, eager)), 'wrapper bridge changed action'
            model.empirical_replay = collector
            adapted = model.infer_action_replay(**kwargs, compile_action_infer=False)
            assert torch.equal(eager['action'], adapted['action']), 'collector changed canonical'
            comparison = dict(wrapper_action_exact=True, collector_eager_exact=True,
                              compiled=compiled, extra_full_policy_calls=3)
            if compiled:
                other = model.infer_action(**kwargs, compile_action_infer=True)['action'].float()
                ref = eager['action'].float()
                diff = other - ref
                rel = float(diff.norm() / ref.norm().clamp_min(1e-6))
                absolute = float(diff.abs().max())
                limit = .05 + .02 * float(ref.abs().max())
                comparison.update(relative_L2=rel, relative_L2_limit=.01,
                                  max_abs=absolute, max_abs_limit=limit, extra_full_policy_calls=4)
                assert rel <= .01 and absolute <= limit, comparison
            comparison.update(seconds=time.perf_counter() - begin,
                scheduler_t=collector.columns['timestep'].tolist(),
                scheduler_delta=collector.columns['delta'].tolist(),
                input_observation=str(base.with_suffix('.npz')), server_identity=server_identity)
            write(out / 'CANONICAL-PARITY.json', comparison)
            first = False
        if arm == 'S' and spec is None:
            selected = json.loads((ROOT / 'training/SELECTED.json').read_text())
            checkpoint = ROOT / 'training/checkpoint-3000.pt'
            assert selected['schema'] == 'fastwam-robotwin-selected-v1'
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == selected['checkpoint_sha256']
            spec = RealSpeculative(model, load_draft(checkpoint), selected['tau'])
            model.empirical_replay = spec
            spec.validate = True
            begin = time.perf_counter()
            model.infer_action_replay(**kwargs, compile_action_infer=compiled)
            torch.cuda.synchronize()
            write(out / 'B7-AUDIT.json', spec.audit_records)
            write(out / 'B7-CALL-SHAPES.json', spec.audit_inputs['call_shapes'])
            write(out / 'B7-ACCEPTANCE-SENSITIVITY.json', spec.acceptance_audit)
            torch.save(spec.audit_inputs, out / 'B7-AUDIT-INPUT.pt')
            write(out / 'SPEC-COLD-VALIDATION.json', dict(seconds=time.perf_counter() - begin,
                scheduler_policy_counts=dict(spec.counts), extra_audit_target_calls=8,
                extra_audit_target_state_rows=14, server_identity=server_identity))
            spec.validate = False
        model.empirical_replay = spec if arm == 'S' else collector
        torch.cuda.synchronize()
        begin = time.perf_counter()
        kwargs = prepare(policy, obs, request['instruction'])
        method = model.infer_action_replay if arm == 'S' or collect else model.infer_action
        result = method(**kwargs, compile_action_infer=compiled)
        action = postprocess(policy, result)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - begin
        if collect:
            torch.save(dict(schema='fastwam-robotwin-canonical-teacher/v1', identity=identity,
                conditions=str(base.with_suffix('.npz')), columns=collector.columns), base.with_suffix('.pt'))
        counts = dict(spec.counts) if arm == 'S' else dict(committed_updates=10,
            target_calls=10, target_B1_calls=10, target_state_rows=10)
        counts['canonical_chunks' if arm == 'T' else 'speculative_chunks'] = 1
        counts['PF_refresh_chunks'] = int(requested_arm == 'P' and arm == 'T')
        assert counts['committed_updates'] == 10
        write(base.with_suffix('.json'), dict(identity=identity, arm=requested_arm, policy_role=arm,
            seconds=seconds, concurrent_quality_timing=True, collector_io_in_policy=collect,
            server_identity=server_identity, counts=counts, rounds=spec.rounds if arm == 'S' else [],
            normalized_action=result['action'].float().cpu().tolist(), action=action.tolist()))
        status.update(status='SERVING', calls=status['calls'] + 1, last_identity=identity, updated=time.time())
        write(out / 'STATUS.json', status)
        return dict(action=action, seconds=seconds, counts=counts, server_identity=server_identity)

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            nonlocal failure
            try:
                size = struct.unpack('!I', receive(self.request, 4))[0]
                assert 0 < size < 16 * 1024 * 1024
                reply = dict(ok=True, result=predict(pickle.loads(receive(self.request, size))))
            except BaseException as exc:
                failure = failure or repr(exc)
                status.update(status='FAILED_LATCHED', error=failure, updated=time.time())
                write(out / 'STATUS.json', status)
                reply = dict(ok=False, error=repr(exc), traceback=traceback.format_exc())
                write(out / f'ERROR-{time.time_ns()}.json', reply)
            payload = pickle.dumps(reply, protocol=4)
            self.request.sendall(struct.pack('!I', len(payload)) + payload)

    class Server(socketserver.TCPServer):
        allow_reuse_address = False

    try:
        with Server(('127.0.0.1', args.port), Handler) as server:
            status.update(status='READY', updated=time.time(), server_identity=server_identity)
            write(out / 'STATUS.json', status)
            server.serve_forever(poll_interval=1.)
    except BaseException as exc:
        status.update(status='STOPPED_OR_FAILED', error=repr(exc), end=time.time())
        write(out / 'STATUS.json', status)
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
