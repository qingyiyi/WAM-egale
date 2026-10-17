"""Produce pinned community suite assets; streamed real downloads, no digests/tests."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import threading
import time
from urllib.parse import urlsplit

import requests

REVISIONS = {
    'object': '489806feb4074eee71f97180948e8b85e33ebeda',
    'goal': '42455fd87bc5b9c4613cc13b03be2f2d80605efe',
    'spatial': '933ab666f8cd21ff71d8765dea9fdd110512034f',
}
WEIGHT_BYTES = 10177831668
ROOT = Path(__file__).resolve().parents[3]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.pending')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', type=Path, required=True)
    ap.add_argument('--asset-root', type=Path, required=True)
    ap.add_argument('--shared-components', type=Path, required=True)
    ap.add_argument('--source-commit', required=True)
    ap.add_argument('--resume-from', type=Path)
    ap.add_argument('--suites', nargs='+', choices=tuple(REVISIONS), default=list(REVISIONS))
    ap.add_argument('--transport-host', choices=('huggingface.co','hf-mirror.com'), default='huggingface.co')
    ap.add_argument('--wall-cap', type=int, default=7200)
    args = ap.parse_args()
    job, base = args.job.resolve(), args.shared_components.resolve(strict=True)
    target = args.asset_root.resolve()
    if job.exists() or (target.exists() and not args.resume_from) or not 600 <= args.wall_cap <= 7200:
        raise ValueError('fresh job/cache and bounded two-hour lifecycle required')
    if len(set(args.suites)) != len(args.suites):raise ValueError('duplicate suite assets')
    if args.resume_from:
        old=args.resume_from.resolve(strict=True)
        old_plan=json.loads((old/'REGISTRATION.json').read_text())
        if (not (old/'FAILED.json').is_file() or Path(old_plan['asset_root']).resolve()!=target
                or old_plan['revisions']!=REVISIONS):
            raise ValueError('resume requires closed failed predecessor and same pinned asset root')
        try:
            live=int(Path('/proc/'+str(old_plan['pid'])+'/stat').read_text().rpartition(') ')[2].split()[19])
        except FileNotFoundError:live=None
        if live==old_plan['birth_ticks']:raise ValueError('predecessor still alive; never mutate active download')
    if len(os.sched_getaffinity(0)) > 1:
        raise ValueError('download orchestrator is pinned to one CPU')
    for name in ('vae', 'text_encoder', 'tokenizer'):
        if not (base/name).is_dir():
            raise ValueError('original shared component absent: ' + name)
    job.mkdir(parents=True); target.mkdir(parents=True,exist_ok=bool(args.resume_from))
    started, birth = time.monotonic(), time.time()
    cap = 40 * 2**30
    registration = dict(goal='three pinned suite model assets, no model load', source_commit=args.source_commit,
        scientific_parent='4f04b35e44cbc00483dc6245ad57abec718385c9', revisions=REVISIONS,
        input_shared_components=str(base), asset_root=str(target), suites=args.suites, transport_host=args.transport_host, wall_cap_sec=args.wall_cap,
        birth_epoch=birth, deadline_epoch=birth+args.wall_cap, work_deadline_epoch=birth+args.wall_cap-120, storage_cap_bytes=cap,
        cpu=1, gpu=0, max_downloads=2, cleanup='HTTP threads joined before exit; retain partial files and receipts',
        predecessor=str(args.resume_from.resolve()) if args.resume_from else None,
        pid=os.getpid(), birth_ticks=int(Path('/proc/self/stat').read_text().rpartition(') ')[2].split()[19]))
    write(job/'REGISTRATION.json', registration)
    rows, lock, stop = {}, threading.RLock(), threading.Event()

    def pulse(suite, **values):
        with lock:
            rows.setdefault(suite, {}).update(values)
            write(job/'status.json', dict(status='PRODUCING_REAL_ASSETS', elapsed_sec=time.monotonic()-started,
                suites=rows, wall_cap_sec=args.wall_cap, no_model_or_GPU=True))

    def guard():
        if stop.is_set() or time.time() >= registration['work_deadline_epoch']:
            raise TimeoutError('asset lifecycle cap or peer failure')
        if shutil.disk_usage(target).free < 10 * 2**30:
            raise RuntimeError('local asset disk reserve10GiB')

    def fetch(suite, relative, dest, expected=None):
        guard()
        method='resolve' if expected is not None else 'raw'
        suffix='/Wjjjh/lingbot-va-libero-' + suite + '/'+method+'/' + REVISIONS[suite] + '/' + relative
        publisher_url='https://huggingface.co'+suffix
        url = 'https://'+args.transport_host+suffix
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            if not args.resume_from or (expected is not None and dest.stat().st_size!=expected):
                raise FileExistsError('existing file has no admitted predecessor or expected byte count')
            return dict(source=publisher_url,transport=url,file=str(dest),bytes=dest.stat().st_size,
                status='REUSED_PREDECESSOR_COMPLETE_PUBLISHED_ASSET',predecessor=str(args.resume_from.resolve()),
                cryptographic_identity='UNKNOWN_NOT_COMPUTED')
        part = dest.with_name(dest.name+'.partial.'+job.name)
        received, last = 0, 0.
        with requests.get(url, stream=True, timeout=(15, 20), headers={'Accept-Encoding': 'identity'}) as r:
            if r.status_code != 200:
                raise RuntimeError('HTTP_status_' + str(r.status_code))
            declared = r.headers.get('Content-Length')
            if expected is not None and declared and int(declared) != expected:
                raise ValueError('publisher file size versus response length differs')
            limit = expected if expected is not None else 2**20
            with part.open('xb') as f:
                for block in r.iter_content(8*2**20):
                    guard(); received += len(block)
                    if received > limit:
                        raise ValueError('asset declared size exceeded')
                    f.write(block)
                    if time.monotonic()-last >= 5:
                        pulse(suite, file=relative, bytes=received, expected_bytes=expected,
                            MiB_sec=received/max(.001, time.monotonic()-started)/2**20)
                        last = time.monotonic()
                f.flush(); os.fsync(f.fileno())
            if expected is not None and received != expected:
                raise ValueError('incomplete model asset')
            os.link(part, dest); part.unlink()
            return dict(source=publisher_url, transport=url, bytes=received, endpoint_host=urlsplit(r.url).hostname,
                file=str(dest), status='COMPLETE_SIZE_ONLY_NO_DIGEST', cryptographic_identity='UNKNOWN_NOT_COMPUTED')

    def suite_assets(suite):
        folder = target/suite
        published, runtime = folder/'publisher', folder/'runtime'
        receipts=[]
        try:
            for name in ('README.md', 'norm_stats.json', 'checkpoint_metadata.json', 'transformer/config.json'):
                receipts.append(fetch(suite, name, published/name))
            norm = json.loads((published/'norm_stats.json').read_text())
            if set(('q01','q99')) <= set(norm):
                q = {k: norm[k] for k in ('q01','q99')}
            elif 'norm_stat' in norm:
                q = {k: norm['norm_stat'][k] for k in ('q01','q99')}
            else:
                raise ValueError('published seven-channel action quantiles absent')
            if any(len(v) != 7 or not all(math.isfinite(float(x)) for x in v) for v in q.values()):
                raise ValueError('published action quantile shape/finite differs')
            if any(q['q01'][i] >= q['q99'][i] for i in range(7)):
                raise ValueError('published used-channel quantiles not increasing')
            receipts.append(fetch(suite, 'transformer/diffusion_pytorch_model.safetensors',
                published/'transformer/diffusion_pytorch_model.safetensors', WEIGHT_BYTES))
            (runtime/'transformer').mkdir(parents=True, exist_ok=False)
            cfg=json.loads((published/'transformer/config.json').read_text())
            cfg['attn_mode']='torch'
            write(runtime/'transformer/config.json', cfg)
            (runtime/'transformer/diffusion_pytorch_model.safetensors').symlink_to(
                published/'transformer/diffusion_pytorch_model.safetensors')
            for name in ('vae','text_encoder','tokenizer'):
                (runtime/name).symlink_to(base/name, target_is_directory=True)
            info=dict(suite='libero_'+suite, checkpoint_id='Wjjjh/lingbot-va-libero-'+suite,
                publisher_revision=REVISIONS[suite], model_path=str(runtime),
                norm_stat={k:list(v)+[0.]*23 for k,v in q.items()},
                shared_component_source=str(base), receipts=receipts,
                archive_config_unchanged=True, runtime_only_attn_mode='torch',
                publisher_SR_not_project_result=True, cryptographic_identity='UNKNOWN_NOT_COMPUTED')
            write(folder/'ASSET-READY.json', info)
            pulse(suite, status='COMPLETE_PINNED_ASSET', bytes=WEIGHT_BYTES)
            return info
        except BaseException as exc:
            stop.set()
            failure=dict(status='FAILED_ASSET_PREPARATION', suite=suite, error_type=type(exc).__name__,
                message=str(exc) if isinstance(exc,(ValueError,RuntimeError,TimeoutError)) else 'network_or_IO_error_see_public_source',
                receipts=receipts, elapsed_sec=time.monotonic()-started)
            write(job/(suite+'-FAILED.json'), failure); pulse(suite, status='FAILED', error_type=type(exc).__name__)
            raise

    def alarm(*_):
        stop.set()
        raise TimeoutError('entire asset job lifecycle expired')
    signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, alarm); signal.alarm(args.wall_cap-120)
    completed=[]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(suite_assets, s) for s in args.suites]
            for f in as_completed(futures):
                completed.append(f.result())
        total=sum(p.stat().st_size for p in target.rglob('*') if p.is_file() and not p.is_symlink())
        if total > cap:
            raise RuntimeError('registered asset storage cap')
        result=dict(status='COMPLETE_REGISTERED_SUITE_ASSETS', wall_sec=time.monotonic()-started,
            asset_bytes=total, suites=completed, CPU=1, GPU=0, model_loads=0, tests=0, digests=0)
        write(job/'RESULT.json',result); write(job/'status.json',result)
        print(json.dumps(dict(status=result['status'],bytes=total,wall_sec=result['wall_sec'])),flush=True)
        return 0
    except BaseException as exc:
        stop.set()
        failure=dict(status='FAILED_OR_PARTIAL_ASSET_PRODUCTION', completed_suites=[x['suite'] for x in completed],
            error_type=type(exc).__name__, wall_sec=time.monotonic()-started,
            retained_partial_files=True, no_automatic_retry=True)
        write(job/'FAILED.json',failure);write(job/'status.json',failure)
        return 2
    finally:
        signal.alarm(0)


if __name__=='__main__':
    raise SystemExit(main())
