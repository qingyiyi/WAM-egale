"""Download the locally recorded public model revision, without file digests.

This performs real asset staging only; it does not instantiate or test a model.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import urllib.parse
import urllib.request


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.download-tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--job', required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    job = args.job.resolve(strict=True)
    deadline = manifest['prepare_deadline_unix']
    model = Path(manifest['remote_model_root'])
    model.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def one(index, item):
        if time.time() >= deadline:
            raise TimeoutError('preparation deadline before asset download')
        target = model / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError('no silent overwrite/retry of an existing model file')
        url = ('https://modelscope.cn/api/v1/models/' + manifest['model_id'] + '/repo?' +
               urllib.parse.urlencode({'Revision': item['revision'], 'FilePath': item['path']}))
        request = urllib.request.Request(url, headers={'User-Agent': 'WAM-frozen-asset-staging/1.0'})
        part = target.with_name(target.name + '.owned-part')
        count = 0
        phase_start = report_at = time.time()
        with urllib.request.urlopen(request, timeout=60) as response, part.open('xb') as output:
            if response.status != 200:
                raise RuntimeError(f'asset HTTP status {response.status}: {item["path"]}')
            length = response.headers.get('Content-Length')
            if length is not None and int(length) != item['bytes']:
                raise RuntimeError('public revision content-length differs from locally recorded asset')
            while True:
                if time.time() >= deadline:
                    raise TimeoutError('preparation deadline during asset download')
                chunk = response.read(4 * 2**20)
                if not chunk:
                    break
                output.write(chunk)
                count += len(chunk)
                if count > item['bytes']:
                    raise RuntimeError('asset exceeds recorded size')
                if time.time() - report_at >= 15:
                    write(job / f'MODEL-DOWNLOAD-{index}.json', dict(status='DOWNLOADING',
                          source_url=url, revision=item['revision'], path=item['path'],
                          bytes=count, expected_bytes=item['bytes'], elapsed_sec=time.time()-phase_start))
                    report_at = time.time()
        if count != item['bytes']:
            raise RuntimeError('incomplete asset length: ' + item['path'])
        part.replace(target)
        result = dict(status='COMPLETE_DECLARED_REVISION_AND_SIZE', path=item['path'],
                      revision=item['revision'], source_url=url, bytes=count,
                      elapsed_sec=time.time()-phase_start, remote_file_digest='NOT_COMPUTED',
                      independent_byte_identity='UNKNOWN_NOT_HASHED')
        write(job / f'MODEL-DOWNLOAD-{index}.json', result)
        return result

    results, errors = [], []
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {pool.submit(one, i, item): item for i, item in enumerate(manifest['files'])}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except BaseException as exc:
                    errors.append(dict(path=futures[future]['path'], error=repr(exc)))
        write(job / 'MODEL-DOWNLOAD-RESULT.json', dict(
            status='COMPLETE_ALL_PINNED_ASSETS' if not errors else 'FAILED_ASSET_STAGING',
            elapsed_sec=time.time()-started, files=results, failures=errors,
            requested=len(manifest['files']), no_model_forward=True, no_digest_computed=True))
        if errors:
            raise RuntimeError(str(errors))
    except BaseException as exc:
        if not (job / 'MODEL-DOWNLOAD-RESULT.json').exists():
            write(job / 'MODEL-DOWNLOAD-RESULT.json', dict(status='FAILED_ASSET_STAGING',
                  error=repr(exc), traceback=traceback.format_exc(), files=results, failures=errors))
        raise


if __name__ == '__main__':
    main()
