"""取固定官方版本的项目内C++ headers；不安装或更改系统环境。"""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

COMMIT = 'f7b19de32c5d1f3cedfc735c2849f12b537522ee'
DEST = Path(__file__).resolve().parents[2] / '.pipeline/cache/cutlass-v3.5.1'


def main():
    if DEST.exists():
        raise FileExistsError('不覆盖已有依赖；请先检查SOURCE.json')
    DEST.mkdir(parents=True)
    url = f'https://codeload.github.com/NVIDIA/cutlass/tar.gz/{COMMIT}'
    files, size = [], 0
    with urllib.request.urlopen(url, timeout=90) as response:
        with tarfile.open(fileobj=response, mode='r|gz') as archive:
            for item in archive:
                parts = Path(item.name).parts[1:]
                if not item.isfile() or not parts or (parts[0] != 'include' and parts != ('LICENSE.txt',)):
                    continue
                if '..' in parts or size + item.size > 100 * 1024**2:
                    raise RuntimeError('依赖路径或容量越界')
                payload = archive.extractfile(item).read()
                path = DEST.joinpath(*parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                size += len(payload)
                files.append(dict(path=str(Path(*parts)), bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
    record = dict(repository='https://github.com/NVIDIA/cutlass', tag='v3.5.1',
        commit=COMMIT, url=url, bytes=size, files=files)
    (DEST / 'SOURCE.json').write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(dict(directory=str(DEST), commit=COMMIT, files=len(files), bytes=size)))


if __name__ == '__main__':
    main()
