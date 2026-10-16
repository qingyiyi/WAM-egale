"""Write-ahead direct-attention intents with immutable hash-chain recovery."""
import hashlib
import json
from pathlib import Path
from counter_journal import Writer, canonical, ZERO


class Counter:
    def __init__(self, path, identity, *, cap, check=lambda: None):
        self.writer = Writer(path, identity)
        self.cap, self.check = cap, check
        self.attempted = self.returned = self.failed = 0
        self.writer.append('OPEN', self.summary())

    def summary(self):
        return dict(attempted=self.attempted, returned=self.returned, failed=self.failed, cap=self.cap)

    def call(self, label, fn):
        self.check()
        if self.attempted >= self.cap:
            raise RuntimeError('direct attention intent cap')
        self.attempted += 1
        self.writer.append('BEFORE', self.summary(), dict(label=label))
        try:
            out = fn()
        except BaseException as e:
            self.failed += 1
            self.writer.append('RAISE', self.summary(), dict(label=label,error_type=type(e).__name__))
            raise
        self.returned += 1
        self.writer.append('RETURN', self.summary(), dict(label=label))
        return out

    def close(self):
        try:
            self.writer.append('CLOSE', self.summary())
        finally:
            self.writer.close()


def recover(path, identity, cap):
    chunks = Path(path).read_bytes().split(b'\n')
    tail = chunks.pop()
    previous, attempted, returned, failed, pending, owner, closed = ZERO, 0, 0, 0, 0, None, False
    for i, line in enumerate(chunks):
        x = json.loads(line); h = x.pop('checksum')
        if (x['seq'] != i+1 or x['previous'] != previous or x['identity'] != identity
                or hashlib.sha256(canonical(x)).hexdigest() != h or closed):
            raise ValueError('direct journal identity/hash chain')
        if i == 0:
            if x['event'] != 'OPEN':
                raise ValueError('OPEN required')
            owner = x['owner']
        elif x['owner'] != owner or x['event'] == 'OPEN':
            raise ValueError('journal owner changed')
        if x['event'] == 'BEFORE':
            if pending: raise ValueError('pending direct intent')
            attempted += 1; pending = 1
        elif x['event'] in ('RETURN','RAISE'):
            if pending != 1: raise ValueError('result without intent')
            if x['event'] == 'RETURN': returned += 1
            else: failed += 1
            pending = 0
        elif x['event'] == 'CLOSE': closed = True
        elif x['event'] != 'OPEN': raise ValueError('unknown direct event')
        expected = dict(attempted=attempted, returned=returned, failed=failed, cap=cap)
        if x['state'] != expected or attempted > cap:
            raise ValueError('direct budget/state mismatch')
        previous = h
    if not chunks or closed and tail: raise ValueError('empty or post-close journal')
    return dict(status='CLOSED' if closed else 'INTERRUPTED_VERIFIED_PREFIX',
                counter=expected, inflight=pending, owner=owner, uncommitted_tail_bytes=len(tail))
