"""Fail-closed single-writer write-ahead counters; CPU/file I/O only.

BEFORE consumes budget before entering public math. RETURN confirms Python
forward return, not asynchronous CUDA device completion. Process death can
leave one unconfirmed intent; recovery never substitutes a parent's old 0.
Every event is fsynced, charged to the same outer timing span in every arm.
No recovery path resumes computation, repairs a file, or signals a process.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

from ledger_core import Ledger, STAGES
from task_contract import counter_enabled as cpu_enabled

MAX_BYTES = 4 * 2**20
ZERO = '0' * 64


class JournalError(RuntimeError):
    pass


def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def proc_start(pid):
    try:
        body = Path('/proc/' + str(pid) + '/stat').read_text()
        return int(body[body.rfind(')') + 2:].split()[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


class Writer:
    def __init__(self, path, identity):
        self.path = Path(path); self.identity = dict(identity)
        self.owner = dict(pid=os.getpid(), uid=os.getuid(), start_ticks=proc_start(os.getpid()))
        self.seq = 0; self.previous = ZERO; self.poisoned = False; self.fd = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        try:
            d = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try: os.fsync(d)
            finally: os.close(d)
        except BaseException:
            self.close(); raise

    def append(self, event, snapshot, detail=None):
        if self.fd is None or self.poisoned or os.getpid() != self.owner['pid']:
            raise JournalError('journal closed, poisoned, or not writer owner')
        core = dict(schema=1, seq=self.seq + 1, previous=self.previous,
                    identity=self.identity, owner=self.owner, event=event,
                    state=snapshot, detail={} if detail is None else detail)
        h = hashlib.sha256(canonical(core)).hexdigest()
        payload = canonical({**core, 'checksum': h}) + b'\n'
        try:
            view = memoryview(payload)
            while view:
                n = os.write(self.fd, view)
                if n <= 0: raise OSError('zero-length journal write')
                view = view[n:]
            os.fsync(self.fd)
        except BaseException:
            self.poisoned = True; raise
        self.seq += 1; self.previous = h

    def close(self):
        if self.fd is not None:
            fd, self.fd = self.fd, None; os.close(fd)


class PersistentLedger(Ledger):
    """Default off. A journal commit must succeed before target math starts."""
    def __init__(self, path=None, identity=None, *, enabled=False, on_commit=None):
        super().__init__(); self.failed = 0; self.pending = 0
        self.writer = None; self.on_commit = on_commit; self.closed = False
        if enabled:
            cpu_enabled()
            if path is None or not identity: raise ValueError('owned journal path/identity required')
            self.writer = Writer(path, identity)
            try: self.publish('OPEN')
            except BaseException: self.writer.close(); raise

    def summary(self):
        return {**super().summary(), 'target_failed': self.failed,
                'target_inflight': self.pending, 'counter_persisted': self.writer is not None,
                'target_completed_means_host_forward_return': True}

    def publish(self, event, detail=None):
        if self.writer is not None:
            self.writer.append(event, self.summary(), detail)
            if self.on_commit is not None: self.on_commit(event, self.summary())

    def begin(self, key):
        if self.closed or self.pending: raise JournalError('closed ledger or outstanding forward')
        super().begin(key); self.publish('BEGIN')

    def before_forward(self):
        if self.closed or self.pending: raise JournalError('closed ledger or outstanding forward')
        super().before_forward(); self.pending = 1; self.publish('BEFORE')

    def after_forward(self):
        if self.closed or self.pending != 1: raise JournalError('RETURN requires one outstanding forward')
        super().after_forward(); self.pending = 0; self.publish('RETURN')

    def forward_failed(self, error):
        if self.pending != 1: raise JournalError('RAISE requires one outstanding forward')
        self.failed += 1; self.pending = 0
        self.publish('RAISE', {'error_type': type(error).__name__})

    def close(self, reason='normal'):
        if self.closed: return
        try:
            if self.writer is not None and not self.writer.poisoned:
                self.publish('CLOSE', {'reason': reason})
        finally:
            self.closed = True
            if self.writer is not None: self.writer.close()


def recover(path, expected_identity):
    """Verified surviving prefix only; missing/corrupt data is never a zero."""
    path = Path(path)
    try: fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except FileNotFoundError:
        return dict(status='UNKNOWN_NO_JOURNAL', ledger=None, exact_math_counts=False)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode) or st.st_uid != os.getuid() or st.st_size > MAX_BYTES:
            raise JournalError('journal file type/owner/size')
        chunks = []
        while True:
            b = os.read(fd, min(65536, MAX_BYTES + 1))
            if not b: break
            chunks.append(b)
            if sum(map(len, chunks)) > MAX_BYTES: raise JournalError('journal above size cap')
        body = b''.join(chunks)
    finally: os.close(fd)
    pieces = body.split(b'\n'); tail = pieces.pop()
    if not pieces:
        return dict(status='UNKNOWN_NO_COMMITTED_HEADER', ledger=None,
                    uncommitted_tail_bytes=len(tail), exact_math_counts=False)
    base = Ledger(); failed = pending = 0; last = None; previous = ZERO; owner = None; terminal = False
    for i, line in enumerate(pieces):
        if not line: raise JournalError('empty/interior journal record')
        try:
            v = json.loads(line)
            if not isinstance(v, dict): raise ValueError('record must be object')
            checksum = v.pop('checksum')
        except (ValueError, KeyError) as exc: raise JournalError('malformed complete record') from exc
        if (v.get('schema') != 1 or v.get('seq') != i + 1 or v.get('previous') != previous
                or hashlib.sha256(canonical(v)).hexdigest() != checksum):
            raise JournalError('record checksum/sequence/hash-chain')
        if v.get('identity') != expected_identity: raise JournalError('journal job/plan identity')
        if i == 0:
            owner = v['owner']
            if (not isinstance(owner,dict) or set(owner)!={'pid','uid','start_ticks'} or
                    any(type(owner[k]) is not int for k in owner) or owner['pid']<=0 or owner['start_ticks']<=0
                    or v['event'] != 'OPEN' or owner['uid'] != st.st_uid):
                raise JournalError('OPEN/owner required')
        elif v.get('owner') != owner or v['event'] == 'OPEN' or terminal:
            raise JournalError('single owner/OPEN/terminal invariant')
        event = v['event']; state = v['state']
        try:
            if event == 'BEGIN':
                if pending: raise JournalError('BEGIN with outstanding forward')
                base.begin(state['stage'])
            elif event == 'BEFORE':
                if pending: raise JournalError('duplicate outstanding forward')
                base.before_forward(); pending = 1
            elif event == 'RETURN':
                if pending != 1: raise JournalError('return without intent')
                base.after_forward(); pending = 0
            elif event == 'RAISE':
                if pending != 1: raise JournalError('raise without intent')
                failed += 1; pending = 0
            elif event == 'CLOSE': terminal = True
            elif event != 'OPEN': raise JournalError('unsupported event')
        except (ValueError, RuntimeError) as exc:
            raise JournalError('invalid budget transition: ' + str(exc)) from exc
        expected = {**base.summary(), 'target_failed': failed, 'target_inflight': pending,
                    'counter_persisted': True, 'target_completed_means_host_forward_return': True}
        if canonical(state) != canonical(expected): raise JournalError('snapshot differs from replayed budget transitions')
        previous = checksum; last = {**v, 'checksum': checksum}
    if terminal and tail: raise JournalError('data after closed journal')
    live = proc_start(owner['pid']) == owner['start_ticks']
    return dict(status='CLOSED' if terminal else 'LIVE_VERIFIED_PREFIX' if live else 'INTERRUPTED_VERIFIED_PREFIX',
                ledger=last['state'], committed_records=len(pieces), last_event=last['event'],
                last_checksum=previous, uncommitted_tail_bytes=len(tail), writer=owner,
                writer_same_process_alive=live, journal_terminal=terminal,
                counter_semantics='write-ahead budget intent / confirmed Python forward return',
                exact_math_counts=False, CUDA_completion_proven=False,
                unconfirmed_call_upper_bound=pending,
                actual_invocation_lower_bound=base.completed + failed,
                actual_invocation_upper_bound=base.attempted,
                possible_unconfirmed_return_upper_bound=pending,
                note='No resume, refund, inferred success, or fallback to stale parent counters.')
