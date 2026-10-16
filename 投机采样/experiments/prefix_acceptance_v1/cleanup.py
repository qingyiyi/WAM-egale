"""Bounded common exit wait; original strict live ownership and group checks.

Terminal Popen children are polled/reaped, never signalled. Live group signals
require exact pinned identity and every live group member's owned ancestry.
"""
import os
from pathlib import Path
import signal
import time

import ownership


def state(pid):
    raw=Path(f'/proc/{pid}/stat').read_text()
    return raw[raw.rfind(')')+2:].split()[0]


def signalable(process,pins,*,lookup=ownership.read_process,state_lookup=state):
    if process.poll() is not None:return None
    try:info=lookup(process.pid)
    except FileNotFoundError:
        if process.poll() is not None:return None
        raise RuntimeError('live root disappeared during cleanup inspection')
    pin=pins.get(info.pid)
    if pin is None:raise RuntimeError('cleanup root is not pinned')
    if not info.argv:
        if (info.start_ticks,info.uid,info.pgid)!=(pin.start_ticks,pin.uid,pin.pgid):
            raise RuntimeError('terminal root identity mismatch')
        if state_lookup(info.pid)!='Z':raise RuntimeError('empty argv live root refused')
        again=lookup(info.pid)
        if again!=info or state_lookup(info.pid)!='Z':raise RuntimeError('terminal root changed')
        process.poll()
        return None
    if not ownership.is_owned(info,pins,lookup):raise RuntimeError('unowned live root cleanup refused')
    return info


def group_owned(info,pins):
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():continue
        try:
            member=ownership.read_process(int(path.name))
            if member.pgid!=info.pgid:continue
            if not member.argv and state(member.pid)=='Z':continue
        except FileNotFoundError:continue
        if not ownership.is_owned(member,pins):raise RuntimeError('unowned live group member; signal refused')


def roots(processes,pins,*,deadline,wait_seconds=10.,killpg=os.killpg):
    records=[]
    def pending():return [p for p in processes if p.poll() is None]
    def wait_until(end):
        while pending() and time.monotonic()<end:time.sleep(.05)
    # One shared natural exit interval, not two seconds per sequential root.
    wait_until(min(time.monotonic()+wait_seconds,deadline-13))
    for sig,end in [(signal.SIGTERM,min(time.monotonic()+7,deadline-5)),(signal.SIGKILL,deadline)]:
        for process in pending():
            info=signalable(process,pins)
            if info is None:continue
            group_owned(info,pins)
            # Recheck the exact root immediately before any live group signal.
            latest=signalable(process,pins)
            if latest is None:continue
            if latest!=info:raise RuntimeError('live root metadata changed before signal')
            try:killpg(info.pgid,sig)
            except ProcessLookupError:process.poll();continue
            records.append(dict(pid=info.pid,signal=int(sig)))
        wait_until(end)
        if not pending():break
    if pending():raise RuntimeError('bounded owned cleanup did not finish')
    return dict(all_roots_reaped=True,signals=records,
                exits=[dict(pid=p.pid,returncode=p.poll()) for p in processes])
