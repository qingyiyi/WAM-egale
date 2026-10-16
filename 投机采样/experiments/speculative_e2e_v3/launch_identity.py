"""Wait for the exec cmdline to settle; never rerun a workload on pin failure."""
import os
import time
import ownership

def bootstrap_key(info):
    return (info.pid,info.ppid,info.pgid,info.start_ticks,info.uid)

def pin(p,cmd,*,lookup=ownership.read_process,clock=time.monotonic,
        sleep=time.sleep,parent_pid=None,on_boot=None):
    parent_pid=os.getpid() if parent_pid is None else parent_pid
    deadline=clock()+2
    first=None
    while clock()<deadline:
        info=lookup(p.pid)
        if info.uid!=os.geteuid() or info.pgid!=p.pid or info.ppid!=parent_pid:
            raise RuntimeError('launch UID/parent/group mismatch')
        if first is None:
            first=bootstrap_key(info)
            if on_boot:on_boot(info)
        if bootstrap_key(info)!=first:
            raise RuntimeError('launch PID identity changed')
        if info.argv==tuple(cmd):
            again=lookup(p.pid)
            if again==info:return info
        if p.poll() is not None:raise RuntimeError('launcher exited before stable identity')
        sleep(.01)
    raise RuntimeError('launch cmdline did not settle within two seconds')
