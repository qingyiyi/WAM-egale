"""Persistent NVML context: retains startup samples, requires stable idle.

Never initializes CUDA, never signals GPU processes. v2 process ABI is explicit;
unsupported NVML symbols/inspection failures block instead of guessing.
"""
import ctypes as C
from ctypes.util import find_library
import importlib.util
from pathlib import Path
import sys
import threading


class Memory(C.Structure):
    # /usr/local/cuda/include/nvml.h: nvmlMemory_v2_t. v1.used includes
    # driver/firmware reservations and cannot mean "task allocation == 0".
    _fields_=[('version',C.c_uint),('total',C.c_ulonglong),('reserved',C.c_ulonglong),
              ('free',C.c_ulonglong),('used',C.c_ulonglong)]
MEMORY_VERSION=C.sizeof(Memory)|(2<<24)
MEMORY_ACCOUNTING='NVML_MEMORY_V2_ALLOCATED'


def memory_fields(m):
    if m.version!=MEMORY_VERSION or m.total!=m.reserved+m.free+m.used:
        raise RuntimeError('NVML v2 memory ABI/accounting inconsistent')
    return dict(memory_total_bytes=int(m.total),memory_used_bytes=int(m.used),
                memory_reserved_bytes=int(m.reserved),memory_free_bytes=int(m.free),
                memory_accounting=MEMORY_ACCOUNTING,memory_struct_version=int(m.version))
class Util(C.Structure):
    _fields_=[('gpu',C.c_uint),('memory',C.c_uint)]
class Proc(C.Structure):
    _fields_=[('pid',C.c_uint),('usedGpuMemory',C.c_ulonglong),('gpuInstanceId',C.c_uint),('computeInstanceId',C.c_uint)]


class NVML:
    def __init__(self, index=0):
        self.lib=C.CDLL(find_library('nvidia-ml') or 'libnvidia-ml.so.1'); self.lock=threading.RLock()
        self._call('nvmlInit_v2')
        self.handle=C.c_void_p(); self._call('nvmlDeviceGetHandleByIndex_v2',C.c_uint(index),C.byref(self.handle))
        buf=C.create_string_buffer(96); self._call('nvmlDeviceGetUUID',self.handle,buf,C.c_uint(len(buf)))
        self.uuid=buf.value.decode()
        # No silent v4 ABI substitution.
        getattr(self.lib,'nvmlDeviceGetComputeRunningProcesses_v2')
        getattr(self.lib,'nvmlDeviceGetGraphicsRunningProcesses_v2')
        getattr(self.lib,'nvmlDeviceGetMemoryInfo_v2')
    def _call(self,name,*args):
        rc=getattr(self.lib,name)(*args)
        if rc!=0: raise RuntimeError('NVML '+name+' error '+str(rc))
    def processes(self,kind):
        entries=(Proc*512)(); n=C.c_uint(512)
        self._call('nvmlDeviceGet'+kind+'RunningProcesses_v2',self.handle,C.byref(n),entries)
        return [dict(pid=int(entries[i].pid),memory_bytes=int(entries[i].usedGpuMemory)) for i in range(n.value)]
    def sample(self):
        with self.lock:
            m=Memory();m.version=MEMORY_VERSION;u=Util()
            self._call('nvmlDeviceGetMemoryInfo_v2',self.handle,C.byref(m))
            self._call('nvmlDeviceGetUtilizationRates',self.handle,C.byref(u))
            return dict(uuid=self.uuid,**memory_fields(m),utilization_percent=int(u.gpu),
                        compute=self.processes('Compute'),graphics=self.processes('Graphics'))
    def close(self):
        with self.lock: self._call('nvmlShutdown')


class Idle:
    def __init__(self, samples=15, max_util=2, interval=1):
        self.required=samples;self.max_util=max_util;self.interval=interval;self.samples=[]
    def add(self,s,monotonic):
        quiet=s.get('memory_accounting')==MEMORY_ACCOUNTING and s['memory_used_bytes']==0 and not s['compute'] and not s['graphics'] and s['utilization_percent']<=self.max_util
        if not quiet: self.samples=[]; return False
        if self.samples and monotonic-self.samples[-1]<self.interval*.9: return False
        self.samples.append(monotonic)
        return len(self.samples)>=self.required and monotonic-self.samples[-self.required]>=self.interval*(self.required-1)*.9


def ownership_module(path):
    name='kv_attr_private_ownership_v2'
    if name in sys.modules: return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module
