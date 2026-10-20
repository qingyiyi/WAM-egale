"""Reproduce historical M39 controls from an explicitly pinned source file.

The current whole_layer.cu is the active FA2 implementation. Rebuilding these
historical controls requires WAM_M39_SOURCE pointing to the archived v1 source.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path

SOURCE_SHA='df8b386f302b9904ac4ef8ab08d171ef6d9e6f8143451831c3318f5e0835ef0e'

def replace_once(s,a,b):
    assert s.count(a)==1,(a,s.count(a));return s.replace(a,b)

def generate(root,kinds,instrument_path=None):
    src=Path(__file__).parent/'csrc'
    source=Path(os.environ.get('WAM_M39_SOURCE',src/'whole_layer.cu'))
    raw=source.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==SOURCE_SHA,(
        'Historical M39 requires WAM_M39_SOURCE with SHA '+SOURCE_SHA+
        '; build current optimized code with build_whole_layer.py')
    root.mkdir(parents=True,exist_ok=True);manifest=[]
    for kind in kinds:
        assert kind in ('sync','async','profile')
        s=raw.decode()
        if kind=='sync':
            s=replace_once(s,'B x[64*32],w[64*32];','B x[64*40],w[64*40];')
            for a,b in [('((uint4*)s.x)[i]=','*(uint4*)(s.x+r*40+c)='),
                        ('((uint4*)s.w)[i]=','*(uint4*)(s.w+r*40+c)='),
                        ('s.x+(m0+j*16)*32+t*16,32','s.x+(m0+j*16)*40+t*16,40'),
                        ('s.w+(n0+j*16)*32+t*16,32','s.w+(n0+j*16)*40+t*16,40')]:s=replace_once(s,a,b)
        else:
            s=replace_once(s,'B x[64*32],w[64*32];','B x[2][64*40],w[2][64*40];')
            begin=s.index('__device__ void gemm_tile(');end=s.index('__device__ void linear(',begin)
            s=s[:begin]+'#include "gemm_pipeline.cuh"\n'+s[end:]
        if kind=='profile':
            assert instrument_path is not None
            spec=importlib.util.spec_from_file_location('wam_profile_instrument',instrument_path)
            mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);s=mod.instrument(s)
        ns='wam_whole_pipeline_'+kind+'_v1';norm_ns='wam_norm_pipeline_'+kind
        s=s.replace('wam_whole_layer_v1',ns).replace('wam_optimized_norm',norm_ns)
        s=s.replace('"norm.cuh"','"norm_'+kind+'.cuh"')
        (root/(kind+'.cu')).write_text(s)
        (root/('norm_'+kind+'.cuh')).write_text((src/'norm.cuh').read_text().replace('wam_optimized_norm',norm_ns))
        manifest.append(dict(kind=kind,namespace=ns,source_SHA=SOURCE_SHA,
            generated_sha256=hashlib.sha256(s.encode()).hexdigest(),
            algorithm_sha256=hashlib.sha256((src/'gemm_pipeline.cuh').read_bytes()).hexdigest()))
    (root/'SOURCE-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return root
