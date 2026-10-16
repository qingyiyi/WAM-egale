"""Real production loader, using a disposable tiny RANDOM CPU checkpoint.

No mocking/rebinding of load_transformer/from_pretrained/configure_model. This
is import/API coverage, not the trained target or a CUDA acceptance result.
"""
import tempfile

import test_cpu as fixture


def loaded_model(dtype=None):
    t=fixture.torch
    dtype=t.float32 if dtype is None else dtype
    prototype=fixture.model(30,dtype=dtype)
    history=fixture.cache_state(prototype)
    with tempfile.TemporaryDirectory(prefix='chc-production-loader-CPU-') as tmp:
        prototype.save_pretrained(tmp,safe_serialization=True)
        # This is precisely the function bound by wan_va_server's bare import.
        m=fixture.public_server.load_transformer(tmp,torch_dtype=dtype,
                                                 torch_device='cpu',attn_mode='torch')
    m=fixture.public_server._configure_model(m,lambda x:x,dtype,t.device('cpu'),eval_mode=True)
    for block,state in zip(m.blocks,history):
        a=block.attn1
        a.init_kv_cache('pos',48,2,4,'cpu',dtype,2)
        fixture.SNAPSHOT.restore_cache(a.attn_caches['pos'],state)
    if type(m).__module__!='modules.model' or t.cuda.is_initialized():
        raise AssertionError('not production loader / CPU path')
    return m
