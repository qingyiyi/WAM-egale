"""Complete GPU KV-byte equality; only small state/RNG copied each warm call."""
import torch
from paths import EXP, load

snap = load('chunk_template_snapshot', EXP / 'fixed_load_cost/snapshot.py')


def pools(model, name='pos'):
    return [b.attn1.attn_caches[name] for b in model.blocks]


def references(model, name='pos'):
    result = []
    for c in pools(model, name):
        slots = c['mask'].nonzero(as_tuple=False).flatten()
        result.append(dict(slots=slots.clone(), k=c['k'][:, slots].clone(),
            v=c['v'][:, slots].clone(), id=c['id'].clone(), mask=c['mask'].clone(),
            is_pred=c['is_pred'].clone(), pool_shape=tuple(c['k'].shape), dtype=c['k'].dtype))
    return result


def equal(model, refs, name='pos'):
    caches = pools(model, name)
    if len(caches) != len(refs):
        return False
    tests = []
    for c, r in zip(caches, refs):
        if tuple(c['k'].shape) != r['pool_shape'] or c['k'].dtype != r['dtype']:
            return False
        for key in ('id', 'mask', 'is_pred'):
            tests.append((c[key].contiguous().view(torch.uint8) == r[key].contiguous().view(torch.uint8)).all())
        for key in ('k', 'v'):
            actual = c[key][:, r['slots']].contiguous().view(torch.uint8)
            expected = r[key].contiguous().view(torch.uint8)
            tests.append((actual == expected).all())
    return bool(torch.stack(tests).all().item())


def light(full):
    return {**full, 'caches': [{k: v for k, v in c.items() if k not in ('k_valid', 'v_valid')}
                             for c in full['caches']]}


def capture_light(server, cuda=True):
    caches = []
    for c in pools(server.transformer, server.cache_name):
        caches.append(dict(pool_shape=list(c['k'].shape), pool_dtype=str(c['k'].dtype),
            slots=snap.cpu(c['mask'].nonzero(as_tuple=False).flatten()),
            **{k: snap.cpu(c[k]) for k in ('id', 'mask', 'is_pred')}))
    return dict(schema='fixed-original-cache/v1',
        fields={k: snap.cpu(getattr(server, k)) for k in snap.FIELDS}, caches=caches,
        streaming=snap.cpu(server.streaming_vae.feat_cache),
        scheduler=snap.cpu(vars(server.scheduler)), action_scheduler=snap.cpu(vars(server.action_scheduler)),
        rng=snap.cpu(snap.rng(cuda)))
