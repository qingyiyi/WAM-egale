"""Private immutable, original-shape BF16 projections for a static diagnostic.

All admission is outside Graph replay. This deliberately does not implement a
production invalidation protocol or reuse the old prompt-cache runtime guards.
"""
from contextlib import contextmanager


class OriginalShapePack:
    def __init__(self, model, batch):
        if batch not in (2, 14):
            raise ValueError('separate real B2 or B14 packs only')
        self.model, self.batch = model, batch
        self.targets = [('text', model.condition_embedder.text_embedder)]
        self.targets += [(f'{i}.{kind}', getattr(block.attn2, 'to_' + kind))
                         for i, block in enumerate(model.blocks) for kind in ('k', 'v')]
        if len(model.blocks) != 30 or len(self.targets) != 61:
            raise ValueError('original thirty-layer, sixty-one-projection interface required')
        self.originals = {name: module.forward for name, module in self.targets}
        self.entries, self.building, self.active = {}, False, False
        self.prompt = None
        self.closed = False

    def _projection(self, name, args, kwargs):
        import torch
        if not self.active or len(args) != 1 or kwargs:
            raise RuntimeError('fixed original positional projection interface required')
        x = args[0]
        if not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[:2] != (self.batch, 512):
            raise RuntimeError('full original shape changed; no CFG collapse or batch slicing')
        if x.dtype != torch.bfloat16:
            raise RuntimeError('original BF16 projection input required')
        if self.building:
            if name in self.entries:
                raise RuntimeError('one first fill per actual projection required')
            y = self.originals[name](*args)
            if not isinstance(y, torch.Tensor) or y.dtype != torch.bfloat16:
                raise RuntimeError('original BF16 projection output required')
            with torch.inference_mode(False), torch.no_grad():
                saved = y.detach().clone()
            self.entries[name] = dict(output=saved, output_version=saved._version,
                input_shape=list(x.shape), input_dtype=str(x.dtype),
                output_shape=list(saved.shape), output_dtype=str(saved.dtype))
            if name == 'text':
                with torch.inference_mode(False), torch.no_grad():
                    self.prompt = x.detach().clone()
            return saved
        if len(self.entries) != 61 or name not in self.entries:
            raise RuntimeError('complete original first fill required before a hit')
        entry = self.entries[name]
        # Metadata assertions execute during eager/capture, never on replay.
        if list(x.shape) != entry['input_shape'] or str(x.dtype) != entry['input_dtype']:
            raise RuntimeError('admitted projection geometry changed')
        return entry['output']

    @contextmanager
    def activate(self, *, build=False):
        if self.active or self.closed or (build and self.entries):
            raise RuntimeError('one private non-nested pack generation required')
        self.active, self.building = True, build
        prior = []
        try:
            for name, module in self.targets:
                prior.append((module, 'forward' in vars(module), vars(module).get('forward')))
                def forward(*args, _name=name, **kwargs):
                    return self._projection(_name, args, kwargs)
                module.forward = forward
            yield self
            if build and len(self.entries) != 61:
                raise RuntimeError('complete real first fill did not visit all 61 projections')
        finally:
            for module, had, value in reversed(prior):
                if had:
                    module.forward = value
                else:
                    del module.forward
            self.active, self.building = False, False

    def admit(self, prompt):
        import torch
        if self.closed or self.active or len(self.entries) != 61 or self.prompt is None:
            raise RuntimeError('complete inactive private pack required for admission')
        if prompt.shape != self.prompt.shape or prompt.dtype != self.prompt.dtype:
            raise RuntimeError('actual prompt shape/dtype mismatch')
        if not torch.equal(prompt.contiguous().view(torch.uint8), self.prompt.contiguous().view(torch.uint8)):
            raise RuntimeError('all actual prompt bytes must be identical before cache reuse')
        for value in self.entries.values():
            if value['output']._version != value['output_version']:
                raise RuntimeError('owned cached projection mutated')
            if not bool(torch.isfinite(value['output']).all()):
                raise RuntimeError('nonfinite cached original projection')

    def tensors(self):
        return [self.prompt] + [entry['output'] for entry in self.entries.values()]

    def summary(self):
        return dict(batch=self.batch, original_M=self.batch * 512, saved_projections=len(self.entries),
            payload_bytes=sum(x['output'].numel() * x['output'].element_size()
                              for x in self.entries.values()),
            shapes={name: {k: v for k, v in x.items() if k not in ('output', 'output_version')}
                    for name, x in self.entries.items()},
            original_Torch_projections=True, output_owned_clone_on_first_fill_only=True,
            cache_hits_return_same_tensor=True, no_CFG_collapse=True,
            static_admission_not_executed_on_Graph_replay=True)

    def close(self):
        if self.active:
            raise RuntimeError('close after restoring all projection forwards')
        self.entries.clear()
        self.prompt = None
        self.closed = True
