"""Per-loaded-model text-only cache; proprio and observations are never cached."""
from collections import OrderedDict
import time
import torch


class PromptCache:
    def __init__(self, model, capacity=64):
        assert capacity > 0 and not model.training
        self.model = model
        self.original = model.encode_prompt
        self.capacity = capacity
        self.entries = OrderedDict()
        self.enabled = True
        self.hits = self.misses = self.disabled_calls = 0
        self.fill_seconds = []
        self.encoder_id = id(model.text_encoder)
        self.tokenizer_id = id(model.tokenizer)
        model.encode_prompt = self.encode_prompt

    def key(self, prompt):
        assert id(self.model.text_encoder) == self.encoder_id
        assert id(self.model.tokenizer) == self.tokenizer_id
        assert not self.model.training
        texts = (prompt,) if isinstance(prompt, str) else tuple(prompt)
        assert texts and all(isinstance(s, str) for s in texts)
        return texts, str(self.model.device), str(self.model.torch_dtype)

    @torch.no_grad()
    def encode_prompt(self, prompt):
        if not self.enabled:
            self.disabled_calls += 1
            return self.original(prompt)
        key = self.key(prompt)
        if key not in self.entries:
            self.misses += 1
            if self.model.device.type == 'cuda':
                torch.cuda.synchronize(self.model.device)
            start = time.perf_counter()
            context, mask = self.original(prompt)
            # Private storage cannot be modified by downstream concatenation,
            # graph inputs or any caller using an in-place operation.
            self.entries[key] = (context.detach().clone(), mask.detach().clone())
            if self.model.device.type == 'cuda':
                torch.cuda.synchronize(self.model.device)
            self.fill_seconds.append(time.perf_counter()-start)
            if len(self.entries) > self.capacity:
                self.entries.popitem(last=False)
        else:
            self.hits += 1
        self.entries.move_to_end(key)
        context, mask = self.entries[key]
        return context.clone(), mask.clone()

    def snapshot(self):
        return dict(hits=self.hits, misses=self.misses, disabled_calls=self.disabled_calls,
                    entries=len(self.entries), capacity=self.capacity,
                    stored_tensor_bytes=sum(x.numel()*x.element_size() for pair in self.entries.values() for x in pair),
                    fill_seconds=list(self.fill_seconds),
                    boundary='text encoder only; before per-call proprio append; both arms share identical cache')
