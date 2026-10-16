"""Explicitly opt-in final T5 embedding lookup; never installed automatically."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import types

import torch


SOURCE = Path(__file__).resolve().parents[2] / 'wam-p0-worktree/wan_va/wan_va_server.py'


class TextBank:
    def __init__(self, path):
        self.path = Path(path).resolve(strict=True)
        self.data = torch.load(self.path, map_location='cpu', weights_only=True)
        d = self.data
        if (d.get('schema') != 'long10-original-T5-bank/v1'
                or d.get('sequence_length') != 512 or d.get('dtype') != 'bfloat16'
                or d.get('output_shape') != [1,512,4096]
                or d.get('torch_version') != str(torch.__version__)
                or d.get('server_source_sha256') != hashlib.sha256(SOURCE.read_bytes()).hexdigest()):
            raise ValueError('text bank source/runtime/schema differs')
        prompts = d['public_task_prompts']
        if set(prompts) != {str(i) for i in range(10)}:
            raise ValueError('exact Long10 public prompt coverage required')
        self.embeddings = d['embeddings']
        if set(self.embeddings) != set(prompts.values()) | {''} or len(self.embeddings) != 11:
            raise ValueError('positive/negative text bank coverage differs')
        for value in self.embeddings.values():
            if (value.shape != (1,512,4096) or value.dtype != torch.bfloat16
                    or value.device.type != 'cpu' or not bool(torch.isfinite(value).all())):
                raise ValueError('invalid cached text tensor')
        self.hits = 0

    def get(self, prompt, *, device, dtype=torch.bfloat16, sequence_length=512):
        if isinstance(prompt, list):
            if len(prompt) != 1:
                raise ValueError('only original single-prompt reset supported')
            prompt = prompt[0]
        if (not isinstance(prompt, str) or prompt not in self.embeddings
                or dtype != torch.bfloat16 or sequence_length != 512):
            raise ValueError('text request not covered by frozen bank')
        self.hits += 1
        # Independent returned storage preserves the original encode API's
        # isolation. Never expose bank storage for in-place model mutation.
        return self.embeddings[prompt].to(device=device, dtype=dtype).clone()


def install(server, bank: TextBank):
    if os.environ.get('RESET_TEXT_BANK_ENABLED') != '1':
        raise ValueError('text bank is default off')
    cfg = server.job_config
    if (Path(cfg.wan22_pretrained_model_name_or_path).resolve()
            != Path(bank.data['checkpoint_path']).resolve()
            or cfg.libero_checkpoint_id != bank.data['target_checkpoint_id']):
        raise ValueError('loaded target and bank checkpoint differ')
    original = server.encode_prompt

    def encode(_server, prompt, negative_prompt=None, do_classifier_free_guidance=True,
               num_videos_per_prompt=1, prompt_embeds=None, negative_prompt_embeds=None,
               max_sequence_length=226, device=None, dtype=None):
        if (num_videos_per_prompt != 1 or prompt_embeds is not None
                or negative_prompt_embeds is not None):
            raise ValueError('only frozen original reset encoding is supported')
        device = device or _server.device
        dtype = dtype or _server.dtype
        positive = bank.get(prompt,device=device,dtype=dtype,
                            sequence_length=max_sequence_length)
        negative = None
        if do_classifier_free_guidance:
            negative = bank.get(negative_prompt or '',device=device,dtype=dtype,
                                sequence_length=max_sequence_length)
        return positive, negative

    server.encode_prompt = types.MethodType(encode,server)

    def restore():
        server.encode_prompt = original

    return restore
