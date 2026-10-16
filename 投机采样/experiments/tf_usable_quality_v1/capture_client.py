"""Preserve the original lean episode loop and save its initial camera pixels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from speculative_e2e_v3 import client_entry as lean

CAMERAS=('observation.images.agentview_rgb','observation.images.eye_in_hand_rgb')


def digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def install(client, out):
    out=Path(out)
    if Path(lean.__file__).resolve()!=Path(__file__).resolve().parent.parent/'speculative_e2e_v3/client_entry.py':
        raise ImportError('original lean full-episode capture source origin differs')
    lean.install(client,out)
    original_init,original_episode=client._init_env,client._run_episode
    memory={}

    def init(env, init_state):
        obs,timestep=original_init(env,init_state)
        if set(obs)!=set(CAMERAS):
            raise ValueError('actual initial observation camera inventory differs')
        images={name:np.array(obs[name],copy=True) for name in CAMERAS}
        if any(x.dtype!=np.uint8 or x.shape!=(128,128,3) for x in images.values()):
            raise ValueError('actual initial cameras are not 128x128x3 uint8 pixels')
        memory.clear();memory.update(images)
        return obs,timestep

    def episode(**kwargs):
        memory.clear()
        record=original_episode(**kwargs)
        if set(memory)!=set(CAMERAS):
            raise ValueError('both actual initial cameras must be captured before policy')
        meta=json.loads((out/'repro_metadata.json').read_text())
        if meta.get('initial_obs_hashes')!={name:digest(memory[name]) for name in CAMERAS}:
            raise ValueError('captured initial camera pixels differ from lean metadata')
        np.savez(out/'initial_observations.npz',**memory)
        return record

    client._init_env,client._run_episode=init,episode
