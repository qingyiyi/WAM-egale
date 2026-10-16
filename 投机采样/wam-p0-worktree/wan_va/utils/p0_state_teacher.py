"""Action-only K2 query-state supervision; opt-in, no extra teacher forwards."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import uuid
import torch


def state_digest(tensor):
    x = tensor.detach().to(device='cpu', dtype=torch.float32).contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()


def teacher_rows(tree, target_directions, query_states, context, step_index, route, depth):
    """Capture all THREE actual predictor inputs, including non-selected branches.

    Leaf queries have no corresponding predictor input at this decision and are
    deliberately excluded. Labels are target CFG, never a draft commit.
    """
    required = ('campaign_id', 'parallel_worker_id', 'episode_attempt_id', 'task_index', 'episode_index', 'rollout_chunk')
    if any(k not in context for k in required):
        raise ValueError('incomplete teacher episode/chunk identity')
    rows = []
    for node in range(3):
        state = tree.root_state if node == 0 else tree.first_states[node - 1]
        previous = tree.root_previous_direction if node == 0 else tree.first_directions[node - 1]
        offset = 0 if node == 0 else 1
        if not torch.equal(state, query_states[node]):
            raise ValueError('predictor/actual target query states differ at node ' + str(node))
        label = target_directions[node]
        if state.shape != previous.shape or label.shape != state.shape:
            raise ValueError('teacher shapes differ')
        if not all(bool(torch.isfinite(x).all()) for x in (state, previous, label)):
            raise ValueError('nonfinite teacher features/label')
        selected = node == 0 or node - 1 == route.parent
        rows.append({**context, 'step_index': int(step_index + offset), 'tree_step_index': int(step_index),
                     'node': node, 'selected_branch': selected, 'accepted_depth': int(depth),
                     'draft_edge_committed': bool((node == 0 and depth >= 1) or (node > 0 and selected and depth == 2)),
                     'state': state.detach().cpu().float().clone(),
                     'previous': previous.detach().cpu().float().clone(),
                     'target': label.detach().cpu().float().clone(),
                     'sigma': tree.sigma_path[offset], 'next_sigma': tree.sigma_path[offset + 1],
                     'timestep': tree.target_timesteps[offset], 'state_sha256': state_digest(state),
                     'target_query_state_sha256': state_digest(query_states[node]),
                     'teacher_label': 'actual_query_target_cfg_direction'})
    return rows


class StateTeacher:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.attempted = 0
        self.saved = 0

    def begin_chunk(self):
        self.attempted = 0
        self.saved = 0

    def end_chunk(self, context):
        try:
            path = self.root / ('CHUNK-' + str(os.getpid()) + '-' + uuid.uuid4().hex + '.json')
            path.write_text(json.dumps({'context': context, 'attempted_trees': self.attempted,
                                       'saved_trees': self.saved, 'nodes_per_tree': 3, 'complete': True}))
        except Exception as exc:
            print('STATE_TEACHER_INVALID_MARKER', repr(exc), flush=True)

    @classmethod
    def from_env(cls):
        root = os.getenv('WAM_P0_STATE_TEACHER_DIR', '').strip()
        return cls(root) if root else None

    def record(self, *, tree, target_directions, query_states, context, step_index, route, depth, contract):
        ident = uuid.uuid4().hex
        self.attempted += 1
        try:
            rows = teacher_rows(tree, target_directions, query_states, context, step_index, route, depth)
            payload = {'schema_version': 'k2-same-query-teacher/v1', 'rows': rows, 'contract': contract,
                       'extra_teacher_forwards': 0, 'captures_all_three_predictor_nodes': True}
            path = self.root / (str(os.getpid()) + '-' + ident + '.pt')
            tmp = path.with_suffix('.tmp')
            torch.save(payload, tmp)
            os.replace(tmp, path)
            self.saved += 1
        except Exception as exc:
            # Diagnostics cannot change the committed policy/control path. A
            # failed capture invalidates the data gate instead of fabricating labels.
            try:
                with (self.root / ('ERROR-' + str(os.getpid()) + '-' + ident + '.json')).open('w') as f:
                    json.dump({'error': repr(exc), 'context': context, 'step_index': step_index}, f)
            except Exception as write_exc:
                print('STATE_TEACHER_INVALID_CAPTURE', repr(exc), repr(write_exc), flush=True)
