"""Same-query root/parent labels from existing verification, without new forwards."""
import json
import os
from pathlib import Path
import traceback
import uuid
import torch
from paths import ROOT, load

native = load('draft_train_native_teacher', ROOT / '投机采样/wam-p0-worktree/wan_va/utils/p0_state_teacher.py')


class Collector:
    def __init__(self, root, provenance):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.provenance = dict(provenance)
        self.attempted = self.saved = self.total = 0
        self.errors = []

    def begin_chunk(self):
        self.attempted = self.saved = 0

    def record(self, *, tree, target_directions, query_states, context, step_index,
               route, depth, contract):
        self.attempted += 1
        ident = uuid.uuid4().hex
        try:
            if int(context['task_index']) != self.provenance['task'] or int(context['episode_index']) != self.provenance['init']:
                raise ValueError('declared collection identity mismatch')
            expected_threshold = self.provenance['threshold'] if self.provenance.get('cpu_fixture') else 0.007839680526450805
            if contract['selector'] != 'prefix_then_cost' or contract['threshold'] != expected_threshold:
                raise ValueError('frozen selector/threshold mismatch')
            rows = native.teacher_rows(tree, target_directions, query_states,
                context, step_index, route, depth)
            for row in rows:
                row.update(split=self.provenance['split'], frame_st_id=int(contract['frame_st_id']),
                    source_state_dtype=str(tree.root_state.dtype),
                    source_target_dtype=str(target_directions.dtype))
            payload = dict(schema_version='bonus-prefix-same-query-teacher/v2', rows=rows,
                contract={**contract, 'mode': 'verifier_k2_b2_bonus', 'bonus_enabled': True,
                    'action_updates_per_chunk': 50, 'provenance': self.provenance},
                extra_teacher_forwards=0, captures_all_three_predictor_nodes=True)
            path = self.root / (str(os.getpid()) + '-' + ident + '.pt')
            temporary = path.with_suffix('.tmp')
            torch.save(payload, temporary)
            os.replace(temporary, path)
            self.saved += 1
            self.total += 1
        except Exception as exc:
            error = dict(error=repr(exc), traceback=traceback.format_exc(), context=context,
                step_index=int(step_index))
            self.errors.append(error)
            (self.root / ('ERROR-' + ident + '.json')).write_text(json.dumps(error))

    def end_chunk(self, context):
        value = dict(context=context, attempted_trees=self.attempted, saved_trees=self.saved,
            nodes_per_tree=3, complete=self.attempted == self.saved and not self.errors,
            provenance=self.provenance)
        path = self.root / ('CHUNK-' + uuid.uuid4().hex + '.json')
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def install(server, root, provenance):
    if server._p0_state_teacher is not None:
        raise ValueError('default-off original teacher must be absent')
    collector = Collector(root, provenance)
    server._p0_state_teacher = collector
    return collector
