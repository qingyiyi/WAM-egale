"""Persistent evidence for conservative, unique-key whole-wave retry."""

import json
from pathlib import Path


def conservative_model_launches(confirmed, reservations):
    """Charge every reserved model even if Popen succeeds but PID pinning fails."""
    return max(confirmed, len(reservations))


def inspect_workers(job, assignments, *, client_admission_started):
    client_admission_started = bool(client_admission_started or
        (Path(job) / 'CLIENT-ADMISSION-INTENT.json').exists())
    observations = []
    for item in assignments[0]:
        out = Path(job) / 'campaigns' / item['arm'] / 'workers' / item['id']
        client = out / 'client'
        teacher = out / 'teacher'
        artifacts = []
        for path in (out / 'client.pid', out / 'FINISH_REQUESTED',
                     out / 'FINISH_READY.json', out / 'FINISH_FAILED.json'):
            if path.exists():
                artifacts.append(str(path))
        for folder in (client, teacher):
            if folder.exists():
                artifacts.extend(str(p) for p in folder.rglob('*') if p.is_file())
        uncertain = False
        episode = None
        ep_path = client / 'episodes.jsonl'
        if ep_path.exists():
            try:
                lines = [json.loads(line) for line in ep_path.read_text().splitlines() if line.strip()]
                if len(lines) == 1:
                    ep = lines[0]
                    episode = {key: ep.get(key) for key in
                               ('task_index', 'episode_index', 'episode_attempt_id', 'status')}
                else:
                    uncertain = True
            except (OSError, ValueError, TypeError):
                uncertain = True
        counter = out / 'counter-state.json'
        if counter.exists():
            try:
                value = json.loads(counter.read_text())
                if (value.get('target_intents', 0) or value.get('request_active')
                        or value.get('active_forwards', 0)):
                    artifacts.append(str(counter))
            except (OSError, ValueError, TypeError):
                uncertain = True
        observations.append(dict(id=item['id'], task=item['task'], init=item['init'],
                                 client_started=bool(client_admission_started),
                                 rollout_artifact_seen=bool(artifacts),
                                 artifact_paths=sorted(set(artifacts)),
                                 observed_episode_identity=episode,
                                 uncertain=uncertain))
    safe = not client_admission_started and all(
        not x['rollout_artifact_seen'] and not x['uncertain'] for x in observations)
    return dict(workers=observations, client_started=bool(client_admission_started),
                rollout_artifact_seen=any(x['rollout_artifact_seen'] for x in observations),
                uncertain=any(x['uncertain'] for x in observations),
                no_client_or_rollout_started=safe,
                observed_completed_client_episodes=sum(
                    x['observed_episode_identity'] is not None and
                    x['observed_episode_identity']['status'] == 'completed'
                    for x in observations))


