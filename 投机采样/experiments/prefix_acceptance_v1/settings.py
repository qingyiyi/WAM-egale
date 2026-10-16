"""Current research entry defaults; old baseline and sealed plans stay explicit."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROFILE = ROOT / '投机采样/configs/current_speculative_candidate.json'


def read():
    p = json.loads(PROFILE.read_text())
    if (p['action_draft_mode'] != 'verifier_k2_b2_bonus' or not p['bonus_enabled']
            or not p['shared_KV'] or p['host_optimization'] != 'O1+O2+O3'
            or p['threshold'] != 0.007839680526450805
            or p['selector'] not in ('global', 'depth_then_cost', 'prefix_then_cost')
            or not p['speed_comparison_deferred']):
        raise ValueError('authorized bonus/acceptance/trace defaults mismatch')
    if not Path(p['server_entry']).is_file():
        raise ValueError('concrete bonus research entry missing')
    if not Path(p['draft_checkpoint']).is_file():
        raise ValueError('frozen deployed draft missing')
    return p


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(read(), ensure_ascii=False, indent=2))
