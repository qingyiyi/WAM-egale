"""CPU-only fixture: frozen AB/BA plan, phase partition, paired statistics."""
from pathlib import Path
import tempfile
import time
import types

import analyze
import client_entry
import speed_contract as contract
import prepare


def test_schedule_and_plan():
    assert [row["arm"] for row in contract.schedule(["0:25", "0:26", "0:27", "0:28"])] == [
        "T", "A_OFF", "A_OFF", "T", "T", "A_OFF", "A_OFF", "T"]
    job = contract.RUNTIME / (contract.PREFIX + "cpu-fixture-does-not-execute")
    if job.exists():
        raise AssertionError("fixture job path already exists")
    plan = prepare.build(job, ["0:25", "0:26", "0:27", "0:28"], "single_task", 19481, 29481)
    assert plan["formal_speed_local_only"] and len(plan["schedule"]) == 8
    assert plan["mode"] == "verifier_k2_b2_bonus"


def test_phase_partition():
    class Env:
        def step(self, _action):
            time.sleep(.0001)
            return None

        def close(self):
            return None

    client = types.SimpleNamespace()
    client._construct_env = lambda cls, args: cls()

    def init(env, _init_state):
        for _ in range(5):
            env.step(None)
        return {}, 5

    client._init_env = init

    def episode(**_kwargs):
        outer = time.monotonic()
        env = client._construct_env(Env, {})
        start = time.monotonic()
        client._init_env(env, None)
        for _ in range(4):
            env.step(None)
        time.sleep(.0002)
        elapsed = time.monotonic() - start
        env.close()
        return dict(env_steps=4, elapsed_sec=elapsed,
                    reset_infer_wall_sec=0., policy_infer_wall_sec=0.,
                    cache_infer_wall_sec=0., outer_episode_wall_sec=time.monotonic()-outer)

    client._run_episode = episode

    class NoopLean:
        @staticmethod
        def install(_client, _out):
            return None

    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp)
        client_entry.install(client, out, lean=NoopLean())
        row = client._run_episode()
        phase = contract.read(out / "PHASE-TIMING.json")
        assert row["formal_speed_phases"] == phase
        assert phase["simulator_step_stabilization_calls"] == 5
        assert phase["simulator_step_policy_calls"] == 4
        assert abs(phase["disjoint_phase_sum_sec"]-row["elapsed_sec"]) < .005
        assert phase["local_reset_stabilization_sec"] >= phase["simulator_step_stabilization_sec"]


def test_statistics():
    rows = [
        dict(task=0, T_success=True, A_OFF_success=True, T_elapsed_sec=100.,
             A_OFF_elapsed_sec=80., episode_speedup=1.25, T_env_steps=100, A_OFF_env_steps=98),
        dict(task=1, T_success=True, A_OFF_success=False, T_elapsed_sec=120.,
             A_OFF_elapsed_sec=130., episode_speedup=120/130, T_env_steps=120, A_OFF_env_steps=800),
    ]
    result = analyze.summarize(rows)
    assert result["both_success_pairs"] == 1
    assert result["all_episode_total_speedup"] == 220/210
    assert result["both_success_total_speedup"] == 1.25
    assert result["A_OFF_successes"] == 1


if __name__ == "__main__":
    test_schedule_and_plan()
    test_phase_partition()
    test_statistics()
    print("PASS_CPU_FORMAL_E2E_SPEED_FIXTURE")
