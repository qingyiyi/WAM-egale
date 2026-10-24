"""Resource relocation and explicitly new-host initial pairing; original client retained."""
from pathlib import Path
import difflib,json,os
P=Path('/root/autodl-tmp/cxz/spec-speed-20261025')
original=(P/'bundle/robotwin/eval/client.py').read_text();new=original
def replace(old,after):
    global new
    assert new.count(old)==1,old
    new=new.replace(old,after)
replace('R = Path("/root/cxz/robotwin")',f'R = Path("{P}")')
replace('J.mkdir(exist_ok=False)','J.mkdir(parents=True,exist_ok=False)')
replace('L = R / "source/lingbot-va-pinned"','L = R / "bundle/robotwin"')
replace('S = R / "source/RoboTwin-pinned"','S = R / "sim/RoboTwin"')
replace('for p in (L, S):','for p in (L, L/"wan_va", S):')
replace('request["_audit_case"] = case["episode_id"]','request["_benchmark"] = dict(case=case["task"]+"-valid049", arm=case["arm"])')
replace('            start = time.time()','''            if route == "reset":
                state.setdefault("episode_core_start_monotonic",time.perf_counter())
            start = time.time()''')
replace('        env.set_instruction = instruction','''        original_close = env.close
        def close(*args,**kwargs):
            if "episode_core_start_monotonic" in state and "warm_episode_seconds" not in state:
                state["warm_episode_seconds"] = time.perf_counter()-state["episode_core_start_monotonic"]
                state["warm_time_boundary"] = "first reset RPC entry to first subsequent env.close entry; setup/import/initial scene excluded"
                save()
            return original_close(*args,**kwargs)
        env.close = close
        env.set_instruction = instruction''')
needle='            state["initial_checks"] = checks'
insert='''            state["historical_A100_initial_checks"] = checks
            reference = R / "results/robotwin/initial-references" / (case["task"]+"-valid049.pt")
            reference.parent.mkdir(parents=True,exist_ok=True)
            actual = {"physical": physical(env), "observation": used_observation(obs), "RNG": rng}
            if reference.exists():
                old = torch.load(reference,map_location="cpu",weights_only=False)
                checks = {k: equal_tree(actual[k],old[k]) for k in actual}
            else:
                torch.save(actual,reference)
                checks = {k: True for k in actual}
            state["initial_checks_scope"] = "Same5090 first arm; historical A100 camera equality is separately observed and not inherited"
            state["initial_checks"] = checks'''
replace(needle,insert)
(P/'client_robotwin.py').write_text(new)
(P/'results/ROBOTWIN-CLIENT-RELOCATION.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='copied-pinned-complete-client',tofile='new5090-complete-client')))
cpus=sorted(os.sched_getaffinity(0))
(P/'WORKER.json').write_text(json.dumps(dict(port=19601,CPU=dict(client=cpus[2:4],server=cpus[:2])),indent=2)+'\n')
(P/'config').mkdir(exist_ok=True)
(P/'config/nvidia_icd.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path='/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0',api_version='1.3.0')),indent=2)+'\n')
(P/'config/egl_nvidia.json').write_text(json.dumps(dict(file_format_version='1.0.0',ICD=dict(library_path='/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0')),indent=2)+'\n')
