"""有界运行：先编译观察器，再做真实完整层精度与两个独立进程计时。"""
from pathlib import Path
import json
import subprocess
import sys
import time

P = Path(__file__).resolve().parent
PY = '/root/autodl-tmp/cxz/envs/wam-sm120/bin/python'


def phase(name, cap, *command):
    subprocess.run([PY, str(P/'phase.py'), name, str(cap), *command], check=True)


if __name__ == '__main__':
    summary = dict(status='RUNNING', start=time.time(), stages=[])
    try:
        phase('generate-profile', 60, PY, str(P/'make_profile.py'))
        phase('build-profile', 1000, PY, str(P/'build_whole_profile.py'),
              'build_ext', '--build-lib', str(P/'build/profile'),
              '--build-temp', str(P/'build-temp/profile'))
        normal = next((P/'build/normal').glob('*.so'))
        profile = next((P/'build/profile').glob('*.so'))
        for number in (1, 2):
            output = P/f'measure-{number}'
            command = [PY, str(P/'measure.py'), '--process', str(number),
                       '--output', str(output), '--candidate', str(normal),
                       '--label', 'W', '--module-name', 'wam_sm120_ffn1_warp_cuda',
                       '--reference', str(P/'S.so'), '--reference-module-name', 'wam_sm120_ffn_stage_split_cuda',
                       '--profile', str(profile),
                       '--profile-module-name', 'wam_sm120_ffn1_warp_profile_cuda']
            if number == 1:
                command.append('--stages')
            phase(f'measure-{number}', 900, *command)
            result = json.loads((output/'RESULT.json').read_text())
            summary['stages'].append(dict(process=number, status=result['status'],
                                          calls=result['calls'], result=str(output/'RESULT.json')))
            job = json.loads((P/'JOB.json').read_text())
            job['calls_before_phase'] = sum(s['calls'] for s in summary['stages'])
            (P/'JOB.json').write_text(json.dumps(job, ensure_ascii=False, indent=2)+'\n')
            (P/'RUN-STATUS.json').write_text(json.dumps(summary, indent=2)+'\n')
        summary['status'] = 'COMPLETE_MEASUREMENTS'
    except BaseException as error:
        summary.update(status='FAILED_OR_PARTIAL', error=repr(error))
    finally:
        summary['end'] = time.time()
        (P/'RUN-STATUS.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(json.dumps(summary), flush=True)
    sys.exit(0 if summary['status'] == 'COMPLETE_MEASUREMENTS' else 1)
