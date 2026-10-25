from pathlib import Path
import json,subprocess,time,sys
P=Path(__file__).resolve().parent
PY='/root/autodl-tmp/cxz/envs/wam-sm120/bin/python'
def phase(name,cap,*command):
    subprocess.run([PY,str(P/'phase.py'),name,str(cap),*command],check=True)
summary=dict(status='RUNNING',start=time.time(),stages=[])
try:
    phase('build-L',1000,PY,str(P/'build_control.py'),'build_ext','--build-lib',str(P/'build/control'),'--build-temp',str(P/'build-temp/control'))
    phase('generate-profile',60,PY,str(P/'make_profile.py'))
    phase('build-profile',1000,PY,str(P/'build_whole_profile.py'),'build_ext','--build-lib',str(P/'build/profile'),'--build-temp',str(P/'build-temp/profile'))
    for number in (1,2):
        command=[PY,str(P/'measure.py'),'--process',str(number),'--output',str(P/f'measure-{number}'),
            '--candidate',str(next((P/'build/normal').glob('*.so'))),'--module-name','wam_sm120_ffn1_cluster_input_cuda','--label','A_SHARE',
            '--control',str(next((P/'build/control').glob('*.so'))),'--control-module','wam_sm120_ffn1_cluster_input_local_cuda',
            '--reference',str(P/'S.so'),'--reference-module-name','wam_sm120_ffn_stage_split_cuda',
            '--profile',str(next((P/'build/profile').glob('*.so'))),'--profile-module-name','wam_sm120_ffn1_cluster_input_profile_cuda']
        if number==1:command.append('--stages')
        phase(f'measure-{number}',900,*command)
        r=json.loads((P/f'measure-{number}/RESULT.json').read_text())
        summary['stages'].append(dict(process=number,status=r['status'],calls=r['calls']))
        job=json.loads((P/'JOB.json').read_text());job['calls_before_phase']=sum(s['calls'] for s in summary['stages'])
        (P/'JOB.json').write_text(json.dumps(job,ensure_ascii=False,indent=2)+'\n')
        (P/'RUN-STATUS.json').write_text(json.dumps(summary,indent=2))
    summary['status']='COMPLETE_MEASUREMENTS'
except BaseException as e:summary.update(status='FAILED_OR_PARTIAL',error=repr(e))
finally:
    summary['end']=time.time()
    (P/'RUN-STATUS.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)
sys.exit(0 if summary['status']=='COMPLETE_MEASUREMENTS' else 1)
