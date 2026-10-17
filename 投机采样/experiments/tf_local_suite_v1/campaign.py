"""One local quality worker over the unchanged finite per-suite campaign."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tf_suite_pipeline_v1.campaign import Campaign as BaseCampaign,OWN as PIPELINE
from tf_suite_pipeline_v1.protocol import require,write,PERMISSIONS


class LocalCampaign(BaseCampaign):
    def __init__(self,job):
        super().__init__(job)
        require(self.p['suite']=='object' and self.p['max_local_models']==1
            and self.p['local_CPU_nice']==10 and os.getpriority(os.PRIO_PROCESS,0)>=10,
            'one low-priority local Object worker required')
        require(self.p['minimum_host_available_bytes']==24*2**30,
            '24GiB host available memory admission required')

    def gpu_admit(self):
        self.guard()
        if self.active:return False
        q=subprocess.run(['nvidia-smi','-i',str(self.p['gpu_index']),
            '--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=10)
        require(q.returncode==0,'actual local GPU metadata unavailable')
        fields=[v.strip() for v in q.stdout.strip().split(',')]
        require(len(fields)==3 and fields[0]==self.p['physical_GPU_UUID'],
            'registered local physical GPU differs')
        procs=subprocess.run(['nvidia-smi','-i',str(self.p['gpu_index']),
            '--query-compute-apps=pid','--format=csv,noheader,nounits'],
            capture_output=True,text=True,timeout=10)
        require(procs.returncode==0,'local visible compute-process inventory unavailable')
        available=[int(v.split()[1])*1024 for v in Path('/proc/meminfo').read_text().splitlines()
            if v.startswith('MemAvailable:')]
        require(len(available)==1,'actual host memory admission unavailable')
        return (not procs.stdout.strip() and int(fields[1])>=self.p['new_model_free_MiB']
            and available[0]>=self.p['minimum_host_available_bytes'])

    def phase_plan(self,step):
        result=super().phase_plan(step)
        write(result/'LOCAL-PLACEMENT.json',dict(schema='tf-suite-local-placement/v1',
            GPU_UUID=self.p['physical_GPU_UUID'],max_models=1,nice=10,
            CPU_profiles=self.p['CPU_profiles'],changed_only='host placement and concurrency',
            inherited_math_source='9aa66fb',formal_timing=False,permissions=PERMISSIONS))
        return result

    def child(self,argv,step,affinity,timeout,*,cleanup_report=False):
        if step=='CPU-final-report':
            require(cleanup_report,'independent terminal descriptive report only')
            argv=[self.p['va_python'],'-B',PIPELINE/'report_v2.py','--campaign',self.job,
                '--out',self.job/'reports/final-v2']
        return super().child(argv,step,affinity,timeout,cleanup_report=cleanup_report)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--job',type=Path,required=True)
    return LocalCampaign(p.parse_args().job).run()

if __name__=='__main__':raise SystemExit(main())
