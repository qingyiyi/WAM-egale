"""Unchanged full episodes with interpreter-specific shared libraries and privately extracted GLVND frontends."""
from pathlib import Path
import sys,argparse,json,traceback

OWN=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(OWN.parent))
from baai_resume_v1.runner import MigratedRunner,protocol,read,write,require,PERMISSIONS


class LibrarySeparatedRunner(MigratedRunner):
    def spawn(self,argv,log,env,affinity,role):
        env=dict(env)
        if role.startswith(('server-','client-')):
            # The Python3.10 CPU client cannot load libtorch_python compiled
            # for Python3.12. Keep original installations and separate lookup.
            interpreter=Path(self.p['client_python'] if role.startswith('client-') else self.p['va_python']).resolve()
            prefix=interpreter.parents[1]
            minor='python3.10' if role.startswith('client-') else 'python3.12'
            other=Path(self.p['va_python'] if role.startswith('client-') else self.p['client_python']).resolve().parents[1]
            env['LD_LIBRARY_PATH']=':'.join(map(str,[Path.home()/'cxz/deps/glvnd-ubuntu24-1.7.0/usr/lib/x86_64-linux-gnu',Path(self.p['root'])/'userlib',prefix/'lib',prefix/'lib'/minor/'site-packages/torch/lib',other/'lib',Path('/usr/local/cuda/compat/lib.real'),Path('/usr/local/cuda/compat/lib'),Path('/usr/local/nvidia/lib'),Path('/usr/local/nvidia/lib64')]))
        return super().spawn(argv,log,env,affinity,role)


def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True);x=a.parse_args();job=x.job.resolve(strict=True)
    try:
        p=protocol.load(job);require(not (job/'RUN-STARTED.json').exists(),'one fresh lifecycle; no partial episode retry')
        runner=LibrarySeparatedRunner(job,p);rc=runner.execute();close=read(job/'FINAL-OWNED-CLEANUP.json')
        write(job/'MIGRATION-RUN-RESULT.json',dict(status='COMPLETE_LONG500_NO_QUALITY_CERTIFICATE' if rc==0 else 'PARTIAL_OR_FAILED_OR_STAGE_PAUSED',native_returncode=rc,complete_new_cases=[cid for cid,r in runner.results.items() if r['status']=='COMPLETE_RECORDED_CASE'],all_owned_groups_empty=close['all_owned_groups_empty'],permissions=PERMISSIONS))
        return rc
    except BaseException as exc:
        f=job/'MIGRATION-RUNNER-ENTRY-FAILED.json'
        if not f.exists():write(f,dict(error=repr(exc),traceback=traceback.format_exc(),permissions=PERMISSIONS))
        raise

if __name__=='__main__':raise SystemExit(main())
