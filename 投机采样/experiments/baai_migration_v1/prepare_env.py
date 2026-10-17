"""Relocate only copied environment paths inside the user's ~/cxz workspace."""
from pathlib import Path
import json,os,subprocess,time


def main():
    os.sched_setaffinity(0,{43})
    base=(Path.home()/'cxz').resolve(strict=True)
    job=base/'jobs/baai-env-relocation-20261018-r1';job.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        with (job/(name+'.json')).open('x') as f:json.dump(value,f,indent=2,ensure_ascii=False)
    write('REGISTRATION',dict(goal='Relocate complete matching copied environments; no model/GPU test',CPU=1,GPU=0,wall_cap_sec=600,storage_cap_bytes=2**20,started_epoch=time.time(),no_SHA=True))
    try:
        direct=base/'jobs/baai-direct-migration-20261017-r1'
        done=json.loads((direct/'envs-RESULT.json').read_text());assert done['status']=='STATIC_SOURCE_AND_EXTRACTION_EXIT0'
        oldjob=base/'jobs/baai-migration-20261017-r2'
        for kind in ('env-va','env-libero'):
            receipt=json.loads((oldjob/(kind+'-CLEANUP.json')).read_text());assert receipt['owned_extractor_reaped']
        assert not (base/'envs-raw-partial-r2').exists()
        (base/'envs').rename(base/'envs-raw-partial-r2')
        (base/'direct-r1/WAM-egale/envs').rename(base/'envs')
        project=base/'WAM-egale'
        bindings=[('/root/autodl-tmp/wam-tf-pf2-20261017/envs',str(base/'envs')),('/rjs/cxz/miniconda3/envs',str(base/'envs')),('/root/autodl-tmp/wam-tf-pf2-20261017',str(project)),('/rjs/cxz/论文探索',str(project))]
        changes=[]
        for prefix in (base/'envs/va',base/'envs/libero-va'):
            for d,dirs,files in os.walk(prefix,followlinks=False):
                for name in files+dirs:
                    p=Path(d)/name
                    if p.is_symlink():
                        before=os.readlink(p);after=before
                        for old,new in bindings:after=after.replace(old,new)
                        if after!=before:
                            assert Path(after).is_relative_to(base)
                            relative=os.path.relpath(after,p.parent);p.unlink();p.symlink_to(relative);changes.append(dict(path=str(p),kind='symlink',before=before,after=relative))
                    elif p.is_file() and (p.suffix in ('.pth','.egg-link') or (name.startswith('__editable__') and name.endswith('.py'))):
                        before=p.read_text();after=before
                        for old,new in bindings:after=after.replace(old,new)
                        if after!=before:
                            p.write_text(after);changes.append(dict(path=str(p),kind='editable_configuration',before=before,after=after))
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONNOUSERSITE='1',CUDA_VISIBLE_DEVICES='',HOME=str(Path.home()))
        for key,suffix in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','XDG_CONFIG_HOME':'config','MPLCONFIGDIR':'cache/matplotlib'}.items():
            value=base/suffix;value.mkdir(parents=True,exist_ok=True);env[key]=str(value)
        versions={}
        for label in ('va','libero-va'):
            prefix=base/'envs'/label
            env['LD_LIBRARY_PATH']=str(prefix/'lib')
            code="import json,sys,importlib.metadata as m;names=['torch','numpy','transformers','diffusers','mujoco','robosuite'];print(json.dumps({'python':sys.version,'executable':sys.executable,'packages':{n:m.version(n) for n in names if m.packages_distributions().get(n.replace('-','_'))}}))"
            r=subprocess.run([str(prefix/'bin/python'),'-B','-c',code],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=90)
            versions[label]=dict(returncode=r.returncode,output=r.stdout);assert r.returncode==0
        write('RESULT',dict(status='COPIED_ENVIRONMENTS_PATH_RELOCATED_AND_METADATA_READ',changes=changes,versions=versions,no_model_import_or_GPU_test=True,no_SHA=True,old_partial_copies_retained=True))
    except BaseException as exc:
        write('FAILED',dict(error=repr(exc),no_retry=True));raise
    finally:write('CLEANUP',dict(status='PREPARATION_PROCESS_EXITS_NO_BACKGROUND_CHILDREN',GPU_used=False,source_deleted=False))

if __name__=='__main__':main()
