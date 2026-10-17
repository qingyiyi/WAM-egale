"""Publish local runtime assets only after a complete authenticated import."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[3]
def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:json.dump(data,f,ensure_ascii=False,indent=2);f.write('\n')

def main():
    a=argparse.ArgumentParser();a.add_argument('--job',type=Path,required=True)
    a.add_argument('--source-manifest',type=Path,required=True);a.add_argument('--metadata',type=Path,required=True)
    a.add_argument('--import-job',type=Path,required=True);a.add_argument('--suite',choices=['object','goal'],required=True)
    a.add_argument('--source-commit',required=True);a.add_argument('--wall-cap',type=int,default=3600)
    x=a.parse_args();job=x.job.resolve();job.mkdir(parents=True,exist_ok=False);start=time.time()
    cap=min(x.wall_cap,7200);deadline=start+cap
    save(job/'REGISTRATION.json',dict(goal='assemble real local suite runtime after complete import',source_commit=x.source_commit,
        suite=x.suite,source_manifest=str(x.source_manifest.resolve()),import_job=str(x.import_job.resolve()),
        CPU=1,GPU=0,origin_epoch=start,deadline_epoch=deadline,storage_cap_bytes=2**20,no_digest=True,
        cleanup='finite single CPU process; no model/remote process created'))
    try:
        while not (x.import_job/(x.suite+'-IMPORTED.json')).exists():
            if time.time()>=deadline:raise TimeoutError('local asset admission bounded wait')
            if (x.import_job/'FAILED.json').exists():raise RuntimeError('registered import failed before suite complete')
            time.sleep(5)
        src=json.loads(x.source_manifest.read_text());entry=next(e for e in src['assets'] if e['suite']==x.suite)
        folder=ROOT/'models/community-libero-20261017-v1'/x.suite;pub=folder/'publisher';runtime=folder/'runtime'
        for name in ('README.md','norm_stats.json','checkpoint_metadata.json','transformer/config.json'):
            if (pub/name).read_bytes()!=(x.metadata/x.suite/name).read_bytes():raise ValueError('small published source bytes differ')
        weight=pub/'transformer/diffusion_pytorch_model.safetensors'
        if weight.stat().st_size!=entry['size']:raise ValueError('complete imported transformer byte count differs')
        cfg=json.loads((pub/'transformer/config.json').read_text());cfg['attn_mode']='torch'
        save(runtime/'transformer/config.json',cfg)
        (runtime/'transformer/diffusion_pytorch_model.safetensors').symlink_to(weight)
        base=ROOT/'models/lingbot-va-posttrain-libero-long-modelscope-20260922'
        for part in ('vae','text_encoder','tokenizer'):(runtime/part).symlink_to(base/part,target_is_directory=True)
        info=entry['ready'];info.update(model_path=str(runtime),shared_component_source=str(base),
            import_source=str(x.import_job.resolve()),source_before_and_after_stat=entry['mtime'],
            no_standalone_hash=True,cryptographic_identity='UNKNOWN_NOT_COMPUTED',publisher_SR_not_project_result=True)
        save(folder/'ASSET-READY.json',info)
        save(job/'RESULT.json',dict(status='COMPLETE_LOCAL_PINNED_RUNTIME',suite=x.suite,asset_ready=str(folder/'ASSET-READY.json'),
            bytes=weight.stat().st_size,CPU=1,GPU=0,wall_sec=time.time()-start,no_digest=True))
        return 0
    except BaseException as e:
        save(job/'FAILED.json',dict(status='FAILED_LOCAL_ASSET_ADMISSION',error=repr(e),wall_sec=time.time()-start));raise

if __name__=='__main__':raise SystemExit(main())
