"""Strict opt-in loader for one fixed MLP child2500 -> continuation3000."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import torch

EXP=Path(__file__).resolve().parent.parent
if str(EXP) not in sys.path:sys.path.append(str(EXP))
from target_trace_draft_train_v1.adapter import ActionPredictor,NORMALIZATION,model_source,sha
from target_trace_draft_finetune_v1.adapter import load_checkpoint as load_parent

SCHEMA='tf-usable-mlp-draft-continuation/v1'
FINAL_STATUS='PASS_MATCHED_MLP_CONT_500_COMPLETE'
ARCHITECTURES={'mlp96':52928}
SEED=20260923
PARENT_SHA='655e8647efa0fa44767b75b8e98a02ec6cbe84fc058b10aa15860f6cd609165e'

def require(ok,msg):
    if not ok:raise ValueError(msg)

def load_checkpoint(path,*,architecture,device,expected_channels,expected_sha256=None):
    channels=tuple(range(expected_channels)) if isinstance(expected_channels,int) else tuple(expected_channels)
    require(channels==tuple(range(7)) and architecture=='mlp96','matched MLP and ordered original7 channels required')
    path=Path(path).resolve(strict=True)
    require(not (path.parent/'FAILED.json').exists(),'failed continuation cell')
    digest=sha(path)
    require(expected_sha256 is None or digest==expected_sha256,'continuation checkpoint bytes differ')
    payload=torch.load(path,map_location='cpu',weights_only=True);m=payload.get('metadata',{});source=model_source()
    require(payload.get('schema')==SCHEMA and payload.get('architecture')==architecture
        and payload.get('specification')==source.SPECS[architecture],'continuation architecture/schema differs')
    require(m.get('condition')=='MLP_CONT' and m.get('seed')==SEED and m.get('initial_total_updates')==2500
        and m.get('continuation_updates')==500 and m.get('total_updates')==3000 and m.get('batch')==512
        and m.get('target_half')==m.get('same_query_half')==256 and m.get('learning_rate')==3e-5
        and m.get('weight_decay')==.01 and m.get('normalization_refit') is False
        and m.get('training_source')=='same_stage2_historical_target_and_original_same_query'
        and m.get('optimizer_state_reset') is True and m.get('deployed') is False,
        'fixed primary MLP_CONT continuation protocol differs')
    parent=Path(m['parent_checkpoint']).resolve(strict=True)
    require(m.get('parent_checkpoint_sha256')==PARENT_SHA and sha(parent)==PARENT_SHA,'child2500 parent bytes differ')
    parent_payload=torch.load(parent,map_location='cpu',weights_only=True)
    require(parent_payload.get('metadata',{}).get('seed')==SEED,'same-seed child2500 required')
    # The frozen parent loader verifies its stage1 lineage, model and exact FIT normalizer.
    parent_predictor=load_parent(parent,architecture=architecture,device='cpu',expected_channels=channels,expected_sha256=PARENT_SHA)
    norm=payload.get('normalization',{})
    require(set(norm)==set(NORMALIZATION),'continuation normalization inventory differs')
    for k,shape in NORMALIZATION.items():
        v=norm[k]
        require(isinstance(v,torch.Tensor) and tuple(v.shape)==shape and v.dtype==torch.float32
            and bool(torch.isfinite(v).all()) and torch.equal(v,getattr(parent_predictor,k))
            and (not k.endswith('std') or bool((v>0).all())),'normalization differs: '+k)
    model=parent_predictor.model
    state=payload.get('model_state_dict',{})
    require(set(state)==set(model.state_dict()),'continuation model inventory differs')
    for k,ref in model.state_dict().items():
        v=state[k]
        require(isinstance(v,torch.Tensor) and v.shape==ref.shape and v.dtype==torch.float32
            and bool(torch.isfinite(v).all()),'invalid continuation tensor: '+k)
    model.load_state_dict(state,strict=True)
    require(torch.device(device).type in ('cpu','cuda'),'CPU/CUDA explicit predictor only')
    return ActionPredictor(model,norm,architecture=architecture,checkpoint_path=path,checkpoint_sha256=digest,device=device)

def load_selected(training_dir,architecture,*,device,expected_channels):
    root=Path(training_dir).resolve(strict=True)
    require(not (root/'FAILED.json').exists() and not (root/'STOP_REQUESTED').exists(),'failed/stopped continuation training')
    final=json.loads((root/'FINAL-RESULT.json').read_text());selected=json.loads((root/'selected.json').read_text())
    require(final.get('status')==FINAL_STATUS and final.get('complete_runs')==1
        and final.get('full_CAL_new_adapter_bitwise_exact') is True
        and final.get('normalization_exact_parent') is True,'complete fixed primary MLP_CONT training required')
    require(selected.get('schema')=='tf-usable-mlp-selection/v1' and selected.get('selection_role')=='fixed_final_checkpoint_needs_closed_loop_DEV'
        and selected.get('robot_quality_validated') is False and selected.get('deployed') is False,'selection is not fixed exploratory final checkpoint')
    row=selected['architectures'][architecture]
    require(row.get('seed')==SEED and row.get('condition')=='MLP_CONT','fixed primary seed/condition required')
    path=Path(row['checkpoint']).resolve(strict=True)
    require(path==root/f'MLP_CONT-seed{SEED}'/'checkpoint.pt','selected path escaped canonical continuation cell')
    record=json.loads((path.parent/'RESULT.json').read_text())
    require(record.get('status')=='PASS_MLP_CONT_CELL' and record.get('checkpoint_sha256')==row.get('checkpoint_sha256')
        and record.get('full_CAL_new_adapter_bitwise_exact') is True and record.get('same_CPU_decode_batch_bitwise_exact') is True
        and record.get('normalization_exact_parent') is True and record.get('seed')==SEED
        and record.get('total_updates')==3000,'continuation cell incomplete')
    result=load_checkpoint(path,architecture=architecture,device=device,expected_channels=expected_channels,
        expected_sha256=row['checkpoint_sha256'])
    result.selection_role=selected['selection_role']
    return result

def load_candidate(checkpoint,architecture,device,expected_channels,expected_sha256=None):
    """Explicit same-seed child2500 TF/MLP or fixed MLP-cont3000; no selection."""
    path=Path(checkpoint).resolve(strict=True)
    require(architecture in ('mlp96','mlp96'),'known matched-capacity architecture required')
    require(not (path.parent/'FAILED.json').exists(),'failed candidate cell')
    payload=torch.load(path,map_location='cpu',weights_only=True)
    require(payload.get('architecture')==architecture and payload.get('metadata',{}).get('seed')==SEED,
        'candidate must be actual seed23, never selected MLP seed22')
    if payload.get('schema')=='target-trace-draft-two-stage/v1':
        return load_parent(path,architecture=architecture,device=device,
            expected_channels=expected_channels,expected_sha256=expected_sha256)
    require(payload.get('schema')==SCHEMA and architecture=='mlp96','unsupported continuation candidate')
    return load_checkpoint(path,architecture=architecture,device=device,
        expected_channels=expected_channels,expected_sha256=expected_sha256)
