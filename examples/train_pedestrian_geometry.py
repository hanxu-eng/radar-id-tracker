"""Matched-budget control/candidate tuning of only pedestrian geometry branches."""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from radar_id_tracker.centerpoint_model import CenterPointDetector
from radar_id_tracker.lidar_head_targets import load_head_manifest
from radar_id_tracker.lidar_geometry_loss import pedestrian_geometry_loss

PARAMETERS=dict(epochs=6,samples_per_sequence_per_epoch=128,batch_size=8,learning_rate=.00005,
    weight_decay=.01,regression_weight=.25,seed=17,gradient_clip=10,beta=.1,
    field_weights=[2,2,1,2,2,2],axis_weight=.5,unit_norm_weight=.1,
    classification_frozen=True,car_frozen=True,frozen_batchnorm_statistics=True,
    validation_used_for_selection=False)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir',type=Path,required=True);p.add_argument('--parent-model',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--variant',choices=('continued_l1','weighted_axis_huber'),required=True)
    p.add_argument('--experiment-protocol',type=Path,required=True);args=p.parse_args();out=args.output_dir
    if (out/'model.json').exists():raise ValueError('frozen geometry model exists')
    protocol=json.loads(args.experiment_protocol.read_text())
    if protocol['geometry_training_parameters']!=PARAMETERS or args.variant not in protocol['variants']:
        raise ValueError('geometry training differs from frozen experiment')
    parent,weights=load_head_manifest(args.parent_model);raw=(args.features_dir/'manifest.json').read_bytes();meta=json.loads(raw)
    if meta['training_sequences']!=parent['training_sequences'] or meta['validation_sequences']!=parent['validation_sequences'] or meta['inference_config']!=parent['inference_config']:
        raise ValueError('geometry training split/input differs from parent head')
    if sha(args.parent_model) not in protocol['parent_head_sha256'].values():raise ValueError('parent head not in frozen protocol')
    torch.manual_seed(17);torch.cuda.manual_seed_all(17);rng=np.random.default_rng(17);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    detector=CenterPointDetector(args.checkpoint,'cuda',.1,'numba')
    heads=nn.ModuleList([deepcopy(detector.network.pts_bbox_head.task_heads[i]) for i in (0,5)]).cuda().eval()
    heads.load_state_dict(torch.load(weights,map_location='cuda',weights_only=True),strict=True)
    initial={k:v.detach().cpu().clone() for k,v in heads.state_dict().items()}
    for p in heads.parameters():p.requires_grad_(False)
    for branch in ('reg','height','dim','rot'):
        for p in getattr(heads[1],branch).parameters():p.requires_grad_(True)
    allowed={name for name,p in heads.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad],lr=PARAMETERS['learning_rate'],weight_decay=PARAMETERS['weight_decay'])
    groups={s:[] for s in meta['training_sequences']}
    for r in meta['records']:
        for kind in ('feature','targets'):
            if sha(args.features_dir/r[kind])!=r[kind+'_sha256']:raise ValueError('geometry feature/target checksum differs')
        groups[r['sequence']].append(r)
    if any(not g for g in groups.values()):raise ValueError('empty geometry training sequence')
    out.mkdir(parents=True,exist_ok=True)
    frozen=dict(parameters=PARAMETERS,variant=args.variant,parent_model_sha256=sha(args.parent_model),
        feature_manifest_sha256=hashlib.sha256(raw).hexdigest(),experiment_protocol_sha256=sha(args.experiment_protocol),
        trainable_parameters=sorted(allowed),source_sha256={n:sha(n) for n in ('examples/train_pedestrian_geometry.py','src/radar_id_tracker/lidar_geometry_loss.py')})
    (out/'protocol.json').write_text(json.dumps(frozen,indent=2)+'\n');log=[]
    for epoch in range(PARAMETERS['epochs']):
        chosen=[g[int(i)] for g in groups.values() for i in rng.integers(len(g),size=128)];rng.shuffle(chosen);losses=[]
        for start in range(0,len(chosen),8):
            batch=chosen[start:start+8]
            x=torch.from_numpy(np.stack([np.load(args.features_dir/r['feature'],allow_pickle=False).astype(np.float32) for r in batch])).cuda()
            with np.load(args.features_dir/batch[0]['targets'],allow_pickle=False) as a:keys=a.files
            arrays=[]
            for r in batch:
                with np.load(args.features_dir/r['targets'],allow_pickle=False) as a:arrays.append({k:a[k] for k in keys})
            targets={k:torch.from_numpy(np.stack([a[k] for a in arrays])).cuda() for k in keys}
            optimizer.zero_grad(set_to_none=True)
            loss=pedestrian_geometry_loss(heads[1](x),targets,args.variant,beta=PARAMETERS['beta'],regression_weight=PARAMETERS['regression_weight'])
            if not torch.isfinite(loss):raise ValueError('nonfinite geometry loss')
            loss.backward();nn.utils.clip_grad_norm_([p for p in heads.parameters() if p.requires_grad],PARAMETERS['gradient_clip']);optimizer.step();losses.append(float(loss.detach()))
        log.append(dict(epoch=epoch+1,mean_geometry_loss=float(np.mean(losses))))
        (out/'training_log.json').write_text(json.dumps(log,indent=2)+'\n');print(json.dumps(dict(variant=args.variant,**log[-1])),flush=True)
    state={k:v.detach().cpu() for k,v in heads.state_dict().items()}
    changed=[k for k in state if not torch.equal(state[k],initial[k])]
    if not set(changed)<=allowed:raise ValueError('frozen classification, Car, velocity or BN state changed')
    audit=dict(changed_parameters=changed,allowed_parameters=sorted(allowed),all_other_tensors_exactly_equal=True,
        parent_model_sha256=frozen['parent_model_sha256'],variant=args.variant)
    (out/'frozen_tensor_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    target=out/'heads.pth';torch.save(state,target)
    model={**parent,'weights_file':target.name,'weights_sha256':sha(target),'geometry_variant':args.variant,
        'geometry_training_parameters':PARAMETERS,'parent_head_model_sha256':sha(args.parent_model),
        'geometry_protocol_sha256':sha(args.experiment_protocol),'geometry_feature_manifest_sha256':frozen['feature_manifest_sha256']}
    (out/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    print(json.dumps(dict(variant=args.variant,model_sha256=sha(out/'model.json'),frozen_tensor_parity=True)),flush=True)


if __name__=='__main__':main()
