"""Fixed-schedule front head fine-tuning; no validation-based model selection."""
from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from torch import nn
from radar_id_tracker.centerpoint_model import CenterPointDetector

PARAMETERS=dict(epochs=12,samples_per_sequence_per_epoch=128,batch_size=8,learning_rate=.0001,
    weight_decay=.01,regression_weight=.25,seed=17,gradient_clip=10,
    frozen_backbone=True,frozen_batchnorm_statistics=True,validation_used_for_selection=False)


def loss_for(head,targets,cls):
    logits=head['heatmap'][:,0]
    heat=targets['heatmap'][:,cls];mask=targets['heatmap_mask'][:,cls]
    p=logits.sigmoid().clamp(1e-4,1-1e-4)
    positive=heat==1
    focal=-(positive*(1-p)**2*p.log()+(~positive)*(1-heat)**4*p**2*(1-p).log())
    classification=(focal*mask).sum()/positive.sum().clamp(min=1)
    field=torch.cat([head[n] for n in ('reg','height','dim','rot')],dim=1).permute(0,2,3,1).reshape(len(p),-1,8)
    indices=targets['indices'][:,cls]
    values=field.gather(1,indices[:,:,None].expand(-1,-1,8))
    valid=targets['valid'][:,cls]
    regression=((values-targets['values'][:,cls]).abs()*valid[:,:,None]).sum()/valid.sum().clamp(min=1)
    return classification+PARAMETERS['regression_weight']*regression


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features-dir',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();out=args.output_dir
    if (out/'model.json').exists():raise ValueError('frozen head model exists; choose a new directory')
    manifest_path=args.features_dir/'manifest.json';meta=json.loads(manifest_path.read_text())
    if set(meta['training_sequences'])&set(meta['validation_sequences']):raise ValueError('training/validation overlap')
    if meta.get('training_parameters',PARAMETERS)!=PARAMETERS:raise ValueError('frozen training parameters differ')
    torch.manual_seed(17);torch.cuda.manual_seed_all(17);rng=np.random.default_rng(17);torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    detector=CenterPointDetector(args.checkpoint,'cuda',.1,'numba')
    heads=nn.ModuleList([deepcopy(detector.network.pts_bbox_head.task_heads[i]) for i in (0,5)]).cuda().eval()
    for head in heads:
        for param in head.vel.parameters():param.requires_grad_(False)
    optimizer=torch.optim.AdamW([p for p in heads.parameters() if p.requires_grad],lr=.0001,weight_decay=.01)
    groups={s:[] for s in meta['training_sequences']}
    for record in meta['records']:
        for kind in ('feature','targets'):
            path=args.features_dir/record[kind]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record[kind+'_sha256']:raise ValueError('training artifact checksum differs')
        groups[record['sequence']].append(record)
    if any(not group for group in groups.values()):raise ValueError('empty training sequence')
    out.mkdir(parents=True,exist_ok=True)
    protocol=dict(**PARAMETERS,training_sequences=meta['training_sequences'],validation_sequences=meta['validation_sequences'],
        feature_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        inference_config=meta['inference_config'],tracker_config=dict(high_confidence=.7,low_confidence=.1,maximum_tentative_tracks=128,duplicate_birth_iou=.85),
        deterministic_algorithms=True,torch_version=torch.__version__,
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ['examples/train_lidar_heads.py','src/radar_id_tracker/lidar_head_targets.py','src/radar_id_tracker/centerpoint_head_adapter.py']},
        **{k:meta[k] for k in ('split_plan_sha256','reserved_sequences','split_input_audit_sha256') if k in meta})
    # Write protocol before optimization and before opening any validation labels.
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    log=[]
    for epoch in range(12):
        chosen=[group[int(i)] for group in groups.values() for i in rng.integers(len(group),size=128)]
        rng.shuffle(chosen);losses=[]
        for start in range(0,len(chosen),8):
            batch=chosen[start:start+8]
            features=torch.from_numpy(np.stack([np.load(args.features_dir/r['feature'],allow_pickle=False).astype(np.float32) for r in batch])).cuda()
            arrays=[np.load(args.features_dir/r['targets'],allow_pickle=False) for r in batch]
            targets={key:torch.from_numpy(np.stack([a[key] for a in arrays])).cuda() for key in arrays[0].files}
            optimizer.zero_grad(set_to_none=True)
            loss=sum(loss_for(head(features),targets,cls) for cls,head in enumerate(heads))
            if not torch.isfinite(loss):raise ValueError('non-finite training loss')
            loss.backward();nn.utils.clip_grad_norm_(heads.parameters(),10);optimizer.step()
            losses.append(float(loss.detach()))
        log.append(dict(epoch=epoch+1,mean_loss=float(np.mean(losses))))
        print(json.dumps(log[-1]),flush=True)
        (out/'training_log.json').write_text(json.dumps(log,indent=2)+'\n')
    state={k:v.detach().cpu() for k,v in heads.state_dict().items()}
    weights=out/'heads.pth';torch.save(state,weights)
    model=dict(format_version=1,adapted_classes=['Car','Pedestrian'],front_only=True,
        base_checkpoint_sha256=meta['base_checkpoint_sha256'],inference_config=meta['inference_config'],
        training_sequences=meta['training_sequences'],validation_sequences=meta['validation_sequences'],
        feature_dtype='float16',weights_file=weights.name,weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
        parameters=PARAMETERS,labels_sha256=meta['labels_sha256'],feature_manifest_sha256=protocol['feature_manifest_sha256'],
        **{k:meta[k] for k in ('split_plan_sha256','reserved_sequences','split_input_audit_sha256') if k in meta})
    (out/'model.json').write_text(json.dumps(model,indent=2)+'\n')
    print(json.dumps(dict(model_sha256=hashlib.sha256((out/'model.json').read_bytes()).hexdigest(),weights_sha256=model['weights_sha256'])),flush=True)

if __name__=='__main__':main()
