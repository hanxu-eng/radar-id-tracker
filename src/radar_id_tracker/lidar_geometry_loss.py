"""Training-only pedestrian regression losses; runtime decoding stays unchanged."""
import torch
from torch.nn import functional as F


def geometry_values(head,targets,cls=1):
    fields=torch.cat([head[n] for n in ('reg','height','dim','rot')],dim=1)
    fields=fields.permute(0,2,3,1).reshape(len(fields),-1,8)
    indices=targets['indices'][:,cls]
    return fields.gather(1,indices[:,:,None].expand(-1,-1,8)),targets['values'][:,cls],targets['valid'][:,cls]


def pedestrian_geometry_loss(head,targets,variant,beta=.1,regression_weight=.25):
    predicted,expected,valid=geometry_values(head,targets)
    if variant=='continued_l1':
        losses=(predicted-expected).abs().sum(-1)
    elif variant=='weighted_axis_huber':
        # Offset XY and log dimensions are important for thin upright boxes.
        weights=predicted.new_tensor([2.,2.,1.,2.,2.,2.])
        location=(F.smooth_l1_loss(predicted[...,:6],expected[...,:6],reduction='none',beta=beta)*weights).sum(-1)
        rotation=predicted[...,6:8];target=expected[...,6:8]
        direct=F.smooth_l1_loss(rotation,target,reduction='none',beta=beta).sum(-1)
        reverse=F.smooth_l1_loss(rotation,-target,reduction='none',beta=beta).sum(-1)
        # yaw and yaw+pi describe the same upright box axis. No heading claim.
        axis=.5*torch.minimum(direct,reverse)
        norm=.1*(torch.linalg.vector_norm(rotation,dim=-1)-1).square()
        losses=location+axis+norm
    else:raise ValueError('unsupported pedestrian geometry loss')
    return regression_weight*(losses*valid).sum()/valid.sum().clamp(min=1)
