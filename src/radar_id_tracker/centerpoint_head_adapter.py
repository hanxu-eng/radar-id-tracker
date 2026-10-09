"""Explicit learned front head; base predictions outside the trained ROI survive."""
from copy import deepcopy
import numpy as np
import torch
from torch import nn
from .lidar_head_targets import load_head_manifest


def in_model_front(box):
    return np.hypot(box[0],box[1])<=50 and abs(np.arctan2(-box[0],box[1]))<=np.deg2rad(40)


class HeadAdapter:
    def __init__(self,manifest,network,device):
        self.metadata,weights=load_head_manifest(manifest)
        self.heads=nn.ModuleList([deepcopy(network.pts_bbox_head.task_heads[i]) for i in (0,5)])
        state=torch.load(weights,map_location='cpu',weights_only=True)
        if any(not torch.isfinite(v).all() for v in state.values()):raise ValueError('non-finite head tensors')
        self.heads.load_state_dict(state,strict=True)
        self.heads.to(device).eval()

    def replace_front(self,original,shared,threshold):
        from .centerpoint_model import decode_task
        result=[r for r in original if r[2] not in (0,8) or not in_model_front(r[0])]
        # Match the frozen float16 training feature cache, with float32 heads.
        x=shared.to(torch.float16).to(torch.float32)
        for index,task in enumerate((0,5)):
            head=self.heads[index](x)
            if task==5:
                # TrafficCone remains from the original head; cannot steal top-K.
                head['heatmap'][:,1]=-100
            boxes,scores,labels=decode_task(head,task,threshold)
            result.extend((box,float(score),int(label)) for box,score,label in zip(boxes,scores,labels)
                          if in_model_front(box))
        return result
