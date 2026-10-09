# Copyright (c) OpenMMLab. All rights reserved.
# Adapted inference architecture; licensed under Apache-2.0, see THIRD_PARTY_NOTICES.
"""Dense PyTorch port of the fixed official CenterPoint pillar02 circle-NMS model.

No MMCV, spconv, custom CUDA kernels, training code, or arbitrary checkpoint code.
This optional module is imported only by the raw-LiDAR inference command.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from .centerpoint_geometry import (TASK_CLASSES,NMS_DISTANCE_SQUARED,
                                  voxelize,circle_nms)

CHECKPOINT_SHA256 = '191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d'
CHECKPOINT_NAME = 'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus_20220811_031844-191a3822.pth'
CHECKPOINT_URL = ('https://download.openmmlab.com/mmdetection3d/v1.0.0_models/centerpoint/'
                  'centerpoint_02pillar_second_secfpn_circlenms_4x8_cyclic_20e_nus/'+CHECKPOINT_NAME)


def decorate_torch(voxels,counts,coors):
    # Use the reference float32 reduction/order on the inference device.
    # A float64 NumPy intermediate changes PFN values and can affect heatmaps.
    mean=voxels[:,:,:3].sum(dim=1,keepdim=True)/counts.to(voxels.dtype).view(-1,1,1)
    center=torch.zeros_like(voxels[:,:,:3])
    for axis,coordinate,size,offset in ((0,3,.2,-51.1),(1,2,.2,-51.1),(2,1,8.,-1.)):
        center[:,:,axis]=voxels[:,:,axis]-(coors[:,coordinate].to(voxels.dtype).unsqueeze(1)*size+offset)
    features=torch.cat([voxels,voxels[:,:,:3]-mean,center],dim=-1)
    mask=torch.arange(voxels.shape[1],device=voxels.device)[None,:]<counts[:,None]
    return features*mask[:,:,None].to(voxels.dtype)


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,3,padding=1,bias=False)
        self.bn=nn.BatchNorm2d(out_channels,eps=1e-5,momentum=.1)
        self.activate=nn.ReLU()
    def forward(self,x):
        return self.activate(self.bn(self.conv(x)))


class PFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(11,64,bias=False)
        self.norm=nn.BatchNorm1d(64,eps=1e-3,momentum=.01)
    def forward(self,x):
        x=self.linear(x)
        x=torch.relu(self.norm(x.permute(0,2,1)).permute(0,2,1))
        return x.max(dim=1).values


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        blocks=[]
        for cin,cout,n in ((64,64,3),(64,128,5),(128,256,5)):
            layers=[nn.Conv2d(cin,cout,3,stride=2,padding=1,bias=False),
                    nn.BatchNorm2d(cout,eps=1e-3,momentum=.01),nn.ReLU()]
            for _ in range(n):
                layers.extend([nn.Conv2d(cout,cout,3,padding=1,bias=False),
                               nn.BatchNorm2d(cout,eps=1e-3,momentum=.01),nn.ReLU()])
            blocks.append(nn.Sequential(*layers))
        self.blocks=nn.ModuleList(blocks)
    def forward(self,x):
        outputs=[]
        for block in self.blocks:
            x=block(x);outputs.append(x)
        return outputs


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.deblocks=nn.ModuleList([
            nn.Sequential(conv,nn.BatchNorm2d(128,eps=1e-3,momentum=.01),nn.ReLU())
            for conv in (nn.Conv2d(64,128,2,stride=2,bias=False),
                         nn.Conv2d(128,128,1,stride=1,bias=False),
                         nn.ConvTranspose2d(256,128,2,stride=2,bias=False))])
    def forward(self,features):
        return torch.cat([block(x) for block,x in zip(self.deblocks,features)],dim=1)


class TaskHead(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.names=('reg','height','dim','rot','vel','heatmap')
        for name,out in zip(self.names,(2,1,3,2,2,classes)):
            setattr(self,name,nn.Sequential(ConvModule(64,64),nn.Conv2d(64,out,3,padding=1)))
    def forward(self,x):
        return {name:getattr(self,name)(x) for name in self.names}


class CenterPointNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.pts_voxel_encoder=nn.Module()
        self.pts_voxel_encoder.pfn_layers=nn.ModuleList([PFN()])
        self.pts_backbone=Backbone()
        self.pts_neck=Neck()
        self.pts_bbox_head=nn.Module()
        self.pts_bbox_head.shared_conv=ConvModule(384,64)
        self.pts_bbox_head.task_heads=nn.ModuleList([TaskHead(len(task)) for task in TASK_CLASSES])
    def extract_features(self,features,coors):
        pillars=self.pts_voxel_encoder.pfn_layers[0](features)
        canvas=features.new_zeros((64,512*512))
        canvas[:,coors[:,2]*512+coors[:,3]]=pillars.T
        x=canvas.reshape(1,64,512,512)
        x=self.pts_neck(self.pts_backbone(x))
        x=self.pts_bbox_head.shared_conv(x)
        return x
    def forward(self,features,coors):
        x=self.extract_features(features,coors)
        return [head(x) for head in self.pts_bbox_head.task_heads]


def decode_task(head,task_id,score_threshold=.1):
    """Top-500 heatmap entries, gravity-center boxes, per-task circle NMS."""
    heat=head['heatmap'].sigmoid()
    _,classes,height,width=heat.shape
    scores,indices=torch.topk(heat.reshape(1,classes,-1),500,dim=2)
    scores,which=torch.topk(scores.reshape(1,-1),500,dim=1)
    class_ids=which//500
    indices=indices.reshape(1,-1).gather(1,which)
    ys=(indices//width).float();xs=(indices%width).float()
    def gather(name):
        value=head[name].permute(0,2,3,1).reshape(1,-1,head[name].shape[1])
        return value.gather(1,indices[:,:,None].expand(-1,-1,value.shape[2]))[0]
    reg=gather('reg');rot=gather('rot')
    boxes=torch.cat([((xs[0]+reg[:,0])*.8-51.2)[:,None],
                     ((ys[0]+reg[:,1])*.8-51.2)[:,None],
                     gather('height'),gather('dim').exp(),
                     torch.atan2(rot[:,0],rot[:,1])[:,None],gather('vel')],dim=1)
    scores=scores[0];class_ids=class_ids[0]
    valid=(scores>score_threshold)&torch.isfinite(boxes).all(1)
    valid&=(boxes[:,:3]>=boxes.new_tensor([-61.2,-61.2,-10.])).all(1)
    valid&=(boxes[:,:3]<=boxes.new_tensor([61.2,61.2,10.])).all(1)
    boxes=boxes[valid].cpu().numpy();scores=scores[valid].cpu().numpy()
    class_ids=class_ids[valid].cpu().numpy()
    keep=circle_nms(boxes[:,:2],scores,NMS_DISTANCE_SQUARED[task_id])
    return boxes[keep],scores[keep],np.array(TASK_CLASSES[task_id])[class_ids[keep]]


class CenterPointDetector:
    def __init__(self,checkpoint: Path,device='cuda',score_threshold=.1,voxel_backend='numpy',head_model=None):
        if not .1<=score_threshold<1:
            raise ValueError('score_threshold must be in [0.1,1)')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=CHECKPOINT_SHA256:
            raise ValueError('checkpoint SHA256 does not match the supported official model')
        if device=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA unavailable; explicitly select --device cpu for CPU inference')
        self.device=torch.device(device)
        self.score_threshold=score_threshold
        if voxel_backend=='numba':
            from .centerpoint_voxel_numba import voxelize_numba
            self.voxelize=voxelize_numba
        elif voxel_backend=='numpy':self.voxelize=voxelize
        else:raise ValueError('voxel backend must be numpy or numba')
        self.voxel_backend=voxel_backend
        self.network=CenterPointNetwork()
        checkpoint_data=torch.load(checkpoint,map_location='cpu',weights_only=True)
        # strict=True is essential: no silently missing or randomly initialized layers.
        self.network.load_state_dict(checkpoint_data['state_dict'],strict=True)
        self.weight_tensors=len(checkpoint_data['state_dict'])
        self.network.to(self.device).eval()
        self.head_adapter=None
        if head_model is not None:
            from .centerpoint_head_adapter import HeadAdapter
            self.head_adapter=HeadAdapter(head_model,self.network,self.device)
    def synchronize(self):
        if self.device.type=='cuda':torch.cuda.synchronize(self.device)
    @torch.inference_mode()
    def predict(self,points):
        voxels,counts,coors=self.voxelize(points)
        if not len(counts):return [],dict(voxels=0,points=len(points))
        coordinates=torch.from_numpy(coors).to(self.device)
        features=decorate_torch(torch.from_numpy(voxels).to(self.device),
                                torch.from_numpy(counts).to(self.device),coordinates)
        shared=self.network.extract_features(features,coordinates)
        heads=[head(shared) for head in self.network.pts_bbox_head.task_heads]
        results=[]
        for task_id,head in enumerate(heads):
            boxes,scores,labels=decode_task(head,task_id,self.score_threshold)
            results.extend((box,float(score),int(label)) for box,score,label in zip(boxes,scores,labels))
        if self.head_adapter is not None:
            results=self.head_adapter.replace_front(results,shared,self.score_threshold)
        return results,dict(voxels=len(counts),points=len(points),retained_voxel_points=int(counts.sum()))
