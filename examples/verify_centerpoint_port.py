"""Compare port tensors with selected, hash-pinned OpenMMLab reference classes.

Optional GPU audit; framework factories are replaced by plain torch equivalents.
This checks numerical architecture/decode parity, not full-framework benchmark
parity. Reference source is separately downloaded and SHA-checked before execution.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Dict,List,Optional,Sequence,Tuple
import warnings
import numpy as np
import torch
from torch import Tensor,nn
from torch.nn import functional as F
from radar_id_tracker.centerpoint_geometry import (voxelize,decorate,KITTI_TO_MODEL,
    transform_points,TASK_CLASSES,circle_nms,NMS_DISTANCE_SQUARED)
from radar_id_tracker.centerpoint_model import CenterPointDetector,decode_task,decorate_torch
from radar_id_tracker.pointcloud_review import read_scan

HASHES={
 'mmdet3d/models/voxel_encoders/utils.py':'bcd942552a941f6b2c4e92421f6296304f69700837a09bf922a3475e66702e4f',
 'mmdet3d/models/voxel_encoders/pillar_encoder.py':'c06cc44b015cb99fd6aceb7b662bafc5215cb92199c1edee897044d733a816c5',
 'mmdet3d/models/backbones/second.py':'8ea4f70db9db4df6ca31bd9ecb43f8dcec7de5a8103193bb22427213b5d0a9be',
 'mmdet3d/models/necks/second_fpn.py':'62bdd452a90fec2c669f9b9bc21837f037391c6e6b8a59c9d6cf2dd04500f83e',
 'mmdet3d/models/dense_heads/centerpoint_head.py':'893ce91db96e38401e9142ecc8af549258c5c32e84c645aa8c9910877e3bbe37',
 'mmdet3d/models/task_modules/coders/centerpoint_bbox_coders.py':'3211b1c6f1a40f7eab60bfc6a03570eca72253b83aa67271205252664a09667a'}


class BaseModule(nn.Module):
    def __init__(self,init_cfg=None):super().__init__()


def build_norm_layer(config,channels):
    config=dict(config);kind=config.pop('type')
    return 'bn',(nn.BatchNorm1d if kind=='BN1d' else nn.BatchNorm2d)(channels,**config)


def build_conv_layer(config,*args,**kwargs):
    config=dict(config);config.pop('type');config.update(kwargs)
    return nn.Conv2d(*args,**config)


def build_upsample_layer(config,**kwargs):
    config=dict(config);config.pop('type');config.update(kwargs)
    return nn.ConvTranspose2d(**config)


class ConvModule(nn.Module):
    def __init__(self,cin,cout,kernel_size,stride=1,padding=0,bias='auto',conv_cfg=None,norm_cfg=None):
        super().__init__()
        self.conv=nn.Conv2d(cin,cout,kernel_size,stride=stride,padding=padding,
                            bias=(norm_cfg is None) if bias=='auto' else bias)
        self.bn=build_norm_layer(norm_cfg,cout)[1] if norm_cfg else nn.Identity()
    def forward(self,x):return F.relu(self.bn(self.conv(x)))


def selected_definitions(root,path,names,namespace):
    data=(root/path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=HASHES[path]:raise ValueError('reference SHA mismatch: '+path)
    module=ast.parse(data.decode())
    selected=[]
    for node in module.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names:
            node.decorator_list=[];selected.append(node)
    if len(selected)!=len(names):raise ValueError('reference definitions missing')
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(root/path),'exec'),namespace)


def reference_voxelize(points,max_voxels=40000,max_points=20):
    voxels=[];counts=[];coords=[];lookup={}
    lo=np.array([-51.2,-51.2,-5],np.float32);hi=np.array([51.2,51.2,3],np.float32)
    size=np.array([.2,.2,8],np.float32)
    for point in points:
        if ((point[:3]<lo)|(point[:3]>=hi)).any():continue
        xyz=np.floor((point[:3]-lo)/size).astype(int)
        if (xyz>=[512,512,1]).any():continue
        key=tuple(xyz)
        if key not in lookup:
            if len(voxels)>=max_voxels:continue
            lookup[key]=len(voxels);voxels.append(np.zeros((max_points,5),np.float32))
            counts.append(0);coords.append([0,xyz[2],xyz[1],xyz[0]])
        index=lookup[key]
        if counts[index]<max_points:
            voxels[index][counts[index]]=point;counts[index]+=1
    return np.array(voxels),np.array(counts),np.array(coords)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference-root',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--scan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--numba',action='store_true',help='also require exact CPU JIT voxelization parity')
    args=p.parse_args();torch.set_num_threads(4)
    namespace=dict(torch=torch,nn=nn,F=F,np=np,Tensor=Tensor,Optional=Optional,
        Tuple=Tuple,Sequence=Sequence,Dict=Dict,List=List,warnings=warnings,
        BaseModule=BaseModule,BaseBBoxCoder=object,ConfigType=dict,OptMultiConfig=Optional[dict],
        build_norm_layer=build_norm_layer,build_conv_layer=build_conv_layer,
        build_upsample_layer=build_upsample_layer,ConvModule=ConvModule)
    for path,names in (
        ('mmdet3d/models/voxel_encoders/utils.py',['PFNLayer','get_paddings_indicator']),
        ('mmdet3d/models/voxel_encoders/pillar_encoder.py',['PillarFeatureNet']),
        ('mmdet3d/models/backbones/second.py',['SECOND']),
        ('mmdet3d/models/necks/second_fpn.py',['SECONDFPN']),
        ('mmdet3d/models/dense_heads/centerpoint_head.py',['SeparateHead']),
        ('mmdet3d/models/task_modules/coders/centerpoint_bbox_coders.py',['CenterPointBBoxCoder'])):
        selected_definitions(args.reference_root,path,names,namespace)
    detector=CenterPointDetector(args.checkpoint)
    port=detector.network
    points=transform_points(read_scan(args.scan),KITTI_TO_MODEL);points[:,3]*=255
    points=np.column_stack([points,np.zeros(len(points),np.float32)])
    audit={}
    for pv,pc,pr in ((40000,20,len(points)),(10,2,20000)):
        v,n,c=voxelize(points[:pr],pv,pc)
        vr,nr,cr=reference_voxelize(points[:pr],pv,pc)
        for actual,expected in zip((v,n,c),(vr,nr,cr)):np.testing.assert_array_equal(actual,expected)
        if args.numba:
            from radar_id_tracker.centerpoint_voxel_numba import voxelize_numba
            jv,jn,jc=voxelize_numba(points[:pr],pv,pc)
            for actual,expected in zip((jv,jn,jc),(vr,nr,cr)):np.testing.assert_array_equal(actual,expected)
    audit['voxelization_exact']=True
    audit['numba_voxelization_exact']=args.numba
    v,n,c=voxelize(points)
    device=detector.device
    v=torch.from_numpy(v).to(device);n=torch.from_numpy(n).to(device);c=torch.from_numpy(c).to(device)
    ref_pfn=namespace['PillarFeatureNet'](in_channels=5,feat_channels=[64],voxel_size=(.2,.2,8),
        point_cloud_range=(-51.2,-51.2,-5,51.2,51.2,3),legacy=False).to(device).eval()
    ref_pfn.load_state_dict(port.pts_voxel_encoder.state_dict(),strict=True)
    ref_backbone=namespace['SECOND'](in_channels=64,out_channels=[64,128,256]).to(device).eval()
    ref_backbone.load_state_dict(port.pts_backbone.state_dict(),strict=True)
    ref_neck=namespace['SECONDFPN'](in_channels=[64,128,256],out_channels=[128,128,128],
        upsample_strides=[.5,1,2],use_conv_for_no_stride=True).to(device).eval()
    ref_neck.load_state_dict(port.pts_neck.state_dict(),strict=True)
    ref_shared=ConvModule(384,64,3,padding=1,norm_cfg={'type':'BN2d'}).to(device).eval()
    ref_shared.load_state_dict(port.pts_bbox_head.shared_conv.state_dict(),strict=True)
    heads=[]
    for task,port_head in zip(TASK_CLASSES,port.pts_bbox_head.task_heads):
        heads_cfg=dict(reg=(2,2),height=(1,2),dim=(3,2),rot=(2,2),vel=(2,2),heatmap=(len(task),2))
        ref_head=namespace['SeparateHead'](64,heads_cfg,final_kernel=3).to(device).eval()
        ref_head.load_state_dict(port_head.state_dict(),strict=True);heads.append(ref_head)
    def compare(name,a,b,tolerance=2e-4):
        error=float((a-b).abs().max());audit[name]=error
        torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
    with torch.inference_mode():
        features=decorate_torch(v,n,c)
        actual=port.pts_voxel_encoder.pfn_layers[0](features)
        expected=ref_pfn(v,n,c)
        compare('pfn_max_abs_error',actual,expected)
        canvas=features.new_zeros((64,512*512));canvas[:,c[:,2]*512+c[:,3]]=expected.T
        ref_blocks=ref_backbone(canvas.reshape(1,64,512,512))
        port_canvas=features.new_zeros((64,512*512));port_canvas[:,c[:,2]*512+c[:,3]]=actual.T
        port_blocks=port.pts_backbone(port_canvas.reshape(1,64,512,512))
        for i,(a,b) in enumerate(zip(port_blocks,ref_blocks)):compare(f'backbone_{i}_max_abs_error',a,b)
        expected=ref_neck(ref_blocks)[0];actual=port.pts_neck(port_blocks)
        compare('neck_max_abs_error',actual,expected)
        ref_x=ref_shared(expected);port_x=port.pts_bbox_head.shared_conv(actual)
        compare('shared_max_abs_error',port_x,ref_x)
        for task_id,(ref_head,port_head) in enumerate(zip(heads,port.pts_bbox_head.task_heads)):
            ref_outputs=ref_head(ref_x);port_outputs=port_head(port_x)
            for name in ref_outputs:compare(f'task_{task_id}_{name}_max_abs_error',port_outputs[name],ref_outputs[name],5e-4)
            # Decode the same head tensor through upstream coder and local coder.
            coder=namespace['CenterPointBBoxCoder']([-51.2,-51.2],4,[.2,.2],
                post_center_range=[-61.2,-61.2,-10,61.2,61.2,10],max_num=500,score_threshold=.1)
            h=port_outputs
            reference=coder.decode(h['heatmap'].sigmoid(),h['rot'][:,0:1],h['rot'][:,1:2],
                h['height'],h['dim'].exp(),h['vel'],reg=h['reg'])[0]
            centers=reference['bboxes'][:,:2].cpu().numpy();scores=reference['scores'].cpu().numpy()
            keep=circle_nms(centers,scores,NMS_DISTANCE_SQUARED[task_id])
            b,s,l=decode_task(h,task_id)
            np.testing.assert_allclose(b,reference['bboxes'][keep].cpu().numpy(),atol=1e-5)
            np.testing.assert_array_equal(s,reference['scores'][keep].cpu().numpy())
            np.testing.assert_array_equal(l,np.array(TASK_CLASSES[task_id])[reference['labels'][keep].cpu().numpy().astype(int)])
    audit.update(passed=True,decode_all_tasks_exact_or_1e_5=True,checkpoint_tensors=detector.weight_tensors,
        scan_sha256=hashlib.sha256(args.scan.read_bytes()).hexdigest(),
        limitation='plain-torch factory shims; not comparison against installed full MMCV CUDA runtime')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
