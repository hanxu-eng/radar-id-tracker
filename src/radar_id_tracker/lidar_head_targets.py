"""Offline NumPy targets for front-scoped Car/Pedestrian head adaptation."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .lidar_inference_config import RawLidarConfig

CELL=.8
SIZE=128
LABELS=('Car','Pedestrian')


def front_mask():
    y,x=np.mgrid[:SIZE,:SIZE]
    mx=(x+.5)*CELL-51.2;my=(y+.5)*CELL-51.2
    return (np.hypot(mx,my)<=50)&(np.abs(np.arctan2(-mx,my))<=np.deg2rad(40))


def gaussian(heatmap,x,y,radius=2):
    lo_x=max(0,x-radius);hi_x=min(SIZE,x+radius+1)
    lo_y=max(0,y-radius);hi_y=min(SIZE,y+radius+1)
    yy,xx=np.mgrid[lo_y:hi_y,lo_x:hi_x]
    sigma=(2*radius+1)/6
    values=np.exp(-((xx-x)**2+(yy-y)**2)/(2*sigma*sigma))
    heatmap[lo_y:hi_y,lo_x:hi_x]=np.maximum(heatmap[lo_y:hi_y,lo_x:hi_x],values)


def make_targets(boxes,ignored=()):
    """Boxes are in model axes, gravity-center xyz. No IDs enter targets."""
    heat=np.zeros((2,SIZE,SIZE),np.float32)
    mask=np.broadcast_to(front_mask(),heat.shape).copy()
    indices=np.zeros((2,128),np.int64);values=np.zeros((2,128,8),np.float32)
    valid=np.zeros((2,128),bool);counts=[0,0]
    for box in ignored:
        if box.label not in LABELS:continue
        center=np.array([box.x,box.y]);radius=np.hypot(box.length,box.width)/2+CELL
        y,x=np.mgrid[:SIZE,:SIZE]
        distance=((x+.5)*CELL-51.2-center[0])**2+((y+.5)*CELL-51.2-center[1])**2
        mask[LABELS.index(box.label)]&=distance>radius*radius
    for box in boxes:
        if box.label not in LABELS:continue
        cls=LABELS.index(box.label);position=(np.array([box.x,box.y])+51.2)/CELL
        x,y=np.floor(position).astype(int)
        if not (0<=x<SIZE and 0<=y<SIZE and front_mask()[y,x]):continue
        n=counts[cls]
        if n>=128:raise ValueError('training target capacity exceeded')
        gaussian(heat[cls],x,y,2)
        mask[cls,y,x]=True
        indices[cls,n]=y*SIZE+x
        values[cls,n]=[*list(position-[x,y]),box.z,*np.log([box.length,box.width,box.height]),np.sin(box.yaw),np.cos(box.yaw)]
        valid[cls,n]=True;counts[cls]+=1
    return dict(heatmap=heat,heatmap_mask=mask,indices=indices,values=values,valid=valid)


def load_head_manifest(path):
    if path.stat().st_size>1024*1024:raise ValueError('head manifest too large')
    data=json.loads(path.read_text())
    if data['format_version']!=1 or data['adapted_classes']!=list(LABELS) or data['front_only'] is not True:
        raise ValueError('unsupported head adapter schema')
    if data['feature_dtype']!='float16':raise ValueError('unsupported training feature precision')
    if data['base_checkpoint_sha256']!='191a38226d2ce990f26cc3f78c454310cc562d81d29e86d8ca8fa39c0a44215d':
        raise ValueError('head adapter base checkpoint differs')
    RawLidarConfig.from_dict(data['inference_config'])
    for key in ('training_sequences','validation_sequences'):
        split=data[key]
        if (not isinstance(split,list) or not split or len(split)>21 or
                any(not isinstance(s,str) or len(s)!=4 or not s.isascii() or not s.isdecimal() for s in split)):
            raise ValueError('head split must contain bounded KITTI sequence IDs')
        if len(set(split))!=len(split):raise ValueError('head split must be unique and disjoint')
    if set(data['training_sequences'])&set(data['validation_sequences']):raise ValueError('head split must be disjoint')
    name=data['weights_file']
    if not isinstance(name,str) or Path(name).name!=name or name in ('','.','..') or '/' in name or '\\' in name:
        raise ValueError('head weights must be a local basename')
    weights=path.parent/name
    if weights.stat().st_size>64*1024*1024 or hashlib.sha256(weights.read_bytes()).hexdigest()!=data['weights_sha256']:
        raise ValueError('head weights checksum or size differs')
    return data,weights
