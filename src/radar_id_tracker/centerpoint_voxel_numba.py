"""Optional serial CPU JIT hard voxelizer; no CUDA/compiler toolchain required.

No fast-math or parallel reordering: point and voxel encounter order are retained.
"""
import numpy as np
from numba import njit


@njit(cache=False)
def _voxelize(points,max_voxels,max_points):
    lo=np.array([-51.2,-51.2,-5.],dtype=np.float32)
    hi=np.array([51.2,51.2,3.],dtype=np.float32)
    size=np.array([.2,.2,8.],dtype=np.float32)
    lookup=np.full((512,512),-1,dtype=np.int64)
    voxels=np.zeros((max_voxels,max_points,5),dtype=np.float32)
    counts=np.zeros(max_voxels,dtype=np.int64)
    coors=np.zeros((max_voxels,4),dtype=np.int64)
    total=0
    for point in points:
        if (point[0]<lo[0] or point[1]<lo[1] or point[2]<lo[2] or
            point[0]>=hi[0] or point[1]>=hi[1] or point[2]>=hi[2]):continue
        x=int(np.floor((point[0]-lo[0])/size[0]))
        y=int(np.floor((point[1]-lo[1])/size[1]))
        z=int(np.floor((point[2]-lo[2])/size[2]))
        if x<0 or x>=512 or y<0 or y>=512 or z!=0:continue
        index=lookup[y,x]
        if index<0:
            if total>=max_voxels:continue
            index=total;lookup[y,x]=index;total+=1
            coors[index,2]=y;coors[index,3]=x
        n=counts[index]
        if n<max_points:
            for channel in range(5):voxels[index,n,channel]=point[channel]
            counts[index]+=1
    return voxels[:total],counts[:total],coors[:total]


def voxelize_numba(points,max_voxels=40000,max_points=20):
    points=np.asarray(points,dtype=np.float32)
    if points.ndim!=2 or points.shape[1]!=5 or not np.isfinite(points).all():
        raise ValueError('expected finite Nx5 xyz/intensity/time-lag')
    if max_voxels<1 or max_points<1:raise ValueError('voxel caps must be positive')
    return _voxelize(points,max_voxels,max_points)
