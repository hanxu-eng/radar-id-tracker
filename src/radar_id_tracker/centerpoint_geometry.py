"""Deterministic preprocessing for the fixed CenterPoint pillar02 checkpoint.

Architecture and decode conventions follow OpenMMLab; see THIRD_PARTY_NOTICES.
No labels or future scans are used here. NumPy-only: the tracker needs no torch.
"""
from __future__ import annotations
from math import atan2, cos, sin
import numpy as np
from .lidar_tracker import BoxDetection

CLASSES = ('Car','Truck','ConstructionVehicle','Bus','Trailer','Barrier',
           'Motorcycle','Bicycle','Pedestrian','TrafficCone')
TASK_CLASSES = ((0,), (1,2), (3,4), (5,), (6,7), (8,9))
NMS_DISTANCE_SQUARED = (4.,12.,10.,1.,.85,.175)
KITTI_TO_MODEL = np.array([[0.,-1.,0.,0.],[1.,0.,0.,0.],
                          [0.,0.,1.,0.],[0.,0.,0.,1.]])
POINT_RANGE = np.array([-51.2,-51.2,-5.,51.2,51.2,3.], dtype=np.float32)


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float32)
    transform = np.asarray(transform, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] not in (4,5) or not np.isfinite(points).all():
        raise ValueError('points must be finite Nx4 or Nx5')
    if (transform.shape != (4,4) or not np.isfinite(transform).all() or
        not np.allclose(transform[3], [0,0,0,1]) or
        not np.allclose(transform[:3,:3].T @ transform[:3,:3], np.eye(3), atol=1e-6) or
        not np.isclose(np.linalg.det(transform[:3,:3]),1.,atol=1e-6)):
        raise ValueError('expected a rigid 4x4 transform')
    result = points.copy()
    result[:,:3] = points[:,:3] @ transform[:3,:3].T + transform[:3,3]
    return result


def voxelize(points, max_voxels=40000, max_points=20):
    """Hard voxelization: first-encountered voxels and points, no random shuffle.

    Preserve input scan order and truncate exactly at the checkpoint caps.
    Returns padded Nx20x5, actual counts, and batch/z/y/x coordinates.
    """
    points = np.asarray(points,dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 5 or not np.isfinite(points).all():
        raise ValueError('expected finite Nx5 xyz/intensity/time-lag')
    if max_voxels < 1 or max_points < 1:
        raise ValueError('voxel caps must be positive')
    inside = ((points[:,:3] >= POINT_RANGE[:3]) & (points[:,:3] < POINT_RANGE[3:])).all(1)
    pts = points[inside]
    coords = np.floor((pts[:,:3]-POINT_RANGE[:3])/np.array([.2,.2,8.],np.float32)).astype(np.int64)
    # Float32 rounding at the upper boundary may produce grid index 512.
    valid = (coords >= 0).all(1) & (coords < [512,512,1]).all(1)
    coords,pts = coords[valid],pts[valid]
    keys = coords[:,1]*512+coords[:,0]
    unique,first,inverse = np.unique(keys,return_index=True,return_inverse=True)
    encounter_order = np.argsort(first,kind='stable')[:max_voxels]
    rank = np.full(len(unique),-1,dtype=np.int64)
    rank[encounter_order] = np.arange(len(encounter_order))
    groups = rank[inverse]
    selected = groups >= 0
    pts,groups = pts[selected],groups[selected]
    order = np.argsort(groups,kind='stable')
    sorted_groups = groups[order]
    starts = np.r_[0,np.flatnonzero(np.diff(sorted_groups))+1] if len(order) else np.array([],dtype=int)
    counts = np.minimum(np.diff(np.r_[starts,len(order)]),max_points) if len(order) else np.array([],dtype=int)
    offsets = np.arange(len(order))-np.repeat(starts,np.diff(np.r_[starts,len(order)]))
    keep = offsets < max_points
    voxels = np.zeros((len(encounter_order),max_points,5),dtype=np.float32)
    voxels[sorted_groups[keep],offsets[keep]] = pts[order[keep]]
    xy = unique[encounter_order]
    coors = np.stack([np.zeros_like(xy),np.zeros_like(xy),xy//512,xy%512],axis=1)
    return voxels,counts.astype(np.int64),coors


def decorate(voxels, counts, coors):
    """PFN input: original 5, cluster offsets 3, voxel-center offsets 3."""
    mean = voxels[:,:,:3].sum(1,keepdims=True)/counts[:,None,None]
    center = coors[:,[3,2,1]]*np.array([.2,.2,8.])+np.array([-51.1,-51.1,-1.])
    features = np.concatenate([voxels,voxels[:,:,:3]-mean,voxels[:,:,:3]-center[:,None,:]],axis=2)
    features *= (np.arange(voxels.shape[1])[None,:] < counts[:,None])[:,:,None]
    return features.astype(np.float32)


def circle_nms(centers, scores, distance_squared, limit=83):
    """Upstream min_radius values are thresholds on squared distance."""
    order = np.argsort(scores,kind='stable')[::-1]
    kept=[]
    while len(order) and len(kept)<limit:
        i=int(order[0]); kept.append(i)
        order=order[1:]
        distance=((centers[order]-centers[i])**2).sum(1)
        order=order[distance>distance_squared]
    return kept


def box_to_world(box, score, class_id, world_from_model):
    """Decoder box uses gravity center, length/width/height, CCW yaw."""
    center = world_from_model @ np.r_[box[:3],1.]
    heading = world_from_model[:3,:3] @ [cos(box[6]),sin(box[6]),0.]
    return BoxDetection(*map(float,center[:3]),*map(float,box[3:6]),
                        atan2(heading[1],heading[0]),float(score),CLASSES[class_id])
