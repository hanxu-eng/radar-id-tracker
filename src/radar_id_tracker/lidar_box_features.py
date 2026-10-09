"""Current-scan, rotation-invariant box support features; no labels or history.

Only xyz geometry and the detector's score enter the vector. Reflectance, yaw,
world position, sequence/frame identifiers and annotation IDs are excluded.
"""
from __future__ import annotations
import numpy as np
from .lidar_tracker import BoxDetection

FEATURE_NAMES=('detector_score','range_m','relative_center_height_m',
               'length_m','width_m','height_m','log_inside_points','log_context_points',
               'extent_length','extent_width','extent_height',
               'std_length','std_width','std_height',
               'mean_length','mean_width','mean_height',
               'upper_fraction','lower_fraction','xy_occupancy','z_occupancy')
FEATURE_VERSION=1


class BoxFeatureIndex:
    """Sparse 1m xy bins accelerate box queries without truncating scan points."""
    def __init__(self,points,world_from_lidar):
        points=np.asarray(points)
        matrix=np.asarray(world_from_lidar,dtype=np.float64)
        if points.ndim!=2 or points.shape[1]!=4 or not np.isfinite(points).all():
            raise ValueError('expected finite current Nx4 scan')
        if (matrix.shape!=(4,4) or not np.isfinite(matrix).all() or
                not np.allclose(matrix[3],[0,0,0,1]) or
                not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6) or
                not np.isclose(np.linalg.det(matrix[:3,:3]),1.,atol=1e-6)):
            raise ValueError('expected rigid LiDAR-to-world transform')
        self.sensor=matrix[:3,3]
        world=points[:,:3].astype(np.float64)@matrix[:3,:3].T+self.sensor
        self.points=world
        self.keys=np.empty(0,dtype=np.int64)
        if len(world):
            cells=np.floor(world[:,:2]).astype(np.int64)
            self.minimum=cells.min(0);self.maximum=cells.max(0)
            self.width=int(self.maximum[1]-self.minimum[1]+1)
            keys=(cells[:,0]-self.minimum[0])*self.width+cells[:,1]-self.minimum[1]
            order=np.argsort(keys,kind='stable')
            self.keys=keys[order];self.points=world[order]

    def _query(self,center,half,yaw):
        if not len(self.keys):return self.points
        c,s=abs(np.cos(yaw)),abs(np.sin(yaw))
        extent=np.array([c*half[0]+s*half[1],s*half[0]+c*half[1]])
        lo=np.maximum(np.floor(center[:2]-extent).astype(np.int64),self.minimum)
        hi=np.minimum(np.floor(center[:2]+extent).astype(np.int64),self.maximum)
        if (lo>hi).any():return self.points[:0]
        rows=np.arange(lo[0],hi[0]+1,dtype=np.int64)
        left=(rows-self.minimum[0])*self.width+lo[1]-self.minimum[1]
        right=(rows-self.minimum[0])*self.width+hi[1]-self.minimum[1]
        starts=np.searchsorted(self.keys,left,'left');ends=np.searchsorted(self.keys,right,'right')
        return np.concatenate([self.points[a:b] for a,b in zip(starts,ends)])

    def extract(self,box: BoxDetection):
        if not isinstance(box,BoxDetection):raise TypeError('box must be BoxDetection')
        center=np.array([box.x,box.y,box.z],dtype=np.float64)
        half=np.array([box.length,box.width,box.height],dtype=np.float64)/2
        points=self._query(center,half*1.5,box.yaw)-center
        c,s=np.cos(box.yaw),np.sin(box.yaw)
        local=points@np.array([[c,-s,0],[s,c,0],[0,0,1]])/half
        context=(np.abs(local)<=1.5+1e-10).all(1)
        inside=(np.abs(local)<=1.+1e-10).all(1)
        p=local[inside];n=len(p)
        result=np.zeros(len(FEATURE_NAMES),dtype=np.float64)
        result[:8]=[box.confidence,np.linalg.norm((center-self.sensor)[:2]),center[2]-self.sensor[2],
                    box.length,box.width,box.height,np.log1p(n),np.log1p(int(context.sum())-n)]
        if n:
            result[8:11]=(p.max(0)-p.min(0))/2
            result[11:14]=p.std(0)/2
            result[14:17]=np.abs(p.mean(0)/2)
            result[17:19]=[(p[:,2]>0).mean(),(p[:,2]<-.5).mean()]
            xy=np.clip(np.floor((p[:,:2]+1)*2).astype(int),0,3)
            z=np.clip(np.floor((p[:,2]+1)*2).astype(int),0,3)
            result[19:21]=[len(np.unique(xy[:,0]*4+xy[:,1]))/16,len(np.unique(z))/4]
        if not np.isfinite(result).all():raise ValueError('non-finite box feature')
        return result

    def extract_many(self,boxes):
        return np.array([self.extract(b) for b in boxes],dtype=np.float64).reshape(-1,len(FEATURE_NAMES))
