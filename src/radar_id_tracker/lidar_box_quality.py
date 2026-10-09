"""Bounded JSON tree inference for an optional learned box-quality score.

No pickle, scikit-learn, torch, labels, or frame/sequence IDs at inference.
Score estimates box quality in the training domain, not calibrated certainty.
"""
import json
from pathlib import Path
import numpy as np
from .lidar_box_features import FEATURE_NAMES,FEATURE_VERSION
from dataclasses import replace


class BoxQualityModel:
    def __init__(self,data):
        if data.get('format_version')!=1 or data.get('feature_version')!=FEATURE_VERSION or tuple(data.get('features',[]))!=FEATURE_NAMES:
            raise ValueError('unsupported box quality feature schema')
        self.data=data
        self.models=data.get('models',{})
        if set(self.models)!={'Car','Pedestrian'}:raise ValueError('Car and Pedestrian models required')
        for model in self.models.values():
            if not np.isfinite(model['intercept']):raise ValueError('invalid model intercept')
            if not 1<=len(model['trees'])<=200:raise ValueError('invalid tree count')
            for nodes in model['trees']:
                if not 1<=len(nodes)<=63:raise ValueError('invalid tree size')
                for i,node in enumerate(nodes):
                    if len(node)!=6 or not np.isfinite(node).all():raise ValueError('invalid tree node')
                    feature,threshold,left,right,leaf,value=node
                    if leaf not in (0,1):raise ValueError('invalid leaf flag')
                    if not leaf and (type(feature) is not int or not 0<=feature<len(FEATURE_NAMES) or
                            type(left) is not int or type(right) is not int or not i<left<len(nodes) or not i<right<len(nodes)):
                        raise ValueError('tree must have valid forward edges')

    @classmethod
    def load(cls,path: Path):
        if path.stat().st_size>5*1024*1024:raise ValueError('quality model exceeds size limit')
        return cls(json.loads(path.read_text()))

    def predict(self,label,features):
        x=np.asarray(features,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(FEATURE_NAMES) or not np.isfinite(x).all():raise ValueError('invalid box features')
        if label not in self.models:raise ValueError('unsupported quality label')
        model=self.models[label];scores=np.full(len(x),model['intercept'],dtype=np.float64)
        for nodes in model['trees']:
            for row in range(len(x)):
                index=0
                while not nodes[index][4]:
                    feature,threshold,left,right,leaf,value=nodes[index]
                    index=left if x[row,feature]<=threshold else right
                scores[row]+=nodes[index][5]
        positive=scores>=0;result=np.empty_like(scores)
        result[positive]=1/(1+np.exp(-scores[positive]))
        e=np.exp(scores[~positive]);result[~positive]=e/(1+e)
        return result

    def rescore(self,index,boxes):
        """Current-scan index + detections only; class labels are never changed."""
        result=list(boxes);details=[]
        for label in ('Car','Pedestrian'):
            positions=[i for i,b in enumerate(boxes) if b.label==label]
            selected=[boxes[i] for i in positions]
            scores=self.predict(label,index.extract_many(selected))
            for i,score in zip(positions,scores):
                result[i]=replace(boxes[i],confidence=float(score))
                details.append(dict(detection_index=i,detector_confidence=boxes[i].confidence,quality_score=float(score)))
        return result,sorted(details,key=lambda r:r['detection_index'])

    def validate_detector_inputs(self,checkpoint_sha256,config):
        if self.data['checkpoint_sha256']!=checkpoint_sha256 or any(
                config.get(key)!=value for key,value in self.data['inference_config'].items() if key!='voxel_backend'):
            raise ValueError('quality model detector input domain differs; explicitly provide matching inference config')
