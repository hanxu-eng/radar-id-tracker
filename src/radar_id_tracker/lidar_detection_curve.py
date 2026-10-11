"""Offline score-ranked one-to-one detection curves in the custom front ROI."""
from collections import defaultdict
import numpy as np
from .lidar_proposal_diagnostics import best_overlap


class DetectionCurve:
    def __init__(self,threshold=.25):
        if not 0<threshold<=1:raise ValueError('invalid IoU threshold')
        self.threshold=threshold;self.gt=0;self.events=[]

    def update(self,truth,proposals,ignored=()):
        """Highest score first, best unused same-class GT; ignore unmatched only."""
        self.gt+=len(truth);unused=list(truth)
        for box in sorted(proposals,key=lambda b:-b.confidence):
            overlap,index=best_overlap(box,unused)
            if overlap>=self.threshold:
                unused.pop(index);self.events.append((box.confidence,1))
            elif best_overlap(box,ignored)[0]<self.threshold:
                self.events.append((box.confidence,0))

    def result(self):
        grouped=defaultdict(lambda:[0,0])
        for score,tp in self.events:
            grouped[score][0]+=tp;grouped[score][1]+=1-tp
        tp=fp=0;points=[]
        for score,(positive,negative) in sorted(grouped.items(),reverse=True):
            tp+=positive;fp+=negative
            points.append(dict(score_threshold=score,tp=tp,fp=fp,fn=self.gt-tp,
                               precision=tp/(tp+fp),recall=tp/self.gt if self.gt else None))
        ap=None;recall96=None
        if self.gt:
            precision=np.array([r['precision'] for r in points])
            recall=np.array([r['recall'] for r in points])
            envelope=np.maximum.accumulate(precision[::-1])[::-1]
            ap=float(np.sum(np.diff(np.r_[0.,recall])*envelope))
            recall96=max((r['recall'] for r in points if r['precision']>=.96),default=0.)
        return dict(gt=self.gt,predictions=len(self.events),average_precision=ap,
                    maximum_recall_at_precision_096=recall96,points=points,
                    metric='custom ROI, score-ranked greedy same-class one-to-one, tied scores grouped, continuous interpolated AP')
