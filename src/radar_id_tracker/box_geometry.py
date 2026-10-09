"""Oriented, upright 3D box overlap without a graphics or geometry dependency."""
from __future__ import annotations
from math import cos, sin
import numpy as np


def corners_xy(box) -> np.ndarray:
    corners = np.array([[1,1],[-1,1],[-1,-1],[1,-1]],dtype=float)
    corners *= [box.length/2,box.width/2]
    rotation = np.array([[cos(box.yaw),-sin(box.yaw)],[sin(box.yaw),cos(box.yaw)]])
    return corners @ rotation.T + [box.x,box.y]


def _cross(a, b) -> float:
    return float(a[0]*b[1]-a[1]*b[0])


def _intersection(subject: np.ndarray, clip: np.ndarray) -> list:
    points = list(subject)
    for start,end in zip(clip,np.roll(clip,-1,axis=0)):
        previous = points
        points = []
        if not previous:
            break
        a = previous[-1]
        da = _cross(end-start,a-start)
        for b in previous:
            db = _cross(end-start,b-start)
            inside_a,inside_b = da >= -1e-9, db >= -1e-9
            if inside_a != inside_b:
                points.append(a+(b-a)*da/(da-db))
            if inside_b:
                points.append(b)
            a,da = b,db
    return points


def iou3d(a, b) -> float:
    if abs(a.x-b.x) > (a.length+a.width+b.length+b.width)/2:
        return 0.
    height = max(0.,min(a.z+a.height/2,b.z+b.height/2)-max(a.z-a.height/2,b.z-b.height/2))
    if height == 0:
        return 0.
    polygon = _intersection(corners_xy(a),corners_xy(b))
    if len(polygon) < 3:
        return 0.
    area = abs(sum(_cross(p,q) for p,q in zip(polygon,polygon[1:]+polygon[:1])))/2
    intersection = area*height
    union = a.length*a.width*a.height + b.length*b.width*b.height - intersection
    return min(1.,max(0.,intersection/union)) if union > 0 else 0.
