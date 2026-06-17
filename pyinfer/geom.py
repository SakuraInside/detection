"""Геометрия bbox (x1,y1,x2,y2)."""
from __future__ import annotations


def bbox_area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def centroid(b):
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def iou_xyxy(a, b) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = bbox_area(a) + bbox_area(b) - inter
    return inter / ua if ua > 0 else 0.0
