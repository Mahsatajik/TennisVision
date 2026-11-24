"""Perspective calibration for ground-plane player locations, never airborne ball heights."""

import cv2
import numpy as np

from .models import Point


class Court:
    def __init__(self, points: list[Point]):
        self.polygon = np.array([[p.x, p.y] for p in points], np.float32)
        if not cv2.isContourConvex(self.polygon) or abs(cv2.contourArea(self.polygon)) < 0.02:
            raise ValueError("Court corners must form a convex quadrilateral in boundary order.")
        if self.polygon[:2, 1].mean() >= self.polygon[2:, 1].mean():
            raise ValueError("Mark the far baseline first, then the near baseline.")
        self.matrix = cv2.getPerspectiveTransform(
            self.polygon, np.array([[0, 0], [8.23, 0], [8.23, 23.77], [0, 23.77]], np.float32)
        )
        self.inverse = np.linalg.inv(self.matrix)

    def ground(self, x, y):
        return cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.matrix)[
            0, 0
        ].tolist()

    def side(self, x, y):
        return "near" if self.ground(x, y)[1] > 23.77 / 2 else "far"

    def image(self, x, y):
        return cv2.perspectiveTransform(np.array([[[x, y]]], np.float32), self.inverse)[
            0, 0
        ].tolist()

    def contains(self, x, y, margin=0.12):
        # Expanded bounds are an association gate, not an in/out line-call decision.
        gx, gy = self.ground(x, y)
        return -8.23 * margin <= gx <= 8.23 * (1 + margin) and -23.77 * margin <= gy <= 23.77 * (
            1 + margin
        )
