"""Interchangeable classical and temporal neural ball detectors with gated association."""

from collections import deque

import cv2
import numpy as np

from .assets import require_model


class Association:
    def __init__(self):
        self.history = deque(maxlen=2)

    def select(self, candidates, time):
        if not candidates:
            return None
        recent = self.history and time - self.history[-1][0] < 0.35
        prediction = np.array(self.history[-1][1:]) if recent else None
        if recent and len(self.history) == 2:
            t0, x0, y0 = self.history[0]
            t1, x1, y1 = self.history[1]
            prediction += np.clip((time - t1) / max(0.01, t1 - t0), 0, 3) * np.array(
                [x1 - x0, y1 - y0]
            )
        ranked = []
        for x, y, score in candidates:
            distance = float(np.linalg.norm(np.array([x, y]) - prediction)) if recent else 0
            if distance > 0.22:
                continue
            ranked.append((score - distance * 2, x, y, score))
        if not ranked:
            return None
        _, x, y, score = max(ranked)
        self.history.append((time, x, y))
        return dict(x=float(x), y=float(y), quality=round(float(score), 3), observed=True)

    def reset(self):
        self.history.clear()


class ClassicalBall:
    name = "Color + motion baseline"

    def __init__(self):
        self.previous = None
        self.association = Association()

    def observe(self, frame):
        pass

    def infer(self, frame, time):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color = cv2.inRange(hsv, np.array([20, 65, 105]), np.array([48, 255, 255]))
        if self.previous is None:
            self.previous = gray
            return None
        motion = cv2.threshold(cv2.absdiff(gray, self.previous), 18, 255, cv2.THRESH_BINARY)[1]
        self.previous = gray
        mask = cv2.bitwise_and(color, cv2.dilate(motion, np.ones((3, 3), np.uint8)))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = frame.shape[:2]
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if not (2 <= area <= max(80, width * height * 0.0002)) or max(w, h) > width * 0.035:
                continue
            compactness = area / max(1, w * h)
            if compactness >= 0.18:
                candidates.append(
                    ((x + w / 2) / width, (y + h / 2) / height, min(0.8, 0.35 + compactness * 0.45))
                )
        return self.association.select(candidates, time)

    def reset(self):
        self.previous = None
        self.association.reset()

    def close(self):
        pass


class TrackNetBall:
    name = "TrackNet V1 / three-frame model"

    def __init__(self):
        import torch

        from .tracknet import TrackNet

        self.torch = torch
        torch.set_num_threads(min(4, torch.get_num_threads()))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.network = TrackNet().to(self.device)
        weights = torch.load(require_model("tracknet"), map_location="cpu", weights_only=True)
        self.network.load_state_dict(weights)
        self.network.eval()
        self.frames = deque(maxlen=3)
        self.association = Association()

    def observe(self, frame):
        # Reference checkpoint expects current, previous, previous-previous RGB frames.
        rgb = cv2.cvtColor(cv2.resize(frame, (640, 360)), cv2.COLOR_BGR2RGB)
        self.frames.append(rgb.transpose(2, 0, 1).astype(np.float32) / 255)

    def infer(self, frame, time):
        if len(self.frames) < 3:
            return None
        tensor = self.torch.from_numpy(
            np.concatenate(list(reversed(self.frames)), axis=0)[None]
        ).to(self.device)
        with self.torch.inference_mode():
            # Argmax over intensity classes avoids a full 256-class softmax allocation.
            heat = (
                self.network(tensor).argmax(dim=1).reshape(360, 640).cpu().numpy().astype(np.uint8)
            )
        binary = (heat >= 128).astype(np.uint8)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
        candidates = []
        for i in range(1, count):
            area = stats[i, cv2.CC_STAT_AREA]
            if 1 <= area <= 200:
                x, y = centers[i]
                quality = float(heat[labels == i].max()) / 255
                candidates.append((x / 640, y / 360, quality))
        return self.association.select(candidates, time)

    def reset(self):
        self.frames.clear()
        self.association.reset()

    def close(self):
        self.network.to("cpu")


def create_detector(name):
    return TrackNetBall() if name == "tracknet" else ClassicalBall()
