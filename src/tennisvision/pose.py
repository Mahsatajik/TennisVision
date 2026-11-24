"""Court-gated MediaPipe poses; near/far are spatial slots, not biometric identities."""

import cv2
import numpy as np

from .assets import require_model

# MediaPipe body indices; omit facial details for a legible sports overlay.
CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (27, 29),
    (29, 31),
    (28, 30),
    (30, 32),
]


class PoseEstimator:
    def __init__(self, court):
        import mediapipe as mp

        self.mp = mp
        self.court = court
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(require_model("pose"))),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_poses=3,
            min_pose_detection_confidence=0.35,
            min_pose_presence_confidence=0.35,
        )
        self.engine = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self.previous = {}
        self.missed = {}

    def infer(self, frame):
        height, width = frame.shape[:2]
        net = self.court.image(4.115, 11.885)[1]
        # Overlapping crops preserve limbs at the net and enlarge the distant player.
        far_x = self.court.polygon[:2, 0]
        near_x = self.court.polygon[2:, 0]
        far_y = self.court.polygon[:2, 1].mean()
        near_y = self.court.polygon[2:, 1].mean()
        crops = [
            (
                "far",
                max(0, int(width * (far_x.min() - 0.10))),
                max(0, int(height * (far_y - 0.20))),
                min(width, int(width * (far_x.max() + 0.10))),
                min(height, int(height * (net + 0.07))),
            ),
            (
                "near",
                max(0, int(width * (near_x.min() - 0.15))),
                max(0, int(height * (net - 0.04))),
                min(width, int(width * (near_x.max() + 0.15))),
                min(height, int(height * (near_y + 0.22))),
            ),
        ]
        output = {}
        for slot, bx1, by1, bx2, by2 in crops:
            regions = []
            if slot in self.previous and self.missed.get(slot, 0) < 4:
                landmarks = self.previous[slot]["landmarks"]
                visible = np.array(
                    [landmarks[i][:2] for i in range(11, 33) if landmarks[i][2] > 0.4]
                )
                if len(visible) >= 6:
                    low, high = visible.min(axis=0), visible.max(axis=0)
                    padding = np.maximum((high - low) * 0.65, [0.06, 0.06])
                    low = np.maximum(0, low - padding)
                    high = np.minimum(1, high + padding)
                    regions.append(
                        (
                            int(low[0] * width),
                            int(low[1] * height),
                            int(high[0] * width),
                            int(high[1] * height),
                        )
                    )
            regions.append((bx1, by1, bx2, by2))
            for x1, y1, x2, y2 in regions:
                candidates = self._candidates(frame, x1, y1, x2, y2, slot)
                if candidates:
                    quality, selected = max(candidates, key=lambda item: item[0])
                    if quality > 0.35:
                        output[slot] = selected | {"quality": round(quality, 3)}
                        self.previous[slot] = output[slot]
                        self.missed[slot] = 0
                        break
            if slot not in output:
                self.missed[slot] = self.missed.get(slot, 0) + 1
        return output

    def _candidates(self, frame, x1, y1, x2, y2, slot):
        height, width = frame.shape[:2]
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return []
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        result = self.engine.detect(
            self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        )
        candidates = []
        for pose in result.pose_landmarks:
            points = [
                [
                    (p.x * (x2 - x1) + x1) / width,
                    (p.y * (y2 - y1) + y1) / height,
                    float(p.visibility),
                ]
                for p in pose
            ]
            ankles = [points[i] for i in (27, 28) if points[i][2] >= 0.3]
            if not ankles:
                continue
            foot = np.mean(ankles, axis=0)[:2].tolist()
            if not self.court.contains(*foot, margin=0.25) or self.court.side(*foot) != slot:
                continue
            quality = float(np.mean([points[i][2] for i in (11, 12, 23, 24, 25, 26, 27, 28)]))
            gx, gy = self.court.ground(*foot)
            quality -= max(0, abs(gx - 4.115) - 4.115) * 0.15
            candidates.append((quality, dict(landmarks=points, foot=foot, ground=[gx, gy])))
        return candidates

    def reset(self):
        self.previous.clear()
        self.missed.clear()

    def close(self):
        self.engine.close()
