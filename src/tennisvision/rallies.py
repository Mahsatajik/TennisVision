"""Pose/ball fusion produces REVIEWABLE rally-end hypotheses, never line calls."""

import math
from collections import deque

from .models import Event


class RallyAnalyzer:
    def __init__(self, config, court):
        self.config, self.court = config, court
        self.events, self.hits = [], []
        self.reset()

    def reset(self):
        self.active_hits = []
        self.last_ball = None
        self.recent = deque(maxlen=3)
        self.crossed = False

    def update(self, time, ball, poses):
        if ball is None:
            if self.last_ball and time - self.last_ball[0] >= self.config.lost_seconds:
                self.finish()
            return
        if self.last_ball and time - self.last_ball[0] >= self.config.lost_seconds:
            self.finish()
        side = self.court.side(ball["x"], ball["y"])
        if self.active_hits and side != self.active_hits[-1]["side"]:
            self.crossed = True
        self.recent.append((time, ball, poses))
        # Require a temporal depth reversal AND nearby arm landmarks. This suppresses
        # many ordinary flight samples; court-side labels alone cannot distinguish bounces.
        if len(self.recent) == 3:
            (t0, b0, _), (t1, b1, p1), (t2, b2, _) = self.recent
            dy0, dy1 = b1["y"] - b0["y"], b2["y"] - b1["y"]
            slot = self.court.side(b1["x"], b1["y"])
            reversal = (dy0 > 0.015 and dy1 < -0.015 and slot == "near") or (
                dy0 < -0.015 and dy1 > 0.015 and slot == "far"
            )
            distances = []
            if slot in p1:
                points = p1[slot]["landmarks"]
                distances = [
                    math.hypot(b1["x"] - points[i][0], b1["y"] - points[i][1])
                    for i in (13, 14, 15, 16)
                    if points[i][2] > 0.35
                ]
            previous = self.active_hits[-1] if self.active_hits else None
            if (
                reversal
                and distances
                and min(distances) < 0.13
                and t2 - t0 < 0.85
                and (not previous or (slot != previous["side"] and t1 - previous["time"] > 0.35))
            ):
                hit = dict(
                    time=round(t1, 3), side=slot, ball=[b1["x"], b1["y"]], quality=b1["quality"]
                )
                self.active_hits.append(hit)
                self.hits.append(hit)
                self.crossed = side != slot
        self.last_ball = (time, ball)

    def finish(self):
        # A missing ball can be occlusion, a miss, a fault, or a cut. Never call it a confirmed point.
        if len(self.active_hits) >= 2 and self.last_ball:
            last = self.active_hits[-1]
            winner = None
            reason = "Tracking stopped after an exchange. Winner is unresolved."
            quality = 0.2
            if self.crossed:
                winner = (
                    self.config.near_player
                    if last["side"] == "near"
                    else 1 - self.config.near_player
                )
                reason = "Possible unreturned shot: last contact crossed the net, then ball tracking was lost. Review the rally; faults and out balls are not resolved."
                quality = 0.45
            self.events.append(
                Event(
                    id=f"rally-{len(self.events) + 1}",
                    time=round(self.last_ball[0] + 0.15, 3),
                    start=self.active_hits[0]["time"],
                    hits=len(self.active_hits),
                    winner=winner,
                    reason=reason,
                    evidence_score=quality,
                )
            )
        self.reset()

    def cut(self):
        # End-of-file and camera cuts do not prove that a point was completed.
        self.reset()
