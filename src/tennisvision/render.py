"""Computer-vision overlays and a tennis scoreboard driven by an event timeline."""

from pathlib import Path

import cv2
import numpy as np

from .pose import CONNECTIONS
from .scoring import TennisScore, timeline
from .video import encode, writer

COLORS = {"near": (176, 246, 194), "far": (244, 188, 92)}


def vision_overlay(frame, poses, ball_history, court, time):
    h, w = frame.shape[:2]
    polygon = (court.polygon * np.array([w, h])).astype(np.int32)
    cv2.polylines(frame, [polygon], True, (156, 164, 101), 1, cv2.LINE_AA)
    net = [court.image(0, 11.885), court.image(8.23, 11.885)]
    a, b = [tuple((np.array(p) * [w, h]).astype(int)) for p in net]
    cv2.line(frame, a, b, (156, 164, 101), 1, cv2.LINE_AA)
    for slot, pose in poses.items():
        points = pose["landmarks"]
        for a, b in CONNECTIONS:
            if min(points[a][2], points[b][2]) > 0.4:
                cv2.line(
                    frame,
                    (int(points[a][0] * w), int(points[a][1] * h)),
                    (int(points[b][0] * w), int(points[b][1] * h)),
                    COLORS[slot],
                    2,
                    cv2.LINE_AA,
                )
        for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
            if points[i][2] > 0.4:
                cv2.circle(
                    frame,
                    (int(points[i][0] * w), int(points[i][1] * h)),
                    3,
                    COLORS[slot],
                    -1,
                    cv2.LINE_AA,
                )
        x, y = pose["foot"]
        cv2.putText(
            frame,
            f"{slot.upper()} COURT",
            (max(2, int(x * w) - 40), min(h - 8, int(y * h) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            COLORS[slot],
            1,
            cv2.LINE_AA,
        )
    recent = [(t, b) for t, b in ball_history if time - t < 0.55]
    for (t0, b0), (t1, b1) in zip(recent, recent[1:]):
        if t1 - t0 < 0.3:
            cv2.line(
                frame,
                (int(b0["x"] * w), int(b0["y"] * h)),
                (int(b1["x"] * w), int(b1["y"] * h)),
                (100, 255, 225),
                2,
                cv2.LINE_AA,
            )
    if recent and time - recent[-1][0] < 0.18:
        ball = recent[-1][1]
        cv2.circle(
            frame, (int(ball["x"] * w), int(ball["y"] * h)), 7, (100, 255, 225), 2, cv2.LINE_AA
        )
    return frame


def score_overlay(frame, score, names, provisional, event_count=1):
    h, w = frame.shape[:2]
    scale = max(0.5, min(1, w / 1280))
    bw = min(w - 24, int(370 * scale))
    x, y = w - bw - 12, 12
    bh = int(126 * scale)
    panel = frame[y : y + bh, x : x + bw]
    panel[:] = cv2.addWeighted(panel, 0.12, np.full_like(panel, (25, 31, 23)), 0.88, 0)
    tag = (
        "NO POINT EVENTS"
        if event_count == 0
        else "ESTIMATED / REVIEW NEEDED"
        if provisional
        else "REVIEWED POINTS"
    )
    cv2.putText(
        frame,
        "TENNISVISION  |  " + tag,
        (x + 10, y + int(20 * scale)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32 * scale,
        (186, 205, 190),
        1,
        cv2.LINE_AA,
    )
    for i in (0, 1):
        yy = y + int((56 + 40 * i) * scale)
        cv2.circle(frame, (x + 10, yy - 4), 3, COLORS["near" if i == 0 else "far"], -1)
        name = names[i].encode("ascii", "replace").decode()[:20]
        cv2.putText(
            frame,
            name,
            (x + 22, yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52 * scale,
            (242, 247, 243),
            1,
            cv2.LINE_AA,
        )
        value = f"{score['sets'][i]}   {score['games'][i]}   {score['points'][i]}"
        cv2.putText(
            frame,
            value,
            (x + bw - int(125 * scale), yy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58 * scale,
            (194, 246, 176),
            1,
            cv2.LINE_AA,
        )


def render_score(
    root: Path, config, events, progress=lambda value, message: None, cancelled=lambda: False
):
    rows, final = timeline(config.initial, events)
    confirmed_rows, confirmed = timeline(config.initial, events, confirmed_only=True)
    provisional = any(e.status == "pending" for e in events)
    cap = cv2.VideoCapture(str(root / "poses.mp4"))
    fps = cap.get(cv2.CAP_PROP_FPS)
    count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    silent = root / "score-silent.mp4"
    out = writer(silent, fps, size)
    score = TennisScore(config.initial).snapshot()
    cursor, index = 0, 0
    try:
        while True:
            if cancelled():
                raise InterruptedError("Rendering cancelled.")
            ok, frame = cap.read()
            if not ok:
                break
            time = index / fps
            while cursor < len(rows) and rows[cursor]["time"] <= time:
                score = rows[cursor]["score"]
                cursor += 1
            score_overlay(frame, score, config.players, provisional, len(events))
            out.write(frame)
            if index % 60 == 0:
                progress(0.85 + 0.1 * index / count, "Rendering scoreboard")
            index += 1
    finally:
        cap.release()
        out.release()
    destination = root / "annotated.next.mp4"
    encode(silent, root / "source.mp4", destination)
    destination.replace(root / "annotated.mp4")
    silent.unlink(missing_ok=True)
    return dict(
        provisional=provisional,
        estimated=final,
        confirmed=confirmed,
        timeline=rows,
        confirmed_timeline=confirmed_rows,
    )
