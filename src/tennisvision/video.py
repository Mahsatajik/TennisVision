"""Video decoding and browser-compatible H.264 export with optional source audio."""

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg


def inspect_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width, height = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        ok, frame = cap.read()
        if not ok or fps < 1 or fps > 240 or frames < 1 or not width or not height:
            raise ValueError(
                "Video could not be decoded. Upload a readable MP4, MOV, AVI, or WebM."
            )
        if frames / fps > 300:
            raise ValueError("Use a continuous clip of at most 5 minutes.")
        return dict(
            fps=fps, frames=frames, width=width, height=height, duration=frames / fps
        ), frame
    finally:
        cap.release()


def encode(silent: Path, audio: Path, destination: Path):
    args = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(silent),
        "-i",
        str(audio),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "21",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-af",
        "apad",
        "-shortest",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    result = subprocess.run(args, capture_output=True, text=True, timeout=900)
    if result.returncode:
        destination.unlink(missing_ok=True)
        raise RuntimeError("Video export failed: " + result.stderr[-600:])


def normalize(source: Path, destination: Path):
    """Normalize phone/VFR inputs to a stable 30-fps analysis timeline, max 1280 pixels wide."""
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "0:a:0?",
            "-vf",
            "scale='min(1280,iw)':-2,fps=30",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if result.returncode:
        destination.unlink(missing_ok=True)
        raise ValueError("Video normalization failed. Try a shorter, readable MP4 clip.")


def writer(path, fps, size):
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not out.isOpened():
        raise RuntimeError("The video encoder could not start.")
    return out
