"""Download the attributed smoke-test clip; do not redistribute it as a dataset."""

import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/abdullahtarek/tennis_analysis/2595ef08a38d00288bcdf21f873b602ffbb2798d/input_videos/input_video.mp4"
DIGEST = "5f2584a2010864d5d8624c325d7f0571d48bd0756164657ef603dc8a97bab22d"


def main():
    root = Path(__file__).resolve().parents[1] / "data"
    root.mkdir(exist_ok=True)
    destination = root / "tennis-sample.mp4"
    if not destination.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            destination.write_bytes(response.read())
    if hashlib.sha256(destination.read_bytes()).hexdigest() != DIGEST:
        raise ValueError("Unexpected sample digest. Inspect the download before use.")
    print(destination)
    print("Source and media attribution: docs/DATA.md")


if __name__ == "__main__":
    main()
