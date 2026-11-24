"""Download versioned model assets atomically and verify known SHA-256 digests."""

import hashlib
import os
import urllib.request
from pathlib import Path

ASSETS = {
    "pose": (
        "pose_landmarker_full.task",
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    ),
    "tracknet": (
        "tracknet_weights.pth",
        "https://huggingface.co/vishnushenoy09/tracknet-v1-tennis/resolve/c53bc2f8d5524596e70737c5e6aa9b35bfb52e9b/tracknet_weights.pth",
    ),
}
SHA256 = {
    "pose": "5134a3aad27a58b93da0088d431f366da362b44e3ccfbe3462b3827a839011b1",
    "tracknet": "c735bc1a1b13a35f179c6492f778ef4ebb9bffd512a96f4d970b32e076653076",
}


def model_dir():
    return Path(os.getenv("TENNISVISION_MODELS", "models")).resolve()


def fetch_models(tracknet=False):
    root = model_dir()
    root.mkdir(parents=True, exist_ok=True)
    for key in ["pose"] + (["tracknet"] if tracknet else []):
        name, url = ASSETS[key]
        destination = root / name
        if not destination.exists():
            temporary = root / (name + ".partial")
            try:
                with (
                    urllib.request.urlopen(url, timeout=60) as response,
                    temporary.open("wb") as out,
                ):
                    while block := response.read(1024 * 1024):
                        out.write(block)
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
        digest = hashlib.sha256(destination.read_bytes()).hexdigest()
        if digest != SHA256[key]:
            raise ValueError(
                f"Checksum mismatch for {name}. Remove the file and download it again."
            )
        print(f"{name}: {digest}")


def require_model(key):
    path = model_dir() / ASSETS[key][0]
    if not path.is_file():
        raise ValueError(
            "Model missing. Run `tennisvision models"
            + (" --tracknet`" if key == "tracknet" else "`")
        )
    return path
