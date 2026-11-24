"""Loopback-first upload API with isolated jobs, cancellation, and reviewed exports."""

import hashlib
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from importlib.resources import files
from pathlib import Path
from threading import Event as CancelEvent
from threading import Lock
from uuid import uuid4

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .assets import ASSETS, model_dir
from .geometry import Court
from .models import AnalysisConfig, BallLabel, Review
from .pipeline import analyze, export_review, write_json
from .scoring import timeline
from .video import inspect_video, normalize

MAX_UPLOAD = 512 * 1024 * 1024
ARTIFACTS = {
    "source.mp4",
    "poster.jpg",
    "preview.jpg",
    "annotated.mp4",
    "report.zip",
    "points.csv",
    "analysis.json",
    "scores.json",
    "events.json",
    "predictions.json",
    "ball.csv",
    "poses.jsonl",
    "motion.csv",
    "config.json",
    "labels.json",
}


class Jobs:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = Lock()
        self.active = None
        self.cancel = CancelEvent()
        self.pool = ThreadPoolExecutor(max_workers=1)

    def path(self, job):
        if not re.fullmatch(r"[0-9a-f]{32}", job):
            raise HTTPException(404, "Match not found.")
        path = self.root / job
        if not (path / "job.json").is_file():
            raise HTTPException(404, "Match not found.")
        return path

    def read(self, job):
        return json.loads((self.path(job) / "job.json").read_text(encoding="utf-8"))

    def save(self, root, data):
        temporary = root / "job.next.json"
        write_json(temporary, data)
        temporary.replace(root / "job.json")

    def submit(self, job, operation, config=None, events=None):
        root = self.path(job)
        with self.lock:
            if self.active:
                raise HTTPException(409, "Another video is processing. Finish or cancel it first.")
            self.active = job
            self.cancel.clear()
            data = self.read(job)
            data.update(
                state="running",
                progress=0,
                message="Starting analysis"
                if operation == "analyze"
                else "Updating reviewed video",
                error=None,
            )
            self.save(root, data)
        self.pool.submit(self._run, job, operation, config, events)

    def _run(self, job, operation, config, events):
        root = self.path(job)
        data = self.read(job)

        def progress(value, message):
            data.update(progress=round(value, 4), message=message)
            self.save(root, data)

        try:
            if operation == "analyze":
                result = analyze(root / "source.mp4", root, config, progress, self.cancel.is_set)
            else:
                scores = export_review(root, config, events, progress, self.cancel.is_set)
                result = data.get("result", {}) | dict(
                    events=[e.model_dump() for e in events], scores=scores
                )
            data.update(state="complete", progress=1, message="Ready for review", result=result)
        except InterruptedError:
            data.update(state="cancelled", message="Analysis cancelled. You can start again.")
        except Exception as error:
            data.update(state="failed", message="Processing failed", error=str(error))
        finally:
            self.save(root, data)
            with self.lock:
                self.active = None


def create_app(output_root=None):
    jobs = Jobs(output_root or os.getenv("TENNISVISION_DATA", "outputs/matches"))

    @asynccontextmanager
    async def lifespan(app):
        yield
        jobs.cancel.set()
        jobs.pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="TennisVision", version="0.1.0", lifespan=lifespan)
    app.state.jobs = jobs
    web = Path(str(files("tennisvision") / "web"))
    app.mount("/static", StaticFiles(directory=web), name="static")

    @app.get("/")
    def home():
        return FileResponse(web / "index.html")

    @app.get("/api/health")
    def health():
        import importlib.util

        return dict(
            ready=True,
            models={key: (model_dir() / value[0]).is_file() for key, value in ASSETS.items()},
            torch=importlib.util.find_spec("torch") is not None,
        )

    @app.post("/api/matches", status_code=201)
    async def upload(video: UploadFile = File(...)):
        job = uuid4().hex
        root = jobs.root / job
        root.mkdir()
        destination = root / "upload.bin"
        size = 0
        try:
            with destination.open("wb") as stream:
                while block := await video.read(1024 * 1024):
                    size += len(block)
                    if size > MAX_UPLOAD:
                        raise HTTPException(413, "Video exceeds the 512 MB upload limit.")
                    stream.write(block)
            original_info, _ = await run_in_threadpool(inspect_video, destination)
            await run_in_threadpool(normalize, destination, root / "source.mp4")
            info, frame = await run_in_threadpool(inspect_video, root / "source.mp4")
            destination.unlink(missing_ok=True)
            cv2.imwrite(str(root / "poster.jpg"), frame)
            data = dict(
                id=job,
                state="uploaded",
                progress=0,
                message="Video ready",
                info=info,
                filename=Path(video.filename or "video").name[:150],
                bytes=size,
                error=None,
                original_info=original_info,
            )
            jobs.save(root, data)
            return data
        except ValueError as error:
            destination.unlink(missing_ok=True)
            raise HTTPException(422, str(error)) from error
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await video.close()

    @app.get("/api/matches/{job}")
    def status(job: str):
        data = jobs.read(job)
        if data["state"] == "running" and jobs.active != job:
            data.update(
                state="failed", error="The server restarted during processing. Start again."
            )
        return data

    @app.get("/api/matches/{job}/files/{name}")
    def artifact(job: str, name: str, download: bool = False):
        if name not in ARTIFACTS:
            raise HTTPException(404, "Artifact not found.")
        path = jobs.path(job) / name
        if not path.is_file():
            raise HTTPException(404, "Artifact is not ready.")
        return FileResponse(path, filename=name if download else None)

    @app.post("/api/matches/{job}/analyze", status_code=202)
    def start(job: str, config: AnalysisConfig):
        if jobs.read(job)["state"] == "complete":
            raise HTTPException(409, "Create a new analysis run to preserve the completed results.")
        try:
            Court(config.court)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        jobs.submit(job, "analyze", config)
        return dict(id=job, state="running")

    @app.post("/api/matches/{job}/fork", status_code=201)
    def fork(job: str):
        source = jobs.path(job)
        data = jobs.read(job)
        new_id = uuid4().hex
        root = jobs.root / new_id
        root.mkdir()
        for name in ("source.mp4", "poster.jpg", "labels.json"):
            if (source / name).is_file():
                shutil.copyfile(source / name, root / name)
        data = {key: value for key, value in data.items() if key != "result"}
        data.update(
            id=new_id, parent=job, state="uploaded", progress=0, message="Video ready", error=None
        )
        jobs.save(root, data)
        return data

    @app.post("/api/matches/{job}/cancel")
    def cancel(job: str):
        jobs.path(job)
        if jobs.active == job:
            jobs.cancel.set()
        return dict(message="Cancellation requested")

    @app.put("/api/matches/{job}/review", status_code=202)
    def review(job: str, review: Review):
        root = jobs.path(job)
        data = jobs.read(job)
        if data["state"] != "complete":
            raise HTTPException(409, "Finish the analysis before reviewing points.")
        ids = [e.id for e in review.events]
        if len(set(ids)) != len(ids) or any(
            e.time > data["info"]["duration"] or e.start > e.time for e in review.events
        ):
            raise HTTPException(422, "Event IDs must be unique and times must be inside the video.")
        config = AnalysisConfig.model_validate_json(
            (root / "config.json").read_text(encoding="utf-8")
        )
        try:
            timeline(config.initial, review.events)
            timeline(config.initial, review.events, confirmed_only=True)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        jobs.submit(job, "review", config, review.events)
        return dict(id=job, state="running")

    @app.post("/api/matches/{job}/preview-score")
    def preview_score(job: str, review: Review):
        root = jobs.path(job)
        if not (root / "config.json").is_file():
            raise HTTPException(409, "Analyze the video first.")
        config = AnalysisConfig.model_validate_json(
            (root / "config.json").read_text(encoding="utf-8")
        )
        try:
            rows, estimated = timeline(config.initial, review.events)
            _, confirmed = timeline(config.initial, review.events, confirmed_only=True)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        return dict(timeline=rows, estimated=estimated, confirmed=confirmed)

    @app.put("/api/matches/{job}/labels")
    def labels(job: str, labels: list[BallLabel]):
        root = jobs.path(job)
        info = jobs.read(job)["info"]
        if (
            len(labels) > 100000
            or len({label.frame for label in labels}) != len(labels)
            or any(label.frame >= info["frames"] for label in labels)
        ):
            raise HTTPException(422, "Label frames must be unique and inside the video.")
        write_json(
            root / "labels.json",
            dict(
                schema_version=1,
                coordinate_system="normalized_image",
                fps=info["fps"],
                width=info["width"],
                height=info["height"],
                input_sha256=hashlib.sha256((root / "source.mp4").read_bytes()).hexdigest(),
                labels=[label.model_dump() for label in sorted(labels, key=lambda v: v.frame)],
            ),
        )
        return dict(saved=len(labels))

    return app
