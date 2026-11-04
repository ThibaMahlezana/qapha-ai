"""Read-only HTTP API for the replay laboratory.

Serves timelines built by app.pipeline.timeline and the individual frames
of the video each timeline references.  Frames are decoded server-side
and sent as JPEG so the browser shows the exact frame a prediction refers
to (and so codec support in the browser does not matter).

Nothing here writes to datasets/; the API cannot alter experiments.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response


NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
DEFAULT_DATA = Path(__file__).resolve().parents[3] / "datasets"


def create_app(data_dir: Path | str | None = None) -> FastAPI:
    data = Path(data_dir or os.environ.get("QAPHA_DATA_DIR") or DEFAULT_DATA).resolve()
    timelines_dir = (data / "timelines").resolve()
    recorded_dir = (data / "recorded").resolve()
    # Videos may live in recorded/ or in a fresh-clip analysis folder.
    video_dirs = [recorded_dir] + sorted(
        p.resolve() for p in data.glob("fresh_clip_analysis*") if p.is_dir())

    app = FastAPI(title="Qapha Lab API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    def load_timeline(name: str) -> dict:
        if not NAME.match(name):
            raise HTTPException(400, "invalid timeline name")
        path = timelines_dir / f"{name}.json"
        if not path.is_file():
            raise HTTPException(404, "timeline not found")
        return json.loads(path.read_text(encoding="utf-8"))

    cache: dict[str, tuple[bytes, ...]] = {}
    cache_lock = threading.Lock()

    def decode_frames(video_name: str) -> tuple[bytes, ...]:
        # One decode per video, shared by all requests.  Without the lock,
        # many simultaneous frame requests each opened the same file, which
        # can fail on Windows and left frames blank in the lab.
        with cache_lock:
            if video_name not in cache:
                if len(cache) >= 6:
                    cache.pop(next(iter(cache)))
                cache[video_name] = _decode(video_name)
            return cache[video_name]

    def _resolve(video_name: str) -> Path:
        """Find a timeline's video, only inside the allowed folders.

        New timelines store a path relative to datasets/, e.g.
        "fresh_clip_analysis_v2/drop_150cm.mp4".  Older ones store a bare
        file name; those are looked up in recorded/ first, then in the
        newest analysis folder first.
        """
        allowed = set(video_dirs)
        if "/" in video_name:
            candidate = (data / video_name).resolve()
            if candidate.parent in allowed and candidate.is_file():
                return candidate
            raise HTTPException(404, "video not found")
        for folder in [recorded_dir] + sorted(video_dirs[1:], reverse=True):
            candidate = (folder / video_name).resolve()
            if candidate.parent == folder and candidate.is_file():
                return candidate
        raise HTTPException(404, "video not found")

    def _decode(video_name: str) -> tuple[bytes, ...]:
        path = _resolve(video_name)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise HTTPException(500, "cannot open video")
        frames = []
        try:
            while True:
                ok, image = capture.read()
                if not ok:
                    break
                ok, encoded = cv2.imencode(
                    ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ok:
                    raise HTTPException(500, "cannot encode frame")
                frames.append(encoded.tobytes())
        finally:
            capture.release()
        return tuple(frames)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "timelines_dir_exists": timelines_dir.is_dir()}

    @app.get("/api/timelines")
    def list_timelines():
        if not timelines_dir.is_dir():
            return []
        items = []
        for path in sorted(timelines_dir.glob("*.json")):
            if not NAME.match(path.stem):
                continue
            t = json.loads(path.read_text(encoding="utf-8"))
            items.append({
                "name": path.stem,
                "frame_count": t.get("frame_count"),
                "concept": (t.get("source", {}).get("label")
                            or t.get("source", {}).get("concept")),
                "video": t.get("video"),
                "summary": t.get("summary"),
            })
        return items

    @app.get("/api/timelines/{name}")
    def get_timeline(name: str):
        return load_timeline(name)

    @app.get("/api/timelines/{name}/frames/{index}")
    def get_frame(name: str, index: int):
        timeline = load_timeline(name)
        video = (timeline.get("video") or {}).get("file")
        if not video:
            raise HTTPException(404, "timeline has no video")
        frames = decode_frames(video)
        offset = int((timeline.get("video") or {}).get("frame_offset") or 0)
        if not 0 <= index < timeline["frame_count"] or index + offset >= len(frames):
            raise HTTPException(404, "frame out of range")
        return Response(frames[index + offset], media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=3600"})

    return app


app = create_app()
