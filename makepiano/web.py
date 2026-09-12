"""Minimal web UI: paste a YouTube URL, get a piano score."""
from __future__ import annotations

import threading
import uuid
from urllib.parse import quote
from pathlib import Path

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .pipeline import run
from .score import ScoreOptions

ROOT = Path(__file__).parent
OUT = Path("output")
OUT.mkdir(exist_ok=True)

app = FastAPI(title="makepiano")
app.mount("/files", StaticFiles(directory=str(OUT)), name="files")

JOBS: dict[str, dict] = {}
_lock = threading.Lock()  # one transcription at a time (model is heavy)


class JobRequest(BaseModel):
    url: str
    bpm: float | None = None
    grid: int = 4
    split: int = 60
    beat_offset: int = 0
    beats_per_bar: int = 4
    stem: str = "none"
    legato: bool = True
    chords: bool = True


def _work(job_id: str, req: JobRequest):
    job = JOBS[job_id]

    def log(msg: str):
        job["log"].append(msg)

    with _lock:
        job["status"] = "running"
        try:
            opts = ScoreOptions(grid=req.grid, split_pitch=req.split, fixed_bpm=req.bpm, beat_offset=req.beat_offset,
                               beats_per_bar=req.beats_per_bar, legato=req.legato, chords=req.chords)
            r = run(req.url, OUT, opts, stem=req.stem, log=log)
            job.update(status="done", title=r.title, files=_files_for(r.workdir))
        except Exception as e:  # noqa: BLE001
            job.update(status="error", error=str(e))


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


def _files_for(workdir: Path) -> dict:
    rel = quote(str(workdir.relative_to(OUT)))
    svgs = sorted((workdir / "svg").glob("page-*.svg"))
    return {
        "pdf": f"/files/{rel}/score.pdf" if (workdir / "score.pdf").exists() else None,
        "midi": f"/files/{rel}/transcription.mid",
        "musicxml": f"/files/{rel}/score.musicxml",
        "svgs": [f"/files/{rel}/svg/{s.name}" for s in svgs],
        "playback": f"/files/{rel}/playback.json",
        "audio_original": f"/files/{rel}/audio_original.m4a" if (workdir / "audio_original.m4a").exists() else None,
        "audio_stem": f"/files/{rel}/audio_stem.m4a" if (workdir / "audio_stem.m4a").exists() else None,
    }


@app.get("/api/results")
def list_results():
    """Previously generated jobs (anything under output/ with playback.json), newest first."""
    dirs = [d for d in OUT.iterdir() if d.is_dir() and (d / "playback.json").exists()]
    dirs.sort(key=lambda d: (d / "playback.json").stat().st_mtime, reverse=True)
    return [{"title": d.name, "files": _files_for(d)} for d in dirs]


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.post("/api/jobs")
def create_job(req: JobRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"id": job_id, "status": "queued", "log": [], "url": req.url}
    bg.add_task(_work, job_id, req)
    return {"id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404)
    return JSONResponse(JOBS[job_id])


def main():
    uvicorn.run("makepiano.web:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
