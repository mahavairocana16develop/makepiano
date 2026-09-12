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
from .separate import STEM_LABELS

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
    stem: str | list[str] = "all"
    legato: bool = True
    chords: bool = True
    level: str = "both"
    force: bool = False
    refine: bool = True
    max_notes: int = 5
    max_span: int = 12
    triplets: str = "16th"
    fidelity: str = "faithful"  # "faithful" | "clean"


def _work(job_id: str, req: JobRequest):
    job = JOBS[job_id]

    def log(msg: str):
        job["log"].append(msg)

    def progress(frac, stage, eta):
        job.update(progress=round(frac, 3), stage=stage, eta=round(eta))

    with _lock:
        job["status"] = "running"
        try:
            opts = ScoreOptions(grid=req.grid, split_pitch=req.split, fixed_bpm=req.bpm, beat_offset=req.beat_offset,
                               beats_per_bar=req.beats_per_bar, legato=req.legato, chords=req.chords, level=req.level,
                               max_notes=req.max_notes, max_span=req.max_span,
                               triplets="off" if req.fidelity == "clean" else req.triplets,
                               short_notes=req.fidelity != "clean")
            r = run(req.url, OUT, opts, stem=req.stem, force=req.force, refine=req.refine, log=log, progress=progress)
            job.update(status="done", title=r.title, files=_files_for(r.workdir))
        except Exception as e:  # noqa: BLE001
            job.update(status="error", error=str(e))


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


LEVEL_LABELS = {"original": "オリジナル", "intermediate": "中級", "beginner": "初級"}


def _level_files(d: Path) -> dict:
    rel = quote(str(d.relative_to(OUT)))
    svgs = sorted((d / "svg").glob("page-*.svg"))
    return {
        "pdf": f"/files/{rel}/score.pdf" if (d / "score.pdf").exists() else None,
        "musicxml": f"/files/{rel}/score.musicxml",
        "svgs": [f"/files/{rel}/svg/{s.name}" for s in svgs],
        "playback": f"/files/{rel}/playback.json",
    }


def _levels_in(d: Path) -> dict:
    levels = {}
    for lv in ("original", "intermediate", "beginner"):
        if (d / lv / "playback.json").exists():
            levels[lv] = {"label": LEVEL_LABELS[lv], **_level_files(d / lv)}
    if not levels and (d / "playback.json").exists():  # pre-levels layout
        levels["original"] = {"label": LEVEL_LABELS["original"], **_level_files(d)}
    return levels


def _files_for(workdir: Path) -> dict:
    rel = quote(str(workdir.relative_to(OUT)))
    stems = {}
    sdir = workdir / "stems"
    if sdir.is_dir():
        for st in ("none", "other", "piano", "no_vocals"):
            d = sdir / st
            if (d / "transcription.mid").exists():
                lv = _levels_in(d)
                if lv:
                    stems[st] = {"label": STEM_LABELS[st], "midi": f"/files/{rel}/stems/{st}/transcription.mid",
                                 "audio": f"/files/{rel}/stems/{st}/audio.m4a" if (d / "audio.m4a").exists() else None,
                                 "levels": lv}
    else:  # pre-stems layouts: one unnamed stem at the job root
        lv = _levels_in(workdir)
        if lv:
            stems["default"] = {"label": "既定", "midi": f"/files/{rel}/transcription.mid",
                                "audio": f"/files/{rel}/audio_stem.m4a" if (workdir / "audio_stem.m4a").exists() else None,
                                "levels": lv}
    return {
        "stems": stems,
        "audio_original": f"/files/{rel}/audio_original.m4a" if (workdir / "audio_original.m4a").exists() else None,
    }


@app.get("/api/results")
def list_results():
    """Previously generated jobs (anything under output/ with playback.json), newest first."""
    def stamp(d: Path):
        cands = list((d / "stems").glob("*/original/playback.json")) if (d / "stems").is_dir() else []
        cands += [d / "original" / "playback.json", d / "playback.json"]
        for f in cands:
            if f.exists():
                return f.stat().st_mtime
        return None
    dirs = [d for d in OUT.iterdir() if d.is_dir() and stamp(d) is not None]
    dirs.sort(key=stamp, reverse=True)
    return [{"title": d.name, "files": f} for d in dirs if (f := _files_for(d))["stems"]]


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(status_code=204)


@app.post("/api/jobs")
def create_job(req: JobRequest, bg: BackgroundTasks):
    job_id = uuid.uuid4().hex[:8]
    JOBS[job_id] = {"id": job_id, "status": "queued", "log": [], "url": req.url, "progress": 0, "stage": "待機中", "eta": None}
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
