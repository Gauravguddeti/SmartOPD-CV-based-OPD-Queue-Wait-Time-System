"""SmartOPD FastAPI server.

Serves the three interfaces (monitor / nurse / patient) and a JSON API.
State persists to data/state.json; all files stay inside the workspace.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import config
from config import PAGES_DIR, SAMPLES_DIR, STATIC_DIR, DATA_DIR
from queue_manager import QueueEngine, QueueError
from store import Store
from worker import VisionWorker

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_MB = 200

app = FastAPI(title="SmartOPD", docs_url="/api/docs")

store = Store(config.STATE_PATH)
engine = QueueEngine.from_dict(store.load(dict))
worker = VisionWorker(engine)

# zones: active set + one saved set per clip file (a fixed camera keeps its
# zones forever; switching demo clips must not reuse another clip's boxes)
_zones: dict = {"wait": None, "doc": None}
_zones_by_source: dict[str, dict] = {}
_active_source: str | None = None


def _load_extras() -> None:
    global _zones, _zones_by_source
    data = store.load(dict)
    z = data.get("zones") or {}
    if isinstance(z, dict):
        _zones["wait"] = z.get("wait")
        _zones["doc"] = z.get("doc")
        worker.set_zones(_zones["wait"], _zones["doc"])
    by_src = data.get("zones_by_source") or {}
    if isinstance(by_src, dict):
        _zones_by_source = {k: v for k, v in by_src.items() if isinstance(v, dict)}


def _save_zones() -> None:
    data = engine.to_dict()
    data["zones"] = _zones
    data["zones_by_source"] = _zones_by_source
    store.save(data)


_load_extras()


def _autostart_demo() -> None:
    """Make the ping-pong build the default demo source when it exists."""
    global _zones, _active_source
    demo = config.SAMPLES_DIR / "pingpong.mp4"
    if demo.exists():
        try:
            worker.start(str(demo), "file")
            _active_source = str(demo)
            saved = _zones_by_source.get(str(demo))
            if saved:
                _zones = {"wait": saved.get("wait"), "doc": saved.get("doc")}
                worker.set_zones(_zones["wait"], _zones["doc"])
        except Exception as e:            # pragma: no cover
            print(f"[SmartOPD] demo autostart failed: {e}")


_autostart_demo()


# --------------------------------------------------------------- persistence
def _autosave_loop() -> None:
    last = -1
    while True:
        time.sleep(2.0)
        if engine.version != last:
            last = engine.version
            data = engine.to_dict()
            data["zones"] = _zones
            data["zones_by_source"] = _zones_by_source
            store.save(data)


threading.Thread(target=_autosave_loop, daemon=True).start()


# ------------------------------------------------------------------- schemas
class TokenIn(BaseModel):
    name: str
    kind: str = "New visit"
    priority: bool = False


class ActionIn(BaseModel):
    action: str            # call | skip | noshow | requeue


class SettingsIn(BaseModel):
    auto_advance: bool | None = None
    time_scale: float | None = None      # demo time factor (clinic-min per video-s)


class SourceIn(BaseModel):
    kind: str              # file | webcam
    value: str = "0"


class ZonesIn(BaseModel):
    wait: list[float] | None = None
    doc: list[float] | None = None


class AutoIn(BaseModel):
    source: str | None = None


def _full_state() -> dict:
    s = engine.to_dict()
    s["stats"] = engine.stats()
    s["eta_by_token"] = {
        str(t.no): engine.eta(t.no)
        for t in engine.tokens if t.status in ("waiting", "called")
    }
    s["zones"] = _zones
    s["vision"] = worker.get_status()
    s["server_now"] = time.time()
    return s


def _err(e: QueueError, code: int = 400) -> JSONResponse:
    return JSONResponse({"error": str(e)}, status_code=code)


# --------------------------------------------------------------------- pages
@app.get("/", include_in_schema=False)
def monitor_page() -> FileResponse:
    return FileResponse(PAGES_DIR / "monitor.html")


@app.get("/nurse", include_in_schema=False)
def nurse_page() -> FileResponse:
    return FileResponse(PAGES_DIR / "nurse.html")


@app.get("/patient", include_in_schema=False)
def patient_page() -> FileResponse:
    return FileResponse(PAGES_DIR / "patient.html")


# ----------------------------------------------------------------- api: state
@app.get("/api/state")
def get_state() -> dict:
    return _full_state()


@app.get("/api/export.csv")
def export_csv() -> Response:
    return Response(engine.csv_text(), media_type="text/csv",
                    headers={"Content-Disposition":
                             "attachment; filename=opd_history.csv"})


# ---------------------------------------------------------------- api: queue
@app.post("/api/tokens")
def add_token(body: TokenIn):
    try:
        tok = engine.register(body.name, body.kind, body.priority)
        return {"token": tok.__dict__, "state": _full_state()}
    except QueueError as e:
        return _err(e)


@app.post("/api/tokens/seed")
def seed_tokens():
    engine.seed_demo()
    return {"state": _full_state()}


@app.post("/api/queue/call-next")
def call_next():
    try:
        tok = engine.call_next()
        return {"token": tok.__dict__, "state": _full_state()}
    except QueueError as e:
        return _err(e)


@app.post("/api/tokens/{no}/action")
def token_action(no: int, body: ActionIn):
    try:
        tok = engine.set_status(no, body.action)
        return {"token": tok.__dict__, "state": _full_state()}
    except QueueError as e:
        return _err(e)


# ----------------------------------------------------------------- api: room
@app.post("/api/room/enter")
def room_enter(request: Request):
    src = request.query_params.get("src", "manual")
    engine.manual_enter(src)
    return {"state": _full_state()}


@app.post("/api/room/undo")
def room_undo():
    try:
        tok = engine.undo_last()
        return {"token": tok.__dict__, "state": _full_state()}
    except QueueError as e:
        return _err(e)


@app.post("/api/settings")
def update_settings(body: SettingsIn):
    if body.auto_advance is not None:
        with engine._lock:
            engine.auto_advance = bool(body.auto_advance)
            engine.version += 1
            engine._log("Queue mode: " +
                       ("camera advances the queue automatically"
                        if engine.auto_advance else
                        "nurse confirms every admission"))
    if body.time_scale is not None:
        ts = max(10.0, min(120.0, float(body.time_scale)))
        with engine._lock:
            engine.time_scale = ts
            engine.room["scale"] = ts
            engine.version += 1
            engine._log(f"Demo time factor set to ×{ts:g} "
                        f"(1 video-s ≈ {ts/60:.1f} clinic-min)")
    return {"state": _full_state()}


@app.post("/api/room/free")
def room_free(request: Request):
    src = request.query_params.get("src", "manual")
    engine.manual_free(src)
    return {"state": _full_state()}


@app.post("/api/simulate")
def simulate():
    try:
        result = engine.simulate()
        return {"result": result, "state": _full_state()}
    except QueueError as e:
        return _err(e)


@app.post("/api/reset")
def reset_all():
    engine.reset()
    worker.clear_room()          # stale occupants must not survive a reset
    return {"state": _full_state()}


# ---------------------------------------------------------------- api: vision
@app.get("/api/snapshot.jpg")
def snapshot_jpg():
    jpg = worker.get_jpeg()
    if jpg is None:
        # tiny grey placeholder so the <img> never breaks
        import cv2, numpy as np
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        frame[:] = (18, 24, 38)
        cv2.putText(frame, "no video source", (170, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 130, 150), 2)
        ok, buf = cv2.imencode(".jpg", frame)
        jpg = buf.tobytes()
    return Response(jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.get("/api/samples")
def list_samples():
    vids = []
    for directory, uploaded in ((config.SAMPLES_DIR, False), (UPLOAD_DIR, True)):
        if not directory.exists():
            continue
        for p in sorted(directory.glob("*")):
            if p.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
                vids.append({"name": p.name, "path": str(p),
                             "size_mb": round(p.stat().st_size / 1e6, 1),
                             "uploaded": uploaded})
    return {"samples": vids}


@app.post("/api/source")
def set_source(body: SourceIn):
    global _zones, _active_source
    if body.kind == "file":
        path = Path(body.value)
        if not path.is_absolute():
            path = config.ROOT / body.value
        path = path.resolve()
        if not str(path).startswith(str(config.ROOT.resolve())):
            return JSONResponse({"error": "Only files inside the workspace are allowed."},
                                status_code=400)
        if not path.exists():
            return JSONResponse({"error": f"File not found: {path}"}, status_code=404)
        status = worker.start(str(path), "file")
        _active_source = str(path)
        saved = _zones_by_source.get(_active_source)
        if saved:
            _zones = {"wait": saved.get("wait"), "doc": saved.get("doc")}
        worker.set_zones(_zones["wait"], _zones["doc"])
        status["zones"] = _zones
        status["zones_from_clip"] = bool(saved)
        return status
    elif body.kind == "webcam":
        _active_source = None
        return worker.start(body.value or "0", "webcam")
    return JSONResponse({"error": "kind must be 'file' or 'webcam'"}, status_code=400)


@app.post("/api/zones")
def set_zones(body: ZonesIn):
    global _zones
    _zones = {"wait": body.wait, "doc": body.doc}
    worker.set_zones(body.wait, body.doc)
    if _active_source:
        _zones_by_source[_active_source] = dict(_zones)
    _save_zones()
    return {"zones": _zones}


@app.post("/api/zones/auto")
def auto_zones(body: AutoIn):
    global _zones
    source = body.source or worker.get_status().get("source")
    if not source:
        return JSONResponse({"error": "No video source selected."},
                            status_code=400)
    res = worker.auto_detect_zones(source)
    if res.get("ok"):
        _zones = {"wait": res.get("wait"), "doc": res.get("doc")}
        if _active_source:
            _zones_by_source[_active_source] = dict(_zones)
        _save_zones()
        res["zones"] = _zones
    return res


@app.post("/api/vision/stop")
def stop_vision():
    worker.stop()
    return worker.get_status()


@app.post("/api/upload")
async def upload_clip(file: UploadFile = File(...)):
    """Store an uploaded clip inside data/uploads/ and return its path,
    so the browser can immediately select it as a vision source."""
    import re
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", file.filename or "clip.mp4")
    if not any(safe.lower().endswith(e) for e in
               (".mp4", ".avi", ".mov", ".mkv", ".webm")):
        return JSONResponse({"error": "Only video files are allowed."},
                            status_code=400)
    dest = UPLOAD_DIR / f"{int(time.time())}_{safe}"
    size = 0
    with dest.open("wb") as out:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_MB << 20:
                out.close()
                dest.unlink(missing_ok=True)
                return JSONResponse(
                    {"error": f"File larger than {MAX_UPLOAD_MB} MB."},
                    status_code=400)
            out.write(chunk)
    return {"path": str(dest), "size_mb": round(size / 1e6, 1)}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
