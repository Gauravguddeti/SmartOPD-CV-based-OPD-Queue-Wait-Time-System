"""Vision worker: a background thread consuming a video source.

Sources:
  * an uploaded/original video file (samples/... or data/uploads/...)
  * a live webcam (cv2.VideoCapture(index))

Loop handling:
  * CONTINUOUS sources (filename contains "pingpong") are frame-continuous
    at the wrap point - the loop restarts WITHOUT resetting the tracker or
    holding events, so track identities (and consultations) survive.
  * Ordinary files loop with a jump cut: that is a REAL SEAM - events are
    held for SEAM_HOLD seconds, the tracker is reset, and occupants are
    re-bound to the people the camera then sees inside the doctor zone.

Every frame is annotated (zones, boxes, track ids, occupant tokens, CCTV
overlay) and the latest JPEG is kept in memory for the web pages.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import cv2

import config
from vision import CentroidTracker, PersonDetector, ZoneAnalyzer


def is_continuous_source(source: str) -> bool:
    """Ping-pong builds are frame-continuous at the wrap; nothing else is."""
    try:
        return "pingpong" in Path(source).stem.lower()
    except (TypeError, ValueError):
        return False


def _draw_label(img, text, x, y, color) -> None:
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, color, 1, cv2.LINE_AA)


class VisionWorker:
    def __init__(self, engine):
        self.engine = engine
        self.lock = threading.Lock()
        self.status = {
            "source": None, "kind": None, "running": False,
            "detector_ready": False, "fps": 0.0, "error": None,
            "frame_ts": 0.0, "loops": 0, "cycles": 0, "continuous": False,
        }
        self._jpeg: bytes | None = None
        self._thread: threading.Thread | None = None
        self._cap: cv2.VideoCapture | None = None
        self._stop = threading.Event()
        self._continuous = False
        self.detector: PersonDetector | None = None
        self.tracker = CentroidTracker()
        self.analyzer = ZoneAnalyzer(
            on_entry=self._on_entry, on_exit=self._on_exit,
            on_passby=self._on_passby, on_note=self._on_note,
            on_crowd=self._on_crowd)

        zones = getattr(engine, "zones", None)
        if zones:
            self.analyzer.set_zones(zones.get("wait"), zones.get("doc"))

        self._load_detector()

    # ------------------------------------------------------------- lifecycle
    def _load_detector(self) -> None:
        def _run():
            try:
                self.detector = PersonDetector()
                with self.lock:
                    self.status["detector_ready"] = True
            except Exception as e:  # pragma: no cover - depends on weights file
                with self.lock:
                    self.status["error"] = f"Detector failed to load: {e}"
        threading.Thread(target=_run, daemon=True).start()

    def start(self, source: str, kind: str) -> dict:
        self.stop()
        self._stop.clear()
        cap = cv2.VideoCapture()
        if kind == "webcam":
            ok = cap.open(int(source))
        else:
            ok = cap.open(source)
        if not ok:
            cap.release()
            with self.lock:
                self.status.update(error=f"Cannot open {kind} source: {source}",
                                   running=False)
            return self.get_status()
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._continuous = is_continuous_source(source) if kind == "file" else False
        # a source switch is a real seam: keep occupants/tokens, hold events,
        # then re-bind to whoever the new source shows inside the zone
        self.tracker = CentroidTracker()
        self.analyzer.begin_seam()
        self.engine.time_scale = 1.0 if kind == "webcam" else config.TIME_SCALE
        with self.lock:
            self._cap = cap
            self.status.update(source=source, kind=kind, running=True,
                               error=None, continuous=self._continuous)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self.get_status()

    def stop(self) -> None:
        self._stop.set()
        th, self._thread = self._thread, None
        if th and th.is_alive():
            th.join(timeout=3.0)
        with self.lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self.status.update(running=False)

    # ----------------------------------------------------------------- loop
    def _loop(self) -> None:
        interval = 1.0 / config.DEFAULT_TARGET_FPS
        eof_failures = 0
        while not self._stop.is_set():
            t0 = time.time()
            with self.lock:
                cap = self._cap
            if cap is None:
                break
            ok, frame = cap.read()
            if not ok:
                if self.status.get("kind") == "file":
                    eof_failures += 1
                    if eof_failures > 3:
                        with self.lock:
                            self.status["running"] = False
                        break
                    if not self._continuous:
                        # jump-cut loop -> real seam (hold + tracker reset +
                        # occupant re-bind).  Continuous sources skip this.
                        self.tracker = CentroidTracker()
                        self.analyzer.begin_seam()
                    if eof_failures > 1:
                        # some codecs refuse a plain seek -> reopen the file
                        src = self.status.get("source")
                        with self.lock:
                            if self._cap is not None:
                                self._cap.release()
                            self._cap = cv2.VideoCapture(src) if src else None
                            cap = self._cap
                        if cap is None:
                            break
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    with self.lock:
                        self.status["cycles"] = self.status.get("cycles", 0) + 1
                    time.sleep(0.05)
                    continue
                time.sleep(0.05)
                continue
            eof_failures = 0

            if self.detector is not None:
                dets = self.detector.detect(frame)
                visible = self.tracker.update(dets)
                for gone in self.tracker.expired:
                    self.analyzer.track_vanished(gone.tid)
                self.analyzer.process(visible)
                self._annotate(frame, visible)
            else:
                self._annotate(frame, [])

            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ok2:
                with self.lock:
                    self._jpeg = buf.tobytes()
                    self.status["frame_ts"] = time.time()
                    self.status["loops"] += 1
                    self.status["fps"] = round(1.0 / max(1e-6, time.time() - t0), 1)
            wait = interval - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)

    # -------------------------------------------------------------- drawing
    def _annotate(self, frame, tracks) -> None:
        h, w = frame.shape[:2]
        z_wait = self.analyzer.wait_zone
        z_doc = self.analyzer.doc_zone

        def rect(zone, color, label):
            if zone and len(zone) == 4:
                x0, y0, x1, y1 = (int(zone[0] * w), int(zone[1] * h),
                                  int(zone[2] * w), int(zone[3] * h))
                overlay = frame.copy()
                cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
                cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
                _draw_label(frame, label, x0 + 5, y0 + 20, color)

        rect(z_wait, (86, 128, 255), "WAITING AREA")
        rect(z_doc, (16, 160, 245), "DOCTOR ROOM")

        for t in tracks:
            x0, y0, x1, y1 = t.bbox
            p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))
            color = (16, 160, 245) if t.in_doc else (90, 200, 90) if t.in_wait \
                else (235, 235, 235)
            cv2.rectangle(frame, p0, p1, color, 2)
            label = f"#{t.tid}"
            if t.in_doc:
                tag = self.engine.occupant_tag(t.tid)   # "token #N" / "ghost"
                if tag:
                    label += f" · {tag}"
            _draw_label(frame, label, x0 + 3, max(16, y0 - 6), color)

        # ---- CCTV look: camera label, timestamp, REC dot -------------------
        if config.CCTV_OVERLAY:
            _draw_label(frame, config.CAMERA_LABEL, 12, h - 14, (255, 255, 255))
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            (tw, _th), _ = cv2.getTextSize(stamp, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            _draw_label(frame, stamp, max(12, w - tw - 40), h - 14, (255, 255, 255))
            cv2.circle(frame, (w - 20, 20), 6, (60, 60, 235), -1)
            _draw_label(frame, "REC", w - 58, 26, (240, 240, 240))

    # ------------------------------------------------------------ callbacks
    def _on_entry(self, occ: dict) -> None:
        token_id = self.engine.begin_occupant(occ["key"], occ["t_in"],
                                              occ["ghost"], src="camera")
        occ["token_id"] = token_id

    def _on_exit(self, key, occ: dict) -> None:
        self.engine.remove_occupant(key, src="camera")

    def _on_passby(self, dwell_s: float) -> None:
        self.engine.pass_by(dwell_s)

    def _on_note(self, msg: str) -> None:
        self.engine.note(msg)

    def _on_crowd(self, in_frame: int, in_waiting: int, in_room: int = 0) -> None:
        self.engine.update_crowd(in_frame, in_waiting, in_room)

    # ----------------------------------------------------------------- api
    def set_zones(self, wait_zone, doc_zone) -> None:
        self.analyzer.set_zones(wait_zone, doc_zone)

    def clear_room(self) -> None:
        """Forget all occupants (queue reset).  The vision side will
        re-ghost whoever is still inside on the next frames."""
        self.analyzer.occupants = {}
        self.analyzer._pending = {}

    def get_jpeg(self) -> bytes | None:
        with self.lock:
            return self._jpeg

    def get_status(self) -> dict:
        with self.lock:
            st = dict(self.status)
        st["occupants"] = len(self.analyzer.occupants)
        return st

    # --------------------------------------------------- zone auto-detection
    def auto_detect_zones(self, source: str, samples: int = 12) -> dict:
        """Sample N frames of a video; cells where people appear in most
        samples form the waiting area; a tight cluster of vanish-point
        detections near a frame edge is the doctor door."""
        if self.detector is None:
            return {"ok": False, "error": "Detector not ready yet."}
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return {"ok": False, "error": f"Cannot open {source}"}
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not n or n < samples:
            cap.release()
            return {"ok": False, "error": "Clip too short to auto-detect."}
        GX, GY = 16, 9
        cnt = [0] * (GX * GY)
        points: list[tuple[float, float]] = []
        for i in range(samples):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (i + 0.5) / samples))
            ok, frame = cap.read()
            if not ok:
                continue
            seen: set[int] = set()
            for d in self.detector.detect(frame):
                x, y = d["cx"], d["cy"] + (d["bbox"][3] - d["bbox"][1]) * 0.3
                points.append((x, y))
                seen.add(min(GY - 1, int(y * GY)) * GX + min(GX - 1, int(x * GX)))
            for k in seen:
                cnt[k] += 1
        cap.release()

        mx = max(cnt) if cnt else 0
        if mx == 0:
            return {"ok": False,
                    "error": "No people found. Draw the zones manually."}
        cells = [k for k, c in enumerate(cnt) if c >= max(1, mx * 0.6)]
        xs = [k % GX for k in cells]
        ys = [k // GX for k in cells]
        wait = [max(0.0, (min(xs) - 1) / GX), max(0.0, (min(ys) - 1) / GY),
                min(1.0, (max(xs) + 2) / GX), min(1.0, (max(ys) + 2) / GY)]

        def inside(z, x, y):
            return z[0] <= x <= z[2] and z[1] <= y <= z[3]

        cx, cy = (wait[0] + wait[2]) / 2, (wait[1] + wait[3]) / 2
        outside = [p for p in points if not inside(wait, p[0], p[1])]
        # A door is only plausible if the vanish cluster sits near a frame
        # edge and is compact.  Otherwise we return None and let the operator
        # draw it - a bogus mid-room box fires phantom occupancy events.
        doc = None
        cluster = _largest_cluster(outside, radius=0.10)
        if len(cluster) >= 2:
            px = [p[0] for p in cluster]
            py = [p[1] for p in cluster]
            bw, bh = max(px) - min(px), max(py) - min(py)
            mx2, my2 = (max(px) + min(px)) / 2, (max(py) + min(py)) / 2
            near_edge = min(mx2, 1 - mx2) < 0.22 or min(my2, 1 - my2) < 0.22
            if bw <= 0.40 and bh <= 0.40 and near_edge:
                doc = [max(0.0, min(px) - 0.04), max(0.0, min(py) - 0.06),
                       min(1.0, max(px) + 0.04), min(1.0, max(py) + 0.06)]
        self.set_zones(wait, doc)
        return {"ok": True, "wait": wait, "doc": doc}


def _largest_cluster(points: list[tuple[float, float]],
                     radius: float = 0.10) -> list[tuple[float, float]]:
    """Greedy single-link clustering; returns the biggest cluster."""
    remaining = list(points)
    best: list[tuple[float, float]] = []
    while remaining:
        seed = remaining.pop()
        cluster = [seed]
        changed = True
        while changed:
            changed = False
            for p in list(remaining):
                if any((p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 <= radius ** 2
                       for c in cluster):
                    cluster.append(p)
                    remaining.remove(p)
                    changed = True
        if len(cluster) > len(best):
            best = cluster
    return best
