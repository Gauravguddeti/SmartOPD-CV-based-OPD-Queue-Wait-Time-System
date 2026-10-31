"""SmartOPD vision module.

Owns everything camera-related:

* YOLO11n person detection (weights from <workspace>/models, never elsewhere)
* a lightweight centroid tracker (greedy nearest-neighbour matching, which is
  plenty for an OPD room and has zero extra dependencies)
* the occupant-based, direction-aware zone state machine that converts raw
  tracks into doctor-room ENTRY/EXIT events + waiting-area crowd counts

Runs entirely on the server.  The browser only displays annotated frames.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import config


# --------------------------------------------------------------------- tracker
@dataclass
class Track:
    tid: int
    cx: float
    cy: float
    bbox: tuple[float, float, float, float]
    miss: int = 0
    hits: int = 1
    in_doc: bool = False
    in_wait: bool = False


class CentroidTracker:
    """Greedy nearest-neighbour matching with a miss budget per track."""

    def __init__(self, max_miss: int = config.TRACK_MAX_MISS,
                 match_dist: float = 0.18):
        self.max_miss = max_miss
        self.match_dist = match_dist
        self.tracks: list[Track] = []
        self._next = 1
        # tracks removed by the most recent update() call
        self.expired: list[Track] = []

    def update(self, detections: list[dict]) -> list[Track]:
        """detections: [{'cx','cy','bbox'}, ...] in normalised coords.

        Returns visible tracks; removed tracks land in self.expired so the
        caller can run 'vanished at the door' logic on them.
        """
        self.expired = []
        used: set[int] = set()
        for tr in self.tracks:
            best_i, best_d = -1, self.match_dist
            for i, d in enumerate(detections):
                if i in used:
                    continue
                dist = ((d["cx"] - tr.cx) ** 2 + (d["cy"] - tr.cy) ** 2) ** 0.5
                if dist < best_d:
                    best_i, best_d = i, dist
            if best_i >= 0:
                used.add(best_i)
                d = detections[best_i]
                tr.cx, tr.cy, tr.bbox = d["cx"], d["cy"], d["bbox"]
                tr.miss = 0
                tr.hits += 1
            else:
                tr.miss += 1
        for i, d in enumerate(detections):
            if i in used:
                continue
            self.tracks.append(Track(tid=self._next, cx=d["cx"], cy=d["cy"],
                                     bbox=d["bbox"]))
            self._next += 1
        survivors = []
        for t in self.tracks:
            if t.miss > self.max_miss:
                self.expired.append(t)
            else:
                survivors.append(t)
        self.tracks = survivors
        return [t for t in self.tracks if t.miss == 0]


# ------------------------------------------------------------------ zones
def zone_contains(zone: list[float] | None, x: float, y: float) -> bool:
    if not zone or len(zone) != 4:
        return False
    x0, y0, x1, y1 = zone
    return x0 <= x <= x1 and y0 <= y <= y1


class ZoneAnalyzer:
    """Occupant-based, direction-aware door logic.

    The analyzer owns per-person room state; the queue engine owns tokens.
    Callbacks (wired by the worker):

      on_entry(occ)     occ = {key, t_in, ghost, src} - confirmed entry;
                        worker asks the engine for a token and stores it in
                        occ["token_id"]
      on_exit(key, occ) occupant left the zone (latched) or vanished
      on_passby(dwell)  person crossed the zone but left before ENTRY_DWELL
      on_note(msg)      reason-coded log lines (SEAM_HOLD / REBOUND)
      on_crowd(f, w, r) per-frame crowd numbers

    Rules:
    * ENTRY needs an outside sighting first, then ENTRY_DWELL seconds of
      continuous presence inside the doctor zone.
    * People already inside when tracking starts (or never seen walking in)
      become GHOST occupants - they never take a token, and their exit is
      logged as EXIT_GHOST.
    * EXIT needs ROOM_LATCH_FRAMES consecutive outside frames, or the track
      vanishing entirely.  Exits are per occupant, so in a swap (one person
      leaving while another is still inside) only the leaver is affected.
    * A REAL seam (source switch / jump-cut loop) freezes events for
      SEAM_HOLD seconds, then re-binds occupants to whatever the camera
      sees: linked occupants first, extra people become ghosts, and if fewer
      people are visible than occupants the unmatched ones stay bound until
      an exit is actually seen.
    """

    def __init__(self, on_entry, on_exit, on_passby, on_note, on_crowd,
                 now_fn=time.time):
        self.on_entry = on_entry
        self.on_exit = on_exit
        self.on_passby = on_passby
        self.on_note = on_note
        self.on_rebound = None          # optional: (mapping, new_ghost_keys)
        self.on_crowd = on_crowd
        self.now = now_fn
        self.wait_zone: list[float] | None = None
        self.doc_zone: list[float] | None = None
        # key -> {key, t_in, ghost, token_id, src, out_run}
        self.occupants: dict = {}
        # pending entry candidates (seen outside, dwell not yet reached)
        self._pending: dict = {}
        self._seen_outside: dict = {}
        self._hold_until: float | None = None
        self._pending_rebind = False

    # ------------------------------------------------------------- lifecycle
    def set_zones(self, wait_zone, doc_zone) -> None:
        self.wait_zone = wait_zone
        self.doc_zone = doc_zone

    def begin_seam(self) -> None:
        """Real seam: hold events, then re-bind (occupants are kept)."""
        self._hold_until = self.now() + config.SEAM_HOLD
        self._pending_rebind = True
        self._pending = {}
        if callable(self.on_note):
            self.on_note(f"[SEAM_HOLD] source seam detected — holding events "
                         f"for {config.SEAM_HOLD:.1f} s")

    def reset_state(self) -> None:
        self.begin_seam()

    def _frozen(self) -> bool:
        return self._hold_until is not None and self.now() < self._hold_until

    def track_vanished(self, key) -> None:
        """The tracker dropped this track entirely (miss budget exhausted)."""
        if self._frozen():
            return              # a seam reset kills all tracks; ignore those
        occ = self.occupants.pop(key, None)
        if occ is not None:
            if callable(self.on_exit):
                self.on_exit(key, occ)
            return
        p = self._pending.pop(key, None)
        if p is not None and callable(self.on_passby):
            self.on_passby(self.now() - p["since"])

    # ------------------------------------------------------------- per frame
    def process(self, tracks: list[Track]) -> dict:
        now = self.now()
        in_frame = len(tracks)
        in_waiting = 0
        in_room = 0
        inside_ids: list = []
        for t in tracks:
            feet_y = t.cy + (t.bbox[3] - t.bbox[1]) * 0.30
            t.in_doc = zone_contains(self.doc_zone, t.cx, feet_y)
            # a person in the doctor room is not 'waiting' any more
            t.in_wait = (not t.in_doc) and zone_contains(self.wait_zone,
                                                         t.cx, feet_y)
            if t.in_doc:
                in_room += 1
                inside_ids.append(t.tid)
            else:
                self._seen_outside[t.tid] = now
            if t.in_wait and t.hits >= 2:
                in_waiting += 1

        if len(self._seen_outside) > 500:      # keep memory bounded
            newest = sorted(self._seen_outside.items(),
                            key=lambda kv: kv[1])[-250:]
            self._seen_outside = dict(newest)

        if self._frozen():
            if callable(self.on_crowd):
                self.on_crowd(in_frame, in_waiting, in_room)
            return {"frame": in_frame, "waiting": in_waiting,
                    "room": in_room, "occupants": len(self.occupants)}

        if self._pending_rebind:
            self._rebind(inside_ids)

        for t in tracks:
            if not t.in_doc:
                occ = self.occupants.get(t.tid)
                if occ is not None:
                    occ["out_run"] += 1
                    if occ["out_run"] >= config.ROOM_LATCH_FRAMES:
                        self.occupants.pop(t.tid, None)
                        if callable(self.on_exit):
                            self.on_exit(t.tid, occ)
                    continue
                p = self._pending.get(t.tid)
                if p is not None:
                    p["out_run"] += 1
                    if p["out_run"] >= config.ROOM_LATCH_FRAMES:
                        self._pending.pop(t.tid, None)
                        if callable(self.on_passby):
                            self.on_passby(now - p["since"])
                continue

            # --- track is inside the doctor zone ---------------------------
            occ = self.occupants.get(t.tid)
            if occ is not None:
                occ["out_run"] = 0
                continue
            p = self._pending.get(t.tid)
            if p is not None:
                p["out_run"] = 0
                if now - p["since"] >= config.ENTRY_DWELL:
                    self._pending.pop(t.tid, None)
                    occ = {"key": t.tid, "t_in": p["since"], "ghost": False,
                           "src": "camera", "out_run": 0, "token_id": None}
                    self.occupants[t.tid] = occ
                    if callable(self.on_entry):
                        self.on_entry(occ)
                continue
            # brand-new track inside without ever being seen outside -> ghost
            if t.tid in self._seen_outside:
                self._pending[t.tid] = {"since": now, "out_run": 0}
            else:
                self._add_ghost(t.tid, now)

        if callable(self.on_crowd):
            self.on_crowd(in_frame, in_waiting, in_room)
        return {"frame": in_frame, "waiting": in_waiting,
                "room": in_room, "occupants": len(self.occupants)}

    # ------------------------------------------------------------------ bits
    def _add_ghost(self, key, t_in: float) -> None:
        occ = {"key": key, "t_in": t_in, "ghost": True, "src": "camera",
               "out_run": 0, "token_id": None}
        self.occupants[key] = occ
        if callable(self.on_entry):
            self.on_entry(occ)

    def _rebind(self, inside_ids: list) -> None:
        """Snapshot re-bind after a seam: linked occupants first, extras
        become ghosts, deficits stay bound until an exit is actually seen."""
        self._pending_rebind = False
        occ_list = sorted(self.occupants.values(),
                          key=lambda o: (o.get("token_id") is None,
                                         o.get("t_in") or 0))
        new_dict: dict = {}
        mapping: list = []
        n = min(len(inside_ids), len(occ_list))
        for i in range(n):
            occ = occ_list[i]
            old_key = occ["key"]
            occ["key"] = inside_ids[i]
            occ["out_run"] = 0
            new_dict[inside_ids[i]] = occ
            mapping.append((old_key, inside_ids[i]))
        extras = 0
        for key in inside_ids[n:]:
            new_dict[key] = {"key": key, "t_in": self.now(), "ghost": True,
                             "src": "camera", "out_run": 0, "token_id": None}
            extras += 1
        kept = len(occ_list) - n
        for j, occ in enumerate(occ_list[n:]):   # kept, but their track is gone
            occ["key"] = f"lost-{j}"
            new_dict[f"lost-{j}"] = occ
        self._pending = {}
        self.occupants = new_dict
        if callable(self.on_note):
            self.on_note(f"[REBOUND] after seam: {n} occupant(s) re-bound, "
                         f"{extras} new ghost(s), {kept} kept without a track")
        if callable(self.on_rebound):
            self.on_rebound(mapping, list(inside_ids[n:]))


# ------------------------------------------------------------------ detector
class PersonDetector:
    """Wraps ultralytics YOLO11n; loads once, thread-safe predict()."""

    def __init__(self, weights: str | None = None, conf: float = config.DEFAULT_CONF):
        from ultralytics import YOLO   # deferred: heavy import
        import torch
        self.torch = torch
        self.model = YOLO(weights or str(config.WEIGHTS_PATH))
        self.conf = conf
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, frame, imgsz: int = 640) -> list[dict]:
        """frame: BGR ndarray. Returns normalised detections (cx, cy, bbox)."""
        res = self.model.predict(frame, conf=self.conf,
                                 classes=[config.PERSON_CLASS],
                                 device=self.device, verbose=False,
                                 imgsz=imgsz)[0]
        h, w = res.orig_shape
        out = []
        for box in res.boxes:
            x0, y0, x1, y1 = (float(v) for v in box.xyxy[0])
            out.append({
                "cx": (x0 + x1) / 2.0 / w,
                "cy": (y0 + y1) / 2.0 / h,
                "bbox": (x0 / w, y0 / h, x1 / w, y1 / h),
            })
        return out
