"""SmartOPD queue engine (occupant-based room model).

Room model
----------
``room.occupants`` is a list of ``{track_id, token_id, t_in, ghost, src}``:

* one entry per person currently inside the doctor zone;
* ``token_id=None`` occupants are GHOSTS (people who were already inside when
  tracking started / after a seam, or walk-ins when no token was available);
* the camera may only move tokens ``waiting/called -> in -> done`` - never
  backwards.  Pass-by protection happens BEFORE admission (a person who stays
  inside less than ``ENTRY_DWELL`` never becomes an occupant at all).

Time scale
----------
``self.time_scale`` converts real seconds into MODEL seconds before anything
enters the ETA model, so a 10-second demo clip yields realistic minutes
(TIME_SCALE=60).  Webcam sources use 1.0.

All public methods take the engine lock; the vision worker thread and the
FastAPI handlers call in concurrently.
"""

from __future__ import annotations

import dataclasses
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass

import config

TOKEN_STATUSES = ("waiting", "called", "in", "done", "skipped", "noshow")


class QueueError(Exception):
    """Raised for invalid queue operations (bad token no, empty queue, ...)."""


@dataclass
class Token:
    no: int
    name: str
    kind: str                    # "New visit" | "Follow-up"
    priority: bool
    status: str = "waiting"
    t_reg: float = 0.0
    t_call: float | None = None
    t_in: float | None = None
    t_out: float | None = None
    entered_by: str = ""         # camera | manual | simulation | nurse
    history: list = dataclasses.field(default_factory=list)   # status journal

    def set_status(self, new: str) -> None:
        self.status = new
        self.history.append(new)


class QueueEngine:
    def __init__(self, now_fn=time.time):
        self.now = now_fn
        self._lock = threading.RLock()
        self.version = 0          # bumped on every mutation (for the saver)
        self.tokens: list[Token] = []
        self.next_no = 1
        self.durations: list[float] = []   # MODEL seconds (real x time_scale)
        self.room: dict = {"occupants": [], "scale": config.TIME_SCALE}
        self.events: list[dict] = []
        self.peak_crowd = 0
        self.crowd = {"frame": 0, "waiting": 0, "room": 0}
        self.auto_advance = True  # camera admits the next token on its own
        self.time_scale = config.TIME_SCALE
        self._manual_seq = 0

    # ------------------------------------------------------------------ util
    def _bump(self) -> None:
        self.version += 1

    def _log(self, message: str) -> None:
        self.events.insert(0, {"t": self.now(), "m": message})
        del self.events[80:]

    def note(self, message: str) -> None:
        """Log a reason-coded line coming from the vision side."""
        with self._lock:
            self._log(message)
            self._bump()

    # ------------------------------------------------------------------ state
    def to_dict(self) -> dict:
        with self._lock:
            occ = [dict(o) for o in self.room["occupants"]]
            linked = [o for o in occ if o.get("token_id") is not None]
            oldest = min((o for o in occ), key=lambda o: o.get("t_in") or 0,
                         default=None)
            return {
                "tokens": [asdict(t) for t in self.tokens],
                "next_no": self.next_no,
                "durations": [round(d, 1) for d in self.durations[-200:]],
                "room": {
                    "occupants": occ,
                    "scale": self.time_scale,
                    # convenience fields so the screens stay backward-compatible
                    "no": linked[0]["token_id"] if linked else None,
                    "t_in": oldest["t_in"] if oldest else None,
                    "src": oldest["src"] if oldest else None,
                },
                "events": self.events,
                "peak_crowd": self.peak_crowd,
                "auto_advance": self.auto_advance,
            }

    @classmethod
    def from_dict(cls, data: dict, now_fn=time.time) -> "QueueEngine":
        eng = cls(now_fn=now_fn)
        try:
            fields = {f.name for f in dataclasses.fields(Token)}
            eng.tokens = []
            for t in data.get("tokens", []):
                t = {k: v for k, v in t.items() if k in fields}
                tok = Token(**t)
                if not tok.history:
                    tok.history = [tok.status]
                eng.tokens.append(tok)
            eng.next_no = int(data.get("next_no", len(eng.tokens) + 1))
            eng.durations = [float(d) for d in data.get("durations", [])]
            room = data.get("room") or {}
            if isinstance(room, dict) and isinstance(room.get("occupants"), list):
                eng.room["occupants"] = [o for o in room["occupants"]
                                         if isinstance(o, dict)]
            eng.events = list(data.get("events", []))
            eng.peak_crowd = int(data.get("peak_crowd", 0))
            eng.auto_advance = bool(data.get("auto_advance", True))
        except (TypeError, ValueError, AttributeError):
            eng = cls(now_fn=now_fn)
        eng.version += 1
        return eng

    # ------------------------------------------------------------------ queue
    def queue_order(self) -> list[Token]:
        """Called patients first, then priority, then registration order."""
        with self._lock:
            called = sorted(
                (t for t in self.tokens if t.status == "called"),
                key=lambda t: (t.t_call or 0.0, t.no),
            )
            waiting = sorted(
                (t for t in self.tokens if t.status == "waiting"),
                key=lambda t: (not t.priority, t.no),
            )
            return called + waiting

    def register(self, name: str, kind: str = "New visit",
                 priority: bool = False) -> Token:
        name = (name or "").strip()
        if not name:
            raise QueueError("Patient name is required.")
        with self._lock:
            tok = Token(no=self.next_no, name=name, kind=kind,
                        priority=bool(priority), t_reg=self.now())
            tok.set_status("waiting")
            self.next_no += 1
            self.tokens.append(tok)
            self._log(f"Token #{tok.no} issued to {tok.name} ({tok.kind}"
                      + (" · priority" if tok.priority else "") + ")")
            self._bump()
            return tok

    def _find(self, no: int) -> Token:
        for t in self.tokens:
            if t.no == no:
                return t
        raise QueueError(f"Token #{no} not found.")

    def call_next(self) -> Token:
        with self._lock:
            for t in self.queue_order():
                if t.status == "waiting":
                    return self.call_token(t.no)
            raise QueueError("No patients waiting.")

    def call_token(self, no: int) -> Token:
        with self._lock:
            t = self._find(no)
            if t.status not in ("waiting", "called"):
                raise QueueError(f"Token #{no} cannot be called (status: {t.status}).")
            if t.status == "waiting":
                t.set_status("called")
                t.t_call = self.now()
                self._log(f"Token #{t.no} ({t.name}) called")
                self._bump()
            return t

    def set_status(self, no: int, action: str) -> Token:
        with self._lock:
            t = self._find(no)
            if action == "skip" and t.status in ("waiting", "called"):
                t.set_status("skipped")
            elif action == "noshow" and t.status in ("waiting", "called"):
                t.set_status("noshow")
            elif action == "requeue" and t.status in ("skipped", "noshow"):
                t.set_status("waiting")
            elif action == "call":
                return self.call_token(no)
            else:
                raise QueueError(f"Cannot '{action}' token #{no} (status: {t.status}).")
            self._log(f"Token #{t.no} ({t.name}) → {t.status}")
            self._bump()
            return t

    # -------------------------------------------------- room: camera events
    def begin_occupant(self, key, t_in: float, ghost: bool,
                       src: str = "camera") -> int | None:
        """An entry was confirmed inside the doctor zone.

        Linking order: a called token first, else (auto-advance on) the next
        waiting token, else the occupant is recorded unlinked.  Ghosts (people
        already inside when tracking started / after a seam) never take a
        token.  Tokens only ever move waiting/called -> in here.

        Returns the linked token number (None for ghosts/unlinked).
        """
        with self._lock:
            if any(o["track_id"] == key for o in self.room["occupants"]):
                return None
            if not t_in:
                t_in = self.now()
            t_in = min(t_in, self.now())
            token_id = None
            auto = False
            cand = None
            if not ghost:
                order = self.queue_order()
                cand = next((t for t in order if t.status == "called"), None)
                if cand is None and self.auto_advance:
                    cand = next((t for t in order if t.status == "waiting"), None)
                    if cand is not None:
                        cand.set_status("called")
                        cand.t_call = t_in
                        auto = True
                if cand is not None:
                    cand.set_status("in")
                    cand.t_in = t_in
                    cand.entered_by = src
                    token_id = cand.no
            self.room["occupants"].append({"track_id": key,
                                           "token_id": token_id,
                                           "t_in": t_in,
                                           "ghost": ghost,
                                           "src": src})
            if ghost:
                self._log(f"[ENTRY_CONFIRMED] person already inside the room "
                          f"zone → ghost occupant (no token)")
            elif token_id:
                how = "admitted automatically" if auto else "(called by nurse)"
                self._log(f"[ENTRY_CONFIRMED] token #{cand.no} ({cand.name}) "
                          f"in consultation {how}")
            else:
                self._log(f"[ENTRY_CONFIRMED] walk-in recorded with no patient "
                          f"in the queue (unlinked occupant)")
            self._bump()
            return token_id

    def remove_occupant(self, key, src: str = "camera") -> bool:
        """An occupant left the zone (or vanished at the door).

        Linked occupant  -> token DONE, duration measured (model seconds).
        Ghost / unlinked -> removed silently (EXIT_GHOST).  Another person
        leaving never touches a token that belongs to someone still inside,
        because exits are matched per occupant.
        """
        with self._lock:
            occ = next((o for o in self.room["occupants"]
                        if o["track_id"] == key), None)
            if occ is None:
                return False
            self.room["occupants"].remove(occ)
            now = self.now()
            dur_real = max(0.0, now - (occ.get("t_in") or now))
            if occ.get("token_id") is not None:
                t = next((x for x in self.tokens if x.no == occ["token_id"]), None)
                if t and t.status == "in":
                    t.set_status("done")
                    t.t_out = now
                dur_model = dur_real * self.time_scale
                learn = dur_real >= config.MIN_LEARN_SECONDS
                if learn:
                    self.durations.append(dur_model)
                extra = "" if learn else " (too short to learn from)"
                who = f"token #{t.no}" if t else "token"
                self._log(f"[EXIT_LINKED] Room free via {src} · {who} done · "
                          f"consultation {dur_model / 60:.1f} min{extra}")
            else:
                self._log(f"[EXIT_GHOST] person left the doctor zone via {src} "
                          f"(no token linked)")
            self._bump()
            return True

    def remap_occupants(self, mapping: list, new_ghosts: list) -> None:
        """After a seam: re-bind occupant mirror entries to fresh track keys
        and register extra people seen inside the zone as new ghosts."""
        with self._lock:
            for old, new in mapping:
                occ = next((o for o in self.room["occupants"]
                            if o["track_id"] == old), None)
                if occ is not None:
                    occ["track_id"] = new
            for new in new_ghosts:
                if any(o["track_id"] == new for o in self.room["occupants"]):
                    continue
                self.room["occupants"].append({"track_id": new,
                                               "token_id": None,
                                               "t_in": self.now(),
                                               "ghost": True,
                                               "src": "camera"})
            self._bump()

    def pass_by(self, dwell_s: float) -> None:
        self.note(f"[PASS_BY] person at the door left after {dwell_s:.1f} s "
                  f"— no token touched")

    def seam_hold(self, seconds: float) -> None:
        self.note(f"[SEAM_HOLD] source seam detected — holding events "
                  f"for {seconds:.1f} s")

    # -------------------------------------------------- room: manual overrides
    def manual_enter(self, src: str = "manual") -> int | None:
        with self._lock:
            self._manual_seq += 1
            return self.begin_occupant(f"manual-{self._manual_seq}",
                                       self.now(), False, src=src)

    def manual_free(self, src: str = "manual") -> bool:
        """Complete the oldest token-bearing occupant; clear ghosts if none."""
        with self._lock:
            occs = self.room["occupants"]
            linked = [o for o in occs if o.get("token_id") is not None]
            if linked:
                occ = min(linked, key=lambda o: o.get("t_in") or 0)
                return self.remove_occupant(occ["track_id"], src=src)
            ghosts = [o for o in occs if o.get("ghost")]
            if ghosts:
                for g in ghosts:
                    self.remove_occupant(g["track_id"], src=src)
                return True
            self._log(f"Room free via {src} · room was already empty")
            return False

    def undo_last(self) -> Token:
        """Undo the most recent completed consultation (camera mistakes)."""
        with self._lock:
            done = [t for t in self.tokens
                    if t.status == "done" and t.t_in and t.t_out]
            if not done:
                raise QueueError("No completed consultation to undo.")
            t = max(done, key=lambda x: x.t_out or 0.0)
            dur_real = (t.t_out - t.t_in) / max(1e-6, self.time_scale)
            for i, d in enumerate(self.durations):
                if abs(d - dur_real * self.time_scale) < 1.0:
                    del self.durations[i]
                    break
            t.set_status("waiting")
            t.t_in = None
            t.t_out = None
            self._log(f"Undo · token #{t.no} ({t.name}) returned to the queue")
            self._bump()
            return t

    def room_state(self) -> bool:
        with self._lock:
            return bool(self.room["occupants"])

    def occupant_tag(self, track_id) -> str:
        """Label for the debug overlay: which token is this track carrying?"""
        with self._lock:
            for o in self.room["occupants"]:
                if o["track_id"] == track_id:
                    return (f"token #{o['token_id']}"
                            if o.get("token_id") is not None else "ghost")
            return ""

    # -------------------------------------------------------------- analytics
    def avg_seconds(self) -> float:
        """Adaptive consultation length in MODEL seconds: 8-min prior blended
        into the rolling median of the last LEARN_WINDOW consultations."""
        with self._lock:
            d = sorted(self.durations[-config.LEARN_WINDOW:])
            if not d:
                return config.PRIOR_SECONDS
            med = statistics.median(d)
            w = min(1.0, len(d) / config.LEARN_RAMP)
            return config.PRIOR_SECONDS * (1 - w) + med * w

    def eta(self, no: int) -> dict | None:
        with self._lock:
            t = self._find(no)
            if t.status not in ("waiting", "called"):
                return None
            order = self.queue_order()
            ahead = next(i for i, x in enumerate(order) if x.no == no)
            avg = self.avg_seconds()                      # model seconds
            rem = 0.0
            linked = [o for o in self.room["occupants"]
                      if o.get("token_id") is not None]
            if linked:
                oldest = min(linked, key=lambda o: o.get("t_in") or 0)
                elapsed_model = max(0.0, self.now() - oldest["t_in"]) * self.time_scale
                rem = max(60.0, avg - elapsed_model)
            mid = rem + ahead * avg                       # model seconds
            return {
                "ahead": ahead,
                "mid_s": round(mid, 1),
                "lo_s": round(mid * config.ETA_LO_FACTOR, 1),
                "hi_s": round(mid * config.ETA_HI_FACTOR, 1),
                # wall-clock estimate: convert model seconds back to real time
                "expected_at": self.now() + mid / max(1e-6, self.time_scale),
            }

    def update_crowd(self, frame: int, waiting: int, room: int = 0) -> None:
        with self._lock:
            self.crowd = {"frame": int(frame), "waiting": int(waiting),
                          "room": int(room)}
            self.peak_crowd = max(self.peak_crowd, int(frame))
            self._bump()

    def stats(self) -> dict:
        with self._lock:
            served = sum(1 for t in self.tokens if t.status == "done")
            waiting = sum(1 for t in self.tokens if t.status == "waiting")
            called = sum(1 for t in self.tokens if t.status == "called")
            return {
                "served": served,
                "waiting": waiting,
                "called": called,
                "avg_min": _mins(self.avg_seconds()),
                "learned": len(self.durations),
                "peak_crowd": self.peak_crowd,
                "crowd": self.crowd,
                "auto_advance": self.auto_advance,
                "time_scale": self.time_scale,
                "occupants": len(self.room["occupants"]),
            }

    # ------------------------------------------------------------------ misc
    def seed_demo(self) -> list[Token]:
        names = [("Asha Patil", "New visit"), ("Rohan Mehta", "Follow-up"),
                 ("Sneha Kulkarni", "New visit"), ("Imran Shaikh", "Follow-up"),
                 ("Priya Nair", "New visit")]
        out = []
        for n, k in names:
            out.append(self.register(n, k))
        return out

    def reset(self) -> None:
        with self._lock:
            self.tokens = []
            self.next_no = 1
            self.durations = []
            self.room = {"occupants": [], "scale": self.time_scale}
            self.events = []
            self.peak_crowd = 0
            self._log("Queue, history and learned times reset")
            self._bump()

    def clear_room(self) -> None:
        """Drop every occupant (used after a queue reset so the vision side
        re-ghosts whoever is still inside on the next frames)."""
        with self._lock:
            self.room = {"occupants": [], "scale": self.time_scale}
            self._log("Room cleared")
            self._bump()

    def csv_text(self) -> str:
        import csv as _csv
        import io as _io

        with self._lock:
            buf = _io.StringIO()
            w = _csv.writer(buf)
            w.writerow(["token", "patient", "type", "priority", "status",
                        "registered", "called", "consultation_start",
                        "consultation_end", "duration_min", "entered_by"])
            for t in sorted(self.tokens, key=lambda x: x.no):
                dur = (t.t_out - t.t_in) / 60.0 if t.t_out and t.t_in else ""
                w.writerow([
                    t.no, t.name, t.kind, "yes" if t.priority else "no", t.status,
                    _hhmmss(t.t_reg), _hhmmss(t.t_call), _hhmmss(t.t_in),
                    _hhmmss(t.t_out),
                    round(dur, 1) if dur != "" else "",
                    t.entered_by,
                ])
            return buf.getvalue()

    def simulate(self) -> dict:
        """Complete one full consultation instantly with a plausible length."""
        with self._lock:
            if not self.queue_order():
                raise QueueError("No patients waiting.")
            if not any(o.get("token_id") for o in self.room["occupants"]):
                self.manual_enter("simulation")
            avg_model = self.avg_seconds()
            dur_model = avg_model * random.uniform(0.6, 1.4)
            dur_real = dur_model / max(1e-6, self.time_scale)
            linked = [o for o in self.room["occupants"]
                      if o.get("token_id") is not None]
            occ = min(linked, key=lambda o: o.get("t_in") or 0)
            occ["t_in"] = self.now() - dur_real     # back-date for honest math
            self.remove_occupant(occ["track_id"], src="simulation")
            self._bump()
            return {"simulated": True, "duration_s": round(dur_real, 1)}


def _mins(seconds: float) -> int:
    return max(0, round(seconds / 60.0))


def _hhmmss(epoch: float | None) -> str:
    if not epoch:
        return ""
    return time.strftime("%H:%M:%S", time.localtime(epoch))
