"""Headless proof: run the REAL pipeline over pingpong.mp4 for 3 full loops.

    python scripts/replay_check.py

Seeds tokens #1..#6 as waiting, replays the clip frame by frame (YOLO +
tracker + zone analyzer + queue engine, video-time clock), prints the event
timeline and asserts:

  1. every token completes (all six served)
  2. tokens complete in order (1 before 2 before ...)
  3. no token ever moves backwards (never 'in' -> 'called'/'waiting')
  4. the ETA model learns from every consultation
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
sys.path.insert(0, str(APP))

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
os.environ["TORCH_HOME"] = str(ROOT / ".torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".mpl")

import cv2  # noqa: E402

import config  # noqa: E402
from queue_manager import QueueEngine  # noqa: E402
from vision import CentroidTracker, PersonDetector, ZoneAnalyzer  # noqa: E402

CLIP = ROOT / "samples" / "pingpong.mp4"
STATE = ROOT / "data" / "state.json"
LOOPS = 3
IMGSZ = 480          # a bit smaller = faster replay, same detections


class VideoClock:
    """Monotonic video-time clock (seconds) - deterministic, no wall time."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def load_zones() -> tuple[list, list]:
    if not STATE.exists():
        sys.exit("no data/state.json - load the clip once in the UI first")
    state = json.loads(STATE.read_text(encoding="utf-8"))
    by_src = state.get("zones_by_source") or {}
    z = by_src.get(str(CLIP)) or state.get("zones") or {}
    if not z.get("wait") or not z.get("doc"):
        sys.exit("pingpong.mp4 has no saved zones - draw WAITING + DOCTOR "
                 "once on the monitor screen, then rerun")
    return z["wait"], z["doc"]


def main() -> int:
    if not CLIP.exists():
        sys.exit("samples/pingpong.mp4 missing - run scripts/make_pingpong.py")
    wait_zone, doc_zone = load_zones()

    clock = VideoClock()
    engine = QueueEngine(now_fn=clock)
    engine.time_scale = config.TIME_SCALE
    engine.auto_advance = True
    names = ["Asha Patil", "Rohan Mehta", "Sneha Kulkarni",
             "Imran Shaikh", "Priya Nair", "Vikram Rao"]
    for n in names:
        engine.register(n)

    def on_entry(occ):
        occ["token_id"] = engine.begin_occupant(occ["key"], occ["t_in"],
                                                occ["ghost"], src="camera")

    def on_exit(key, occ):
        engine.remove_occupant(key, src="camera")

    analyzer = ZoneAnalyzer(
        on_entry=on_entry, on_exit=on_exit,
        on_passby=engine.pass_by, on_note=engine.note,
        on_crowd=lambda f, w, r: engine.update_crowd(f, w, r),
        now_fn=clock)
    analyzer.set_zones(wait_zone, doc_zone)

    detector = PersonDetector()
    tracker = CentroidTracker()

    cap = cv2.VideoCapture(str(CLIP))
    if not cap.isOpened():
        sys.exit("cannot open pingpong.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    print(f"replaying {CLIP.name}: {LOOPS} loops, {len(names)} tokens waiting")
    print(f"zones wait={wait_zone} doc={doc_zone}")
    print("-" * 78)

    frames_total = 0
    loops_done = 0
    drain_start_frame: int | None = None
    loop_frames = 0
    marks = []
    while True:
        ok, frame = cap.read()
        if not ok:
            loops_done += 1
            if loops_done < LOOPS:
                marks.append((loops_done, clock.t))
                print(f"[loop] wrap {loops_done} at t={clock.t:6.1f}s "
                      f"(continuous - tracker kept, occupants kept)")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # continuous: no seam
                continue
            # spec allows a drain: its own event pattern ends every half-loop
            # with an ENTRY, so the last token completes just after 3 loops
            pending = [t for t in engine.tokens
                       if t.status in ("waiting", "called", "in")]
            if not pending:
                break
            if drain_start_frame is None:
                drain_start_frame = frames_total
                loop_frames = 2 * int(frames_total / max(1, loops_done))
                print(f"[loop] drain: {len(pending)} token(s) still open "
                      f"({', '.join('#' + str(t.no) for t in pending)}) "
                      f"- continuing up to one extra loop")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            if frames_total - drain_start_frame >= loop_frames:
                print("[loop] drain limit reached")
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames_total += 1
        clock.t = frames_total / fps
        dets = detector.detect(frame, imgsz=IMGSZ)
        visible = tracker.update(dets)
        for gone in tracker.expired:
            analyzer.track_vanished(gone.tid)
        analyzer.process(visible)
    cap.release()

    # ------------------------------------------------------------- report
    print("-" * 78)
    print("EVENT TIMELINE (oldest first)")
    for e in reversed(engine.events):
        mm, ss = divmod(int(e["t"]), 60)
        print(f"  [{mm:02d}:{ss:02d}] {e['m']}")
    print("-" * 78)

    print("TOKENS")
    for t in engine.tokens:
        dur = (t.t_out - t.t_in) if (t.t_out and t.t_in) else None
        dur_txt = f"{dur:5.1f}s real = {dur * engine.time_scale / 60:4.1f} min" \
            if dur else "—"
        print(f"  #{t.no} {t.name:<15} {t.status:<9} {dur_txt}   "
              f"history={'>'.join(t.history)}")
    print(f"\nserved={engine.stats()['served']}/6 · "
          f"learned={len(engine.durations)} · "
          f"avg={engine.avg_seconds() / 60:.1f} min (model)")

    # ---------------------------------------------------------- assertions
    failures = []
    done = [t for t in engine.tokens if t.status == "done"]
    if len(done) != len(names):
        failures.append(f"only {len(done)}/{len(names)} tokens completed")

    if len(done) == len(names):
        outs = [t.t_out for t in sorted(engine.tokens, key=lambda x: x.no)]
        if outs != sorted(outs):
            failures.append("tokens did not complete in order")

    for t in engine.tokens:
        hist = t.history
        if "in" in hist:
            after = hist[hist.index("in") + 1:]
            if any(s in ("called", "waiting") for s in after):
                failures.append(f"token #{t.no} moved backwards: {hist}")

    if len(engine.durations) != len(done):
        failures.append(f"only {len(engine.durations)}/{len(done)} "
                        f"consultations trained the ETA model")

    print()
    if failures:
        print("REPLAY FAIL")
        for f in failures:
            print("  ✗", f)
        return 1
    print("REPLAY PASS - all tokens completed in order, never backwards, "
          "every consultation learned")
    return 0


if __name__ == "__main__":
    sys.exit(main())
