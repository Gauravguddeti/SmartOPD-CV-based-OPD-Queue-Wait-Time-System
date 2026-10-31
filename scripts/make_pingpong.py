"""Build a seamless ping-pong demo source: forward + reversed frames.

    python scripts/make_pingpong.py [--src samples/clip_a.mp4]

Output: samples/pingpong.mp4 = every frame of the source forward, then the
same frames in reverse (same resolution/fps, no audio).  Both joins and the
wrap-around are frame-continuous, so the server treats this source as
CONTINUOUS: the tracker is never reset and no SEAM_HOLD is applied at the
loop point - people simply reverse direction and their track IDs survive.

If samples/clip_a.mp4 does not exist, the first other clip in samples/ is
used and a copy is saved as clip_a.mp4 so the name stays stable.
Zones saved for the source clip are inherited by pingpong.mp4 (same camera).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
STATE = ROOT / "data" / "state.json"


def pick_source(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            sys.exit(f"source not found: {p}")
        return p
    named = SAMPLES / "clip_a.mp4"
    if named.exists():
        return named
    for p in sorted(SAMPLES.glob("*.mp4")):
        if p.stem.lower() != "pingpong":
            return p
    sys.exit("no source clip found in samples/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="source clip (default: clip_a.mp4)")
    args = ap.parse_args()

    src = pick_source(args.src)
    if src.name != "clip_a.mp4":
        (SAMPLES / "clip_a.mp4").write_bytes(src.read_bytes())   # stable name

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        sys.exit(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames_jpg: list[bytes] = []
    size = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if size is None:
            h, w = frame.shape[:2]
            size = (w, h)
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        frames_jpg.append(buf.tobytes())
    cap.release()
    if not frames_jpg:
        sys.exit("no frames decoded")

    out_path = SAMPLES / "pingpong.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, size)
    n = len(frames_jpg)
    import numpy as np
    for i in list(range(n)) + list(range(n - 1, -1, -1)):     # forward + reverse
        frame = cv2.imdecode(np.frombuffer(frames_jpg[i], np.uint8),
                             cv2.IMREAD_COLOR)
        writer.write(frame)
    writer.release()

    dur = n / fps
    print(f"[make_pingpong] source      : {src}")
    print(f"[make_pingpong] output      : {out_path}")
    print(f"[make_pingpong] frames      : {n} forward + {n} reverse = {2 * n}")
    print(f"[make_pingpong] fps / length: {fps:.1f} / {2 * dur:.1f} s "
          f"(continuous at both joins)")

    # inherit the source clip's saved zones (same camera framing)
    try:
        state = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}
        by_src = state.get("zones_by_source") or {}
        src_zones = by_src.get(str(src)) or by_src.get(str(src.resolve()))
        if src_zones:
            by_src[str(out_path)] = src_zones
            state["zones_by_source"] = by_src
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=1),
                             encoding="utf-8")
            print(f"[make_pingpong] zones       : inherited from {src.name}")
        else:
            print("[make_pingpong] zones       : none saved for the source - "
                  "draw them once on the monitor screen")
    except (json.JSONDecodeError, OSError) as e:
        print(f"[make_pingpong] zone copy skipped: {e}")


if __name__ == "__main__":
    main()
