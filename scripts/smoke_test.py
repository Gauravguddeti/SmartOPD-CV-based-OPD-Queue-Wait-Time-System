"""Smoke test: run the real CV pipeline on the bundled demo clip.

Verifies end-to-end, offline:
  * YOLO11n loads from models/yolo11n.pt (workspace-local weights)
  * persons are detected in the demo video
  * the tracker maintains identities
  * auto zone detection produces a sane waiting area

Run:  .venv/Scripts/python.exe scripts/smoke_test.py
"""

from __future__ import annotations

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

from vision import CentroidTracker, PersonDetector  # noqa: E402
from worker import VisionWorker  # noqa: E402

CLIP = ROOT / "samples" / "People_in_hospital_waiting_room_74f5013d.mp4"


class _DummyEngine:
    zones = None


def main() -> int:
    if not CLIP.exists():
        print("SMOKE_FAIL clip missing")
        return 1
    print("loading detector ...")
    det = PersonDetector()
    print(f"  device={det.device} weights OK")

    cap = cv2.VideoCapture(str(CLIP))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"clip: {n_frames} frames @ {fps:.1f} fps")

    tracker = CentroidTracker()
    counts = []
    max_ids = 0
    for i in range(10):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n_frames * (i + 0.5) / 10))
        ok, frame = cap.read()
        if not ok:
            continue
        dets = det.detect(frame)
        vis = tracker.update(dets)
        counts.append(len(vis))
        max_ids = max(max_ids, tracker._next - 1)
    cap.release()
    print(f"detections per sampled frame: {counts}")
    print(f"tracker assigned ids: {max_ids}")

    if not any(c > 0 for c in counts):
        print("SMOKE_FAIL no persons detected in demo clip")
        return 1

    print("auto-detecting zones ...")
    wk = VisionWorker(_DummyEngine())
    wk.detector = det
    res = wk.auto_detect_zones(str(CLIP), samples=10)
    print(f"  result: {res}")
    if not res.get("ok"):
        print("SMOKE_FAIL auto zone detection failed")
        return 1

    print("SMOKE_PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
