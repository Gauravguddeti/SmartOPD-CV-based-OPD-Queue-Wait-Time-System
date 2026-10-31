"""Central configuration for SmartOPD.

Importing this module pins every cache/config directory of our heavy
dependencies (ultralytics, torch, matplotlib) to folders INSIDE the project
workspace, so nothing is ever written outside it.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
PAGES_DIR = ROOT / "app" / "pages"
STATIC_DIR = ROOT / "app" / "static"

STATE_PATH = DATA_DIR / "state.json"
WEIGHTS_PATH = MODELS_DIR / "yolo11n.pt"

for _d in (DATA_DIR, MODELS_DIR, SAMPLES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- keep every tool's files inside the workspace -------------------------
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
os.environ.setdefault("TORCH_HOME", str(ROOT / ".torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mpl"))

# ---- queue / wait-time model constants -------------------------------------
PRIOR_SECONDS = 480.0      # 8-min prior (in MODEL seconds) before learning kicks in
LEARN_WINDOW = 12          # rolling window of measured consultations
LEARN_RAMP = 4             # full trust in measured median after N samples
MIN_LEARN_SECONDS = 4.0    # real-seconds: shorter consultations do not train the model
TIME_SCALE = 25.0          # DEMO time factor: clinic-minutes per video-second for FILE
                           # sources (webcam = 1.0).  10-15 s of clip ≈ 4-6 min of clinic
                           # time.  Tunable live from the nurse desk (10-120).
ETA_LO_FACTOR = 0.75
ETA_HI_FACTOR = 1.30

# ---- vision constants -------------------------------------------------------
PERSON_CLASS = 0           # COCO class id for "person"
DEFAULT_CONF = 0.35
DEFAULT_TARGET_FPS = 10
ENTRY_DWELL = 2.0          # real-seconds inside the doctor zone before an entry confirms
ROOM_LATCH_FRAMES = 4      # consecutive empty frames that confirm a leave
TRACK_MAX_MISS = 15        # frames a track may be missing before it expires
SEAM_HOLD = 1.5            # real-seconds to freeze events after a jump-cut seam

# ---- demo / CCTV overlay ----------------------------------------------------
CCTV_OVERLAY = True        # camera label + live timestamp drawn on frames
CAMERA_LABEL = "CAM-01  OPD WAITING AREA"
