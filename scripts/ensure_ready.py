"""First-run checks for SmartOPD.

Run by run.bat (double-click).  Makes sure:
  * the local .venv exists (created with py -3.11)
  * requirements are installed inside it
  * models/yolo11n.pt weights are present (downloads on first run)

Every cache/config path is pinned INSIDE the project folder.
Exits non-zero on failure with a readable message.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_DIR = ROOT / "app"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
WEIGHTS = ROOT / "models" / "yolo11n.pt"

os.environ["YOLO_CONFIG_DIR"] = str(ROOT / ".ultralytics")
os.environ["TORCH_HOME"] = str(ROOT / ".torch")
os.environ["MPLCONFIGDIR"] = str(ROOT / ".mpl")
os.environ.pop("PIP_USER", None)
os.environ["PIP_CACHE_DIR"] = str(ROOT / ".pip-cache")


def die(msg: str) -> None:
    print(f"[ensure_ready] ERROR: {msg}")
    sys.exit(1)


def main() -> None:
    for d in (".tmp", ".pip-cache", ".ultralytics", ".torch", ".mpl",
              "models", "samples", "data", "data/uploads"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    if not VENV_PY.exists():
        print("[ensure_ready] creating local .venv (py -3.11) ...")
        r = subprocess.run(["py", "-3.11", "-m", "venv", str(ROOT / ".venv")])
        if r.returncode != 0:
            die("could not create .venv - install Python 3.11 from python.org "
                "and make sure the 'py' launcher is available.")

    # deps present? (python-multipart renamed its import in newer versions)
    probe = subprocess.run(
        [str(VENV_PY), "-c",
         "try:\n"
         "    import fastapi, uvicorn, ultralytics\n"
         "    import python_multipart\n"
         "except ImportError:\n"
         "    import fastapi, uvicorn, ultralytics, multipart\n"],
        capture_output=True, text=True)
    if probe.returncode != 0:
        print("[ensure_ready] installing requirements into .venv ...")
        r = subprocess.run([str(VENV_PY), "-m", "pip", "install", "--no-user",
                            "-r", str(ROOT / "requirements.txt")])
        if r.returncode != 0:
            die("pip install failed - check your internet connection.")

    # weights present?
    if not WEIGHTS.exists():
        print("[ensure_ready] downloading YOLO11n weights (6 MB) ...")
        r = subprocess.run(
            [str(VENV_PY), "-c",
             "from ultralytics import YOLO; YOLO(r'%s')" % str(WEIGHTS)],
            capture_output=True, text=True)
        if r.returncode != 0 or not WEIGHTS.exists():
            die("could not download yolo11n.pt - check your internet connection.")

    print("[ensure_ready] environment OK - starting SmartOPD ...")


if __name__ == "__main__":
    main()
