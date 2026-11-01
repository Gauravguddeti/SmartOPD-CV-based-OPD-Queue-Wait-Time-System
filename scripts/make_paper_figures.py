"""Generate the vector figures for the IEEE paper (paper/figures/*.pdf).

  fig_architecture.pdf : system architecture block diagram
  fig_fuzzy_sets.pdf   : input + output membership functions of the fuzzy ETA
  fig_fuzzy_curve.pdf  : defuzzified wait vs queue position for 3 lengths
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

import fuzzy  # noqa: E402

OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#1a1a1a"
ACC = "#1f4e79"
FILL = "#eaf1f8"
FILL2 = "#f4f4f4"


def box(ax, x, y, w, h, text, fill=FILL, edge=ACC, fs=7.4, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=0.9, edgecolor=edge, facecolor=fill))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=INK, fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p0, p1, style="-|>", lw=0.9, color=ACC, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=8,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1, shrinkB=1))


def architecture():
    fig, ax = plt.subplots(figsize=(7.1, 2.75))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # source
    box(ax, 0.015, 0.62, 0.115, 0.24, "Camera /\nclip /\nwebcam", FILL2, INK, 7.0)

    # vision worker
    box(ax, 0.155, 0.50, 0.44, 0.44, "", "#ffffff", ACC)
    ax.text(0.375, 0.885, "VisionWorker thread  (~10-16 FPS, CPU)", ha="center",
            va="center", fontsize=7.0, color=ACC, fontweight="bold")
    box(ax, 0.168, 0.655, 0.128, 0.16, "YOLO11n\nperson detector", FILL, ACC, 6.6)
    box(ax, 0.312, 0.655, 0.128, 0.16, "Centroid\ntracker", FILL, ACC, 6.6)
    box(ax, 0.456, 0.655, 0.126, 0.16, "Zone\nanalyzer\n(occupants)", FILL, ACC, 6.3)
    arrow(ax, (0.296, 0.735), (0.312, 0.735))
    arrow(ax, (0.440, 0.735), (0.456, 0.735))
    box(ax, 0.168, 0.525, 0.414, 0.105,
        "annotation: boxes, track ids, occupant tags, zones, CCTV stamp",
        "#fbfbfb", "#9bb7d4", 6.2)

    # engine
    box(ax, 0.625, 0.50, 0.21, 0.44, "", "#ffffff", ACC)
    ax.text(0.73, 0.885, "QueueEngine", ha="center", va="center", fontsize=7.0,
            color=ACC, fontweight="bold")
    box(ax, 0.637, 0.645, 0.186, 0.17, "tokens · occupants\nroom events", FILL, ACC, 6.3)
    box(ax, 0.637, 0.525, 0.186, 0.10, "fuzzy ETA model", "#f2f7ef", "#4f7a3a", 6.3)

    # storage
    box(ax, 0.855, 0.62, 0.125, 0.22, "state.json\n(atomic save)", FILL2, INK, 6.4)
    arrow(ax, (0.835, 0.73), (0.855, 0.73), ls=":")

    # screens stacked below the engine, fed by a JSON bus
    screens = (("(1) Monitoring console", 0.36),
               ("(2) Nurse desk", 0.21),
               ("(3) Patient display", 0.06))
    for label, yy in screens:
        box(ax, 0.625, yy, 0.35, 0.11, label, FILL, ACC, 6.9)
        arrow(ax, (0.607, yy + 0.082), (0.625, yy + 0.082))
    ax.plot([0.607, 0.607], [0.50, 0.115], color=ACC, lw=0.9)
    ax.text(0.66, 0.022, "screens poll  /api/state  over HTTP JSON (1 Hz)",
            fontsize=6.0, color=INK, ha="center", va="center")

    # annotated JPEG feed into the monitoring console
    ax.plot([0.30, 0.30, 0.62], [0.50, 0.395, 0.395], color="#9bb7d4", lw=0.9)
    arrow(ax, (0.62, 0.395), (0.625, 0.395), color="#9bb7d4")
    ax.text(0.312, 0.432, "annotated\nJPEG feed", fontsize=6.0, color="#4d7093",
            ha="left", va="center")

    arrow(ax, (0.130, 0.74), (0.155, 0.74))
    arrow(ax, (0.595, 0.74), (0.625, 0.74))

    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT / "fig_architecture.pdf")
    plt.close(fig)


def membership_sets():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.1, 2.05))

    xs = [i * 0.1 for i in range(0, 401)]          # 0..40 min
    ax1.plot(xs, [fuzzy.m_short(x) for x in xs], color="#c00000", lw=1.1, label="short")
    ax1.plot(xs, [fuzzy.m_medium(x) for x in xs], color="#1f4e79", lw=1.1, label="medium")
    ax1.plot(xs, [fuzzy.m_long(x) for x in xs], color="#4f7a3a", lw=1.1, label="long")
    ax1.set_xlabel("learned consultation length (clinic min)", fontsize=7)
    ax1.set_ylabel("membership", fontsize=7)
    ax1.set_title("(a) input: consultation length", fontsize=7.4)
    ax1.tick_params(labelsize=6.4, length=2)
    ax1.legend(fontsize=6.2, frameon=False, loc="center right")
    ax1.set_ylim(-0.03, 1.05)
    ax1.grid(alpha=0.25, lw=0.4)

    xs2 = [i * 0.5 for i in range(0, 241)]         # 0..120 min
    for fn, lab, c in ((fuzzy.o_tiny, "tiny", "#c00000"),
                       (fuzzy.o_small, "small", "#e08214"),
                       (fuzzy.o_medium, "medium", "#1f4e79"),
                       (fuzzy.o_large, "large", "#4f7a3a"),
                       (fuzzy.o_huge, "huge", "#7030a0")):
        ax2.plot(xs2, [fn(x) for x in xs2], lw=1.1, color=c, label=lab)
    ax2.set_xlabel("predicted wait (clinic min)", fontsize=7)
    ax2.set_title("(b) output: wait-time sets", fontsize=7.4)
    ax2.tick_params(labelsize=6.4, length=2)
    ax2.legend(fontsize=6.2, frameon=False, ncol=2)
    ax2.set_ylim(-0.03, 1.05)
    ax2.grid(alpha=0.25, lw=0.4)

    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "fig_fuzzy_sets.pdf")
    plt.close(fig)


def fuzzy_curve():
    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    aheads = list(range(0, 9))
    for avg, lab, c in ((3.0, "short (3 min)", "#c00000"),
                        (8.0, "medium (8 min)", "#1f4e79"),
                        (18.0, "long (18 min)", "#4f7a3a")):
        mids = [fuzzy.estimate(avg_min=avg, samples=8, cv=0.2, ahead=a,
                               progress=0.3).mid_min for a in aheads]
        ax.plot(aheads, mids, marker="o", ms=2.6, lw=1.1, color=c, label=lab, clip_on=False)
    ax.set_xlabel("patients ahead in the queue", fontsize=7)
    ax.set_ylabel("defuzzified wait (clinic min)", fontsize=7)
    ax.tick_params(labelsize=6.4, length=2)
    ax.legend(fontsize=6.2, frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_fuzzy_curve.pdf")
    plt.close(fig)


if __name__ == "__main__":
    architecture()
    membership_sets()
    fuzzy_curve()
    print("figures written to", OUT)
