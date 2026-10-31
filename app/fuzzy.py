"""Fuzzy wait-time estimator — a small Mamdani inference system.

Why fuzzy?
----------
The crisp formula `wait = remainder + people_ahead x average` is brittle:
a single noisy measurement swings the number, and it cannot express
"we are not sure yet".  A fuzzy layer reasons with *linguistic* terms and
returns a range plus a confidence tag, which is both easier to explain to
a nurse and safer for patients.

Inputs (all crisp values, fuzzified with triangular / trapezoidal sets)
----------------------------------------------------------------------
avg_min     learned consultation length in CLINIC minutes
samples     how many consultations the average is based on
cv          coefficient of variation of those measurements (spread)
ahead       people ahead of this token in the queue
progress    elapsed / average of the consultation currently in the room
            (None when the room is empty)

Output
------
mid / lo / hi in clinic minutes + a linguistic label + confidence tag.

Everything is pure Python (no numpy, no scipy) and deterministic, so it is
straightforward to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass

import config

# ---------------------------------------------------------------- membership
def _tri(x: float, a: float, b: float, c: float) -> float:
    """Triangular membership: 0 at a, 1 at b, 0 at c."""
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    return (x - a) / (b - a) if x < b else (c - x) / (c - b)


def _trap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Trapezoidal membership: ramps a->b, holds b..c, falls c->d."""
    if x <= a or x >= d:
        return 0.0
    if b <= x <= c:
        return 1.0
    return (x - a) / (b - a) if x < b else (d - x) / (d - c)


# ------------------------------------------------------------- input sets
def m_short(minutes: float) -> float:
    return _tri(minutes, 0.0, 2.5, 6.0)


def m_medium(minutes: float) -> float:
    return _tri(minutes, 4.0, 8.0, 14.0)


def m_long(minutes: float) -> float:
    return _trap(minutes, 11.0, 16.0, 45.0, 45.0)


def m_none_ahead(x: float) -> float:
    return _tri(x, -1.0, 0.0, 2.0)


def m_few_ahead(x: float) -> float:
    return _tri(x, 1.0, 3.0, 6.0)


def m_many_ahead(x: float) -> float:
    return _trap(x, 4.0, 6.0, 99.0, 99.0)


def m_just_started(p: float) -> float:
    return _tri(p, 0.0, 0.15, 0.6)


def m_mid_progress(p: float) -> float:
    return _tri(p, 0.35, 0.8, 1.3)


def m_ending(p: float) -> float:
    return _trap(p, 1.0, 1.4, 5.0, 5.0)


def m_tight(cv: float) -> float:
    return _tri(cv, 0.0, 0.05, 0.2)


def m_moderate(cv: float) -> float:
    return _tri(cv, 0.1, 0.3, 0.55)


def m_wide(cv: float) -> float:
    return _trap(cv, 0.4, 0.6, 5.0, 5.0)


def m_low_conf(samples: int) -> float:
    return _trap(float(samples), -1.0, 0.0, 2.0, 3.5)


def m_high_conf(samples: int) -> float:
    return _trap(float(samples), 2.5, 4.0, 99.0, 99.0)


# ------------------------------------------------------------- output sets
def o_tiny(x: float) -> float:
    return _tri(x, 0.0, 0.8, 2.5)


def o_small(x: float) -> float:
    return _tri(x, 1.0, 4.0, 9.0)


def o_medium(x: float) -> float:
    return _tri(x, 6.0, 13.0, 24.0)


def o_large(x: float) -> float:
    return _tri(x, 18.0, 32.0, 55.0)


def o_huge(x: float) -> float:
    return _trap(x, 45.0, 70.0, 120.0, 120.0)


UNIVERSE = [i * 0.5 for i in range(241)]        # 0 .. 120 min, step 0.5


@dataclass
class FuzzyResult:
    mid_min: float
    lo_min: float
    hi_min: float
    label: str
    confidence: str
    fired: list            # (rule, strength) for transparency / debugging


def _label(mid: float) -> str:
    if mid < 1.5:
        return "almost your turn"
    if mid < 6:
        return "a few minutes"
    if mid < 18:
        return "about 5–18 min"
    if mid < 35:
        return "about 15–35 min"
    return "over 35 min"


def estimate(avg_min: float, samples: int, cv: float, ahead: int,
             progress: float | None) -> FuzzyResult:
    """Run the Mamdani inference and return the defuzzified wait.

    avg_min  : learned average in clinic minutes (PRIOR before learning)
    samples  : number of measured consultations behind that average
    cv       : coefficient of variation of those measurements
    ahead    : people ahead of this token
    progress : elapsed/avg of the current consultation, or None
    """
    avg_min = max(0.5, avg_min)
    cv = max(0.0, min(cv, 3.0))
    p = progress if progress is not None else 0.0

    short, medium, long_ = m_short(avg_min), m_medium(avg_min), m_long(avg_min)
    none_a, few_a, many_a = (m_none_ahead(ahead), m_few_ahead(ahead),
                             m_many_ahead(ahead))
    ending = m_ending(p) if progress is not None else 0.0
    low_conf = m_low_conf(samples)
    wide = m_wide(cv)

    # rule: (strength, output set)
    rules = [
        ("short+none", min(short, none_a), o_tiny),
        ("short+few", min(short, few_a), o_small),
        ("short+many", min(short, many_a), o_medium),
        ("medium+none", min(medium, none_a), o_small),
        ("medium+few", min(medium, few_a), o_medium),
        ("medium+many", min(medium, many_a), o_large),
        ("long+none", min(long_, none_a), o_medium),
        ("long+few", min(long_, few_a), o_large),
        ("long+many", min(long_, many_a), o_huge),
        # a consultation that has already overrun its average is likely to
        # finish soon -> the first waiting patient is nearly up
        ("ending+none", min(ending, none_a), o_tiny),
        # little measured data yet -> stay close to the prior, not optimistic
        ("lowconf", min(low_conf, 1.0) * 0.6, o_medium),
    ]

    fired = [(name, round(s, 3)) for name, s, _ in rules if s > 0.0]
    num = den = 0.0
    for _name, strength, fn in rules:
        if strength <= 0.0:
            continue
        for x in UNIVERSE:
            mu = min(strength, fn(x))
            if mu > 0.0:
                num += x * mu
                den += mu

    if den > 0.0:
        mid = num / den
    else:
        # nothing fired (should not happen) -> crisp fallback
        mid = avg_min * (ahead + 1)

    mid = max(0.75, mid)                      # never promise less than 45 s

    # ---- bounds: uncertainty from spread + sample count -------------------
    spread = 0.18 + 0.32 * wide + 0.22 * (1.0 - m_high_conf(samples))
    if progress is not None and ahead == 0:
        spread *= 0.75                        # current consult gives us detail
    lo = max(0.25, mid * (1.0 - spread))
    hi = mid * (1.0 + spread * 1.15)

    confidence = ("high" if m_high_conf(samples) > 0.5
                  else "medium" if samples >= 2 else "low")
    if cv > 0.6 and confidence == "high":
        confidence = "medium"                 # high spread undermines trust

    return FuzzyResult(mid_min=round(mid, 2), lo_min=round(lo, 2),
                       hi_min=round(hi, 2), label=_label(mid),
                       confidence=confidence, fired=fired)


def prior_only() -> FuzzyResult:
    """Estimate before any consultation has been measured."""
    return estimate(avg_min=config.PRIOR_SECONDS / 60.0, samples=0, cv=1.0,
                    ahead=1, progress=None)
