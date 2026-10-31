"""Tests for the fuzzy wait-time estimator (app/fuzzy.py).

The fuzzy system must be sensible in all the directions a nurse would
expect: more people ahead -> longer wait, longer consultations -> longer
wait, a consultation that has overrun its average -> the next patient is
nearly up, and little data -> wider bounds (never false precision).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import fuzzy  # noqa: E402


def est(**kw):
    defaults = dict(avg_min=8.0, samples=8, cv=0.2, ahead=0, progress=None)
    defaults.update(kw)
    return fuzzy.estimate(**defaults)


def test_more_people_ahead_means_longer_wait():
    a0 = est(ahead=0)
    a3 = est(ahead=3)
    a6 = est(ahead=6)
    assert a0.mid_min < a3.mid_min < a6.mid_min


def test_longer_consultations_mean_longer_wait():
    short = est(avg_min=3.0, ahead=2)
    long_ = est(avg_min=20.0, ahead=2)
    assert short.mid_min < long_.mid_min


def test_progress_ending_shrinks_first_patient_estimate():
    fresh = est(ahead=0, progress=0.1)      # consultation just started
    ending = est(ahead=0, progress=1.6)     # overrun its average
    assert ending.mid_min < fresh.mid_min


def test_low_confidence_widens_bounds():
    sure = est(samples=10, cv=0.1)
    unsure = est(samples=0, cv=1.0)
    sure_width = sure.hi_min - sure.lo_min
    unsure_width = unsure.hi_min - unsure.lo_min
    assert unsure_width > sure_width
    assert unsure.confidence == "low"
    assert sure.confidence == "high"


def test_high_spread_downgrades_confidence():
    steady = est(samples=10, cv=0.1)
    noisy = est(samples=10, cv=0.9)
    assert steady.confidence == "high"
    assert noisy.confidence in ("medium", "low")
    assert (noisy.hi_min - noisy.lo_min) > (steady.hi_min - steady.lo_min)


def test_prior_when_no_measurements():
    res = fuzzy.prior_only()
    assert 5.0 <= res.mid_min <= 20.0       # around the 8-minute prior
    assert res.confidence == "low"


def test_bounds_are_ordered_and_positive():
    for ahead in (0, 1, 4, 10):
        res = est(ahead=ahead)
        assert 0 < res.lo_min <= res.mid_min <= res.hi_min
        assert res.mid_min >= 0.75          # never promise less than ~45 s


def test_labels_are_monotonic():
    small = est(ahead=0, progress=1.8, avg_min=2.0, samples=10)
    big = est(ahead=12, avg_min=25.0, samples=10)
    assert small.label in ("almost your turn", "a few minutes")
    assert big.label in ("about 15–35 min", "over 35 min")


def test_rules_fire_for_transparency():
    res = est(ahead=2, avg_min=9.0)
    names = [n for n, _ in res.fired]
    assert "medium+few" in names


def test_progress_none_is_safe():
    res = est(progress=None)
    assert res.mid_min > 0
