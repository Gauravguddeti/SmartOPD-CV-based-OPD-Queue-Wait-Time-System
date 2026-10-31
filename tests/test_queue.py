"""Unit tests for the SmartOPD queue engine (occupant-based room model).

Room rules under test:
  * room.occupants - many people can be inside; ghosts carry no token
  * entries admit: called token first, else next waiting (auto-advance),
    else unlinked
  * exits complete ONLY the leaver's token (swap-safe)
  * tokens never move backwards (waiting/called -> in -> done only)
  * TIME_SCALE multiplies measured durations before they train the model
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config  # noqa: E402
from queue_manager import QueueEngine, QueueError  # noqa: E402


class FakeClock:
    def __init__(self, start=1_000_000.0):
        self.t = start

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += sec


def make_engine(scale: float = 1.0):
    clk = FakeClock()
    eng = QueueEngine(now_fn=clk)
    eng.time_scale = scale
    return eng, clk


# ------------------------------------------------------------------ tokens
def test_register_and_order():
    eng, _ = make_engine()
    eng.register("Asha Patil")
    eng.register("Rohan Mehta", priority=True)
    order = eng.queue_order()
    assert [t.no for t in order] == [2, 1], "priority jumps the queue"


def test_call_next_and_lifecycle():
    eng, clk = make_engine()
    a = eng.register("A")
    eng.register("B")
    called = eng.call_next()
    assert called.no == a.no
    clk.advance(30)
    assert eng.begin_occupant("t1", clk(), ghost=False, src="camera") == a.no
    assert a.status == "in"
    clk.advance(300)
    assert eng.remove_occupant("t1", src="camera") is True
    assert a.status == "done"
    assert eng.durations == [300]          # scale=1: model == real
    assert eng.room["occupants"] == []


def test_call_next_respects_priority():
    eng, _ = make_engine()
    eng.register("A")
    b = eng.register("B", priority=True)
    assert eng.call_next().no == b.no


def test_empty_queue_call_next_raises():
    eng, _ = make_engine()
    try:
        eng.call_next()
        assert False, "should raise"
    except QueueError:
        pass


def test_skip_noshow_requeue():
    eng, _ = make_engine()
    t = eng.register("A")
    eng.set_status(t.no, "skip")
    assert t.status == "skipped"
    eng.set_status(t.no, "requeue")
    assert t.status == "waiting"
    eng.set_status(t.no, "noshow")
    eng.set_status(t.no, "requeue")
    assert t.status == "waiting"


def test_invalid_token_raises():
    eng, _ = make_engine()
    try:
        eng._find(999)
        assert False
    except QueueError:
        pass


# ------------------------------------------------------- camera admissions
def test_entry_auto_advances_queue():
    eng, clk = make_engine()
    a = eng.register("A")
    b = eng.register("B")
    assert eng.begin_occupant("track-1", clk(), ghost=False) == a.no
    assert a.status == "in"
    clk.advance(60)
    assert eng.remove_occupant("track-1") is True
    assert a.status == "done"
    # next entry admits the next waiting token
    assert eng.begin_occupant("track-2", clk(), ghost=False) == b.no
    assert b.status == "in"


def test_entry_prefers_nurse_called_token():
    eng, clk = make_engine()
    eng.register("A")
    b = eng.register("B", priority=True)
    eng.call_token(b.no)                     # nurse called B explicitly
    assert eng.begin_occupant("track-1", clk(), ghost=False) == b.no
    assert b.status == "in"


def test_entry_with_empty_queue_is_unlinked():
    eng, clk = make_engine()
    tok = eng.begin_occupant("track-1", clk(), ghost=False)
    assert tok is None                       # nobody to admit
    assert len(eng.room["occupants"]) == 1   # but the walk-in is recorded
    assert eng.room["occupants"][0]["token_id"] is None


def test_ghost_never_takes_a_token():
    eng, clk = make_engine()
    a = eng.register("A")
    tok = eng.begin_occupant("ghost-1", clk(), ghost=True)
    assert tok is None
    assert a.status == "waiting"             # untouched
    # a real walk-in still admits normally beside the ghost
    assert eng.begin_occupant("track-1", clk(), ghost=False) == a.no
    assert len(eng.room["occupants"]) == 2


# ------------------------------------------------------------ exits / swap
def test_swap_exit_is_per_occupant():
    """Woman (ghost) exits while the man (linked) stays inside."""
    eng, clk = make_engine()
    eng.register("A")
    eng.begin_occupant("man", clk(), ghost=False)       # A in
    eng.begin_occupant("woman", clk(), ghost=True)      # ghost inside too
    assert len(eng.room["occupants"]) == 2
    assert eng.remove_occupant("woman") is True         # ghost leaves
    assert eng.tokens[0].status == "in"                 # man unaffected
    assert eng.room["occupants"][0]["track_id"] == "man"


def test_ghost_exit_does_not_touch_tokens():
    eng, clk = make_engine()
    eng.begin_occupant("ghost-1", clk(), ghost=True)
    assert eng.remove_occupant("ghost-1") is True
    assert eng.durations == []                          # nothing learned


def test_vanished_track_exits_its_occupant():
    eng, clk = make_engine()
    eng.register("A")
    eng.begin_occupant("track-9", clk(), ghost=False)
    assert eng.remove_occupant("track-9") is True       # 'vanished at the door'
    assert eng.tokens[0].status == "done"


# ------------------------------------------------- never move backwards
def test_no_backwards_after_admission():
    """The old 'exited within 15 s -> revert' rule is deleted: a camera exit
    always completes the token, whatever the duration."""
    eng, clk = make_engine()
    a = eng.register("A")
    eng.begin_occupant("track-1", clk(), ghost=False)
    assert a.status == "in"
    clk.advance(8)                            # far below the old 15 s rule
    eng.remove_occupant("track-1")
    assert a.status == "done"
    assert "called" not in a.history[a.history.index("in") + 1:]
    assert "waiting" not in a.history[a.history.index("in") + 1:]


def test_status_history_journal():
    eng, clk = make_engine()
    a = eng.register("A")
    eng.call_next()
    eng.begin_occupant("t", clk(), ghost=False)
    eng.remove_occupant("t")
    assert a.history == ["waiting", "called", "in", "done"]


# --------------------------------------------------------- learning / scale
def test_duration_learned_above_min_only():
    eng, clk = make_engine()
    eng.register("A")
    eng.register("B")
    eng.begin_occupant("t1", clk(), ghost=False)      # admits A
    clk.advance(2)                            # < MIN_LEARN_SECONDS (4 s)
    eng.remove_occupant("t1")
    assert eng.durations == []
    eng.begin_occupant("t2", clk(), ghost=False)      # admits B
    clk.advance(10)
    eng.remove_occupant("t2")
    assert eng.durations == [10]


def test_time_scale_inflates_model_durations():
    eng, clk = make_engine(scale=config.TIME_SCALE)
    eng.register("A")
    eng.begin_occupant("t1", clk(), ghost=False)
    clk.advance(10)                           # 10 real seconds
    eng.remove_occupant("t1")
    assert eng.durations == [600]             # 10 min of model time
    # avg = 0.75*prior(8 min) + 0.25*measured(10 min) = 8.5 model-min
    assert abs(eng.avg_seconds() - 510.0) < 0.1


def test_time_scale_shortens_real_learn_gate():
    """MIN_LEARN is checked in REAL seconds even though durations are scaled."""
    eng, clk = make_engine(scale=config.TIME_SCALE)
    eng.register("A")
    eng.register("B")
    eng.begin_occupant("t1", clk(), ghost=False)
    clk.advance(3)                            # 3 real s < MIN_LEARN(4 s)
    eng.remove_occupant("t1")
    assert eng.durations == []
    eng.begin_occupant("t2", clk(), ghost=False)
    clk.advance(5)                            # 5 real s = 5 model minutes
    eng.remove_occupant("t2")
    assert eng.durations == [300]


def test_eta_uses_scaled_elapsed_and_wallclock():
    eng, clk = make_engine(scale=config.TIME_SCALE)
    a = eng.register("A")
    b = eng.register("B")
    eng.begin_occupant("t1", clk(), ghost=False)     # A admitted at t
    clk.advance(10)                                  # 10 real s = 10 model min
    e = eng.eta(b.no)
    assert e is not None and e["ahead"] == 0
    # remaining = avg(8 min) - elapsed(10 min) floored at 1 min
    assert e["mid_s"] == 60
    # wall-clock expectation stays in real time (10 model min -> 10 real s)
    assert abs(e["expected_at"] - (clk() + 1.0)) < 1.0


# ------------------------------------------------------- manual overrides
def test_manual_enter_and_free():
    eng, clk = make_engine()
    eng.register("A")
    eng.register("B")
    assert eng.manual_enter("nurse") is not None     # admits A
    assert eng.tokens[0].status == "in"
    clk.advance(60)
    assert eng.manual_free("nurse") is True
    assert eng.tokens[0].status == "done"
    assert eng.manual_free("nurse") is False         # room already empty


def test_manual_free_clears_ghosts():
    eng, clk = make_engine()
    eng.begin_occupant("ghost-1", clk(), ghost=True)
    assert eng.manual_free() is True
    assert eng.room["occupants"] == []


def test_undo_last():
    eng, clk = make_engine()
    a = eng.register("A")
    eng.call_next()
    eng.begin_occupant("t", clk(), ghost=False)
    clk.advance(120)
    eng.remove_occupant("t")
    assert eng.stats()["served"] == 1 and eng.durations == [120]
    t = eng.undo_last()
    assert t.no == a.no and t.status == "waiting"
    assert eng.durations == []
    assert eng.stats()["served"] == 0


def test_undo_without_history_raises():
    eng, _ = make_engine()
    try:
        eng.undo_last()
        assert False
    except QueueError:
        pass


def test_simulate_completes_consultation():
    eng, _ = make_engine()
    eng.register("A")
    res = eng.simulate()
    assert res["simulated"] is True
    assert eng.tokens[0].status == "done"
    assert len(eng.durations) == 1


def test_simulate_requires_patients():
    eng, _ = make_engine()
    try:
        eng.simulate()
        assert False
    except QueueError:
        pass


# ------------------------------------------------------------- persistence
def test_roundtrip_serialization_with_occupants():
    eng, clk = make_engine()
    eng.register("Asha Patil", "New visit", priority=True)
    eng.begin_occupant("track-3", clk(), ghost=False)
    eng.begin_occupant("ghost-1", clk(), ghost=True)
    data = eng.to_dict()
    eng2 = QueueEngine.from_dict(data, now_fn=clk)
    assert eng2.tokens[0].status == "in"
    assert eng2.tokens[0].priority is True
    assert len(eng2.room["occupants"]) == 2
    assert eng2.occupant_tag("track-3") == "token #1"
    assert eng2.occupant_tag("ghost-1") == "ghost"


def test_old_state_format_migrates():
    """An older-format state (single-dict room) must not crash the loader."""
    eng = QueueEngine.from_dict({
        "tokens": [{"no": 1, "name": "A", "kind": "New visit",
                    "priority": False, "status": "done", "t_reg": 1.0}],
        "room": {"no": 1, "t_in": 5.0, "src": "camera"},
    }, now_fn=time.time)
    assert eng.tokens[0].status == "done"
    assert eng.room["occupants"] == []


def test_corrupt_state_recovers():
    eng = QueueEngine.from_dict({"tokens": "garbage"}, now_fn=time.time)
    assert eng.tokens == []


def test_stats():
    eng, clk = make_engine()
    eng.register("A")
    eng.register("B")
    eng.begin_occupant("t", clk(), ghost=False)
    s = eng.stats()
    assert s["waiting"] == 1 and s["called"] == 0 and s["served"] == 0
    assert s["occupants"] == 1
    eng.update_crowd(7, 5, 1)
    assert eng.stats()["peak_crowd"] == 7
    assert eng.stats()["crowd"]["room"] == 1


def test_csv_export():
    eng, clk = make_engine()
    eng.register("Asha Patil")
    eng.call_next()
    eng.begin_occupant("t", clk(), ghost=False)
    clk.advance(300)
    eng.remove_occupant("t")
    csv = eng.csv_text()
    assert "Asha Patil" in csv and "done" in csv and "camera" in csv
