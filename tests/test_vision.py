"""Tests for the vision logic: tracker + occupant-based zone analyzer.

The analyzer follows every person inside the doctor zone as an occupant
(with ghost occupants carrying no token).  These tests drive an injectable
clock and build multi-person frames.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))

import config  # noqa: E402
from vision import CentroidTracker, Track, ZoneAnalyzer, zone_contains  # noqa: E402


# ------------------------------------------------------------------ tracker
def test_tracker_assigns_and_expires():
    tr = CentroidTracker(max_miss=2, match_dist=0.2)
    vis = tr.update([{"cx": 0.5, "cy": 0.5, "bbox": (0.4, 0.4, 0.6, 0.6)}])
    assert len(vis) == 1 and vis[0].tid == 1
    vis = tr.update([{"cx": 0.52, "cy": 0.5, "bbox": (0.42, 0.4, 0.62, 0.6)}])
    assert len(vis) == 1 and vis[0].tid == 1 and vis[0].hits == 2
    tr.update([])
    tr.update([])
    assert tr.tracks != []          # still within the miss budget
    tr.update([])
    assert tr.tracks == [] and len(tr.expired) == 1


def test_tracker_two_people_no_swap():
    tr = CentroidTracker()
    tr.update([{"cx": 0.2, "cy": 0.5, "bbox": (0.1, 0.4, 0.3, 0.6)},
               {"cx": 0.8, "cy": 0.5, "bbox": (0.7, 0.4, 0.9, 0.6)}])
    ids = sorted(t.tid for t in tr.tracks)
    tr.update([{"cx": 0.25, "cy": 0.5, "bbox": (0.15, 0.4, 0.35, 0.6)},
               {"cx": 0.75, "cy": 0.5, "bbox": (0.65, 0.4, 0.85, 0.6)}])
    assert sorted(t.tid for t in tr.tracks) == ids


# ------------------------------------------------------------------- zones
def test_zone_contains():
    z = [0.1, 0.1, 0.5, 0.5]
    assert zone_contains(z, 0.3, 0.3)
    assert not zone_contains(z, 0.6, 0.3)
    assert not zone_contains(None, 0.3, 0.3)
    assert not zone_contains([0.1, 0.1], 0.3, 0.3)


class FakeClock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, sec):
        self.t += sec


class Recorder:
    def __init__(self):
        self.events = []
        self.crowd = []

    def entry(self, occ): self.events.append(("entry", occ["key"], occ["ghost"]))
    def exit(self, key, occ): self.events.append(("exit", key, occ["ghost"]))
    def passby(self, dwell): self.events.append(("passby", round(dwell, 1)))
    def note(self, msg): self.events.append(("note", msg))
    def crowd_cb(self, f, w, r): self.crowd.append((f, w, r))


def make_analyzer():
    rec = Recorder()
    clk = FakeClock()
    az = ZoneAnalyzer(on_entry=rec.entry, on_exit=rec.exit,
                      on_passby=rec.passby, on_note=rec.note,
                      on_crowd=rec.crowd_cb, now_fn=clk)
    az.set_zones([0.0, 0.0, 0.6, 1.0],    # waiting = left 60 %
                 [0.7, 0.0, 1.0, 1.0])    # doctor  = right 30 %
    return az, clk, rec


def tk(tid, cx, feet_y=0.8, hits=5):
    t = Track(tid=tid, cx=cx, cy=feet_y - 0.1,
              bbox=(cx - 0.05, feet_y - 0.3, cx + 0.05, feet_y))
    t.hits = hits
    return t


def walk_in(az, clk, tid):
    """Seen outside, then inside with enough dwell to confirm an entry."""
    az.process([tk(tid, 0.2)])            # spotted in the waiting area
    clk.advance(1.0)
    az.process([tk(tid, 0.85)])           # crosses into the doctor zone
    clk.advance(config.ENTRY_DWELL + 0.2)
    az.process([tk(tid, 0.85)])           # dwell reached -> occupant


def walk_out(az, clk, tid, frames=None):
    frames = frames or config.ROOM_LATCH_FRAMES
    for _ in range(frames):
        az.process([tk(tid, 0.2)])
        clk.advance(0.2)


# -------------------------------------------------------------- entries
def test_entry_after_dwell_then_exit():
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=7)
    assert rec.events == [("entry", 7, False)]
    assert 7 in az.occupants
    walk_out(az, clk, tid=7)
    assert rec.events[-1] == ("exit", 7, False)


def test_pass_by_never_creates_occupant():
    az, clk, rec = make_analyzer()
    az.process([tk(3, 0.2)])              # seen outside
    clk.advance(0.4)
    for _ in range(3):                    # ~1.2 s inside: below ENTRY_DWELL
        az.process([tk(3, 0.85)])
        clk.advance(0.4)
    walk_out(az, clk, tid=3)
    assert [e[0] for e in rec.events] == ["passby"]
    assert az.occupants == {}


def test_ghost_when_inside_at_tracking_start():
    az, clk, rec = make_analyzer()
    az.process([tk(9, 0.85)])             # never seen outside
    assert rec.events == [("entry", 9, True)]
    assert az.occupants[9]["ghost"] is True
    walk_out(az, clk, tid=9)
    assert rec.events[-1] == ("exit", 9, True)


def test_flicker_does_not_end_occupancy():
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=1)
    for _ in range(config.ROOM_LATCH_FRAMES - 1):
        az.process([tk(1, 0.2)])          # dropped detections
        clk.advance(0.2)
    assert not any(e[0] == "exit" for e in rec.events)
    az.process([tk(1, 0.85)])             # found again
    walk_out(az, clk, tid=1)              # now a real exit
    assert ("exit", 1, False) in rec.events


def test_vanish_at_the_door_is_an_exit():
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=4)
    az.track_vanished(4)                  # track disappears at the door
    assert ("exit", 4, False) in rec.events
    assert az.occupants == {}


def test_pending_vanish_is_a_passby():
    az, clk, rec = make_analyzer()
    az.process([tk(5, 0.2)])
    clk.advance(0.5)
    az.process([tk(5, 0.85)])             # just stepped in
    az.track_vanished(5)                  # gone before the dwell
    assert [e[0] for e in rec.events] == ["passby"]


# ------------------------------------------------------------- swap case
def test_swap_ghost_leaves_linked_stays():
    """Woman (ghost, inside since start) exits while the man stays inside."""
    az, clk, rec = make_analyzer()
    az.process([tk(2, 0.85)])             # woman already inside -> ghost
    walk_in(az, clk, tid=3)               # man walks in -> linked occupant
    walk_out(az, clk, tid=2)              # woman leaves
    assert ("exit", 2, True) in rec.events
    assert 3 in az.occupants, "the man must still be inside"
    assert ("exit", 3, False) not in rec.events


def test_two_walk_ins_are_two_occupants():
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=1)
    walk_in(az, clk, tid=2)
    assert set(az.occupants) == {1, 2}
    assert [e for e in rec.events if e[0] == "entry"] == [
        ("entry", 1, False), ("entry", 2, False)]


# --------------------------------------------------------------- seams
def test_jumpcut_seam_rebinds_occupant():
    """Jump-cut loop: hold events, reset tracker, re-bind the same patient."""
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=7)
    az.begin_seam()                        # jump-cut loop point
    assert any(e[0] == "note" and "SEAM_HOLD" in e[1] for e in rec.events)
    clk.advance(config.SEAM_HOLD + 0.1)
    az.process([tk(99, 0.85)])             # tracker reset: new id, same person
    assert any(e[0] == "note" and "REBOUND" in e[1] for e in rec.events)
    assert list(az.occupants) == [99], "occupant re-bound to the new track"
    assert not any(e[0] in ("entry", "exit") for e in rec.events[1:]), \
        "seam must not emit entry/exit events"
    walk_out(az, clk, tid=99)
    assert rec.events[-1] == ("exit", 99, False)


def test_seam_keeps_occupant_when_nobody_returns():
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=7)
    az.begin_seam()
    clk.advance(config.SEAM_HOLD + 0.1)
    az.process([])                         # nobody visible inside
    assert any("kept without a track" in e[1] for e in rec.events if e[0] == "note")
    assert len(az.occupants) == 1          # kept, waiting for a real exit


def test_seam_extras_become_ghosts():
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=7)
    az.begin_seam()
    clk.advance(config.SEAM_HOLD + 0.1)
    az.process([tk(51, 0.85), tk(52, 0.85)])   # two people inside after seam
    assert len(az.occupants) == 2
    assert az.occupants[51]["ghost"] is False  # linked occupant re-bound
    assert az.occupants[52]["ghost"] is True   # extra person is a ghost


def test_events_frozen_during_seam_hold():
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=7)
    az.begin_seam()
    clk.advance(0.3)                       # inside the hold window
    for _ in range(config.ROOM_LATCH_FRAMES + 2):
        az.process([])                     # everything vanished (tracker reset)
        clk.advance(0.1)
    assert not any(e[0] == "exit" for e in rec.events), \
        "no exits may fire while a seam is held"


def test_continuous_loop_keeps_identity():
    """Ping-pong: the wrap is frame-continuous - no seam, no re-bind."""
    az, clk, rec = make_analyzer()
    walk_in(az, clk, tid=7)
    for _ in range(10):                    # frames keep coming across the wrap
        az.process([tk(7, 0.85)])
        clk.advance(0.1)
    assert list(az.occupants) == [7]
    assert not any(e[0] == "note" for e in rec.events)


# --------------------------------------------------------------- crowd
def test_waiting_count_excludes_room():
    az, clk, rec = make_analyzer()
    seated = tk(1, 0.2); seated.hits = 5
    fresh = tk(2, 0.3); fresh.hits = 1
    inside = tk(3, 0.85); inside.hits = 5
    az.process([seated, fresh, inside])
    frame_n, waiting, room = rec.crowd[-1]
    assert frame_n == 3
    assert waiting == 1, "fresh track not counted; person in the room not waiting"
    assert room == 1
