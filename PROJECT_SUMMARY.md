# SmartOPD — Project Summary
### CV-based OPD Queue & Wait-Time Prediction System
*Status: working prototype, running and tested (37 unit tests green) · Oct 2026*

---

## 1. The problem

In a small hospital outpatient department (OPD):

- Patients have no idea how long their wait will be, so they crowd the desk to ask.
- The reception/nurse manually announces tokens and keeps paper records.
- Nobody measures real consultation times, so any "estimated wait" is a guess.

## 2. What SmartOPD does

A single camera watches the OPD waiting area and the doctor's door. A YOLO deep-learning
model detects and tracks every person; a rule engine converts that motion into
**semantic events** ("a patient walked into the doctor's room", "the patient walked out").
Those events drive a **live digital queue**: tokens are admitted automatically as patients
go in, completed when they come out, and an **adaptive wait-time model** — trained on the
real consultation lengths the camera measured — predicts each waiting patient's ETA.

Three browser interfaces share one server, so the system runs as three screens:

| # | Interface | URL | Used by |
|---|-----------|-----|---------|
| ① | **Monitoring Console** | `http://127.0.0.1:8000/` | operator/IT — camera view with tracking overlay, zone drawing, live crowd, event log |
| ② | **Nurse Desk** | `http://127.0.0.1:8000/nurse` | nurse — register patients, call/announce, skip & no-show, undo, CSV export |
| ③ | **Patient Display** | `http://127.0.0.1:8000/patient` | waiting-area TV — IN CONSULTATION, NEXT UP / PLEASE PROCEED, waiting count, per-token ETA lookup |

All processing and storage happen **on the local machine only** — no cloud, no frames
ever leave the building (privacy by design).

## 3. Evolution from the prototype to the current system

| | v1 prototype | current |
|---|---|---|
| Detection | COCO-SSD loaded from a CDN **in the browser** → broke offline ("Detector failed to load") | **YOLO11n runs on the server**, weights stored in the project folder, fully offline after setup |
| Tracking | naive nearest-match in JS, flicker caused false door events | centroid tracker + **direction-aware, time-confirmed** zone state machine |
| Queue link | camera events never touched the queue (unlinked) | **camera drives the queue** (auto-advance) with pass-by protection and nurse override |
| State | browser `localStorage` (one screen only) | server-side JSON, shared by all three screens, atomic writes, crash-safe |
| Wait-time model | 8-min prior + last-10 median (could be poisoned by 0-min readings) | same idea, formalised + **outlier-guarded** (sub-4 s readings never train it) + unit tests |
| Ops | none | one-click `run.bat`, local `.venv`, locked dependencies, 47 tests, API docs at `/api/docs` |

## 4. Architecture

```
                ┌──────────────────────── server (app/) ────────────────────────┐
 camera file ──▶│ VisionWorker (background thread, ~10–16 FPS)                     │
 or webcam     │   cv2 capture → YOLO11n (person class) → CentroidTracker         │
               │        │                                  │                      │
               │        │ annotated JPEG (boxes, ids,     │ ZoneAnalyzer          │
               │        │ zones, CCTV overlay)            │ direction-aware       │
               │        ▼                                  │ room events           │
               │  /api/snapshot.jpg                        ▼                      │
               │                                     QueueEngine                 │
               │                              tokens · room · ETA model          │
               └───────┬──────────────────────────────┬──────────┬──────────────┘
                       │ poll /api/state (1 s)        │ enter /  │ settings
                       │                              │ free     │
        ┌──────────────┴──────────┐   ┌───────────────┴──┐   ┌───┴────────────┐
        │ ① Monitoring Console    │   │ ② Nurse Desk     │   │ ③ Patient      │
        │ zones · crowd · events  │   │ tokens · calls   │   │ Display (TV)   │
        └─────────────────────────┘   └──────────────────┘   └────────────────┘
```

**Tech stack:** Python 3.11 · FastAPI + Uvicorn · Ultralytics YOLO11n (weights `yolo11n.pt`,
6 MB, CPU or GPU) · OpenCV · vanilla HTML/CSS/JS front-end (no build step) · pytest.

**Folder layout (everything inside `D:\clg stuff\CV`, nothing installed globally):**

```
run.bat                  one-click launcher (creates .venv, starts server, opens browser)
requirements.txt / requirements.lock.txt
app/    config.py  server.py  queue_manager.py  vision.py  worker.py  store.py
        pages/ (monitor.html · nurse.html · patient.html)   static/ (css/js)
models/ yolo11n.pt              samples/ demo clips       data/ state.json + uploads
tests/  test_queue.py (16) · test_vision.py (21)      scripts/ ensure_ready.py · smoke_test.py
```

## 5. The CV pipeline (how a video becomes events)

1. **Capture** — the worker thread reads the selected source (demo clip, uploaded clip,
   or webcam) at ~10–16 FPS on CPU. File sources **loop forever** to act like a live
   CCTV feed; at every loop the tracker is reset so the jump-cut cannot fake events.
2. **Detection** — YOLO11n, `person` class only, confidence ≥ 0.35, 640 px input.
3. **Tracking** — centroid tracker (greedy nearest-neighbour matching). Each person keeps
   an ID; a track may miss 15 frames (~1.5 s) before it expires, so brief occlusions
   don't spawn new identities.
4. **Occupant-based door logic (the heart of the system)** — every person inside the
   *doctor zone* becomes an **occupant** `{track_id, token_id, t_in}`:
   - **ENTRY** = a track *seen outside* walks in and stays ≥ **2 s** continuously, then the
     occupant takes the next token. Someone already inside when tracking starts becomes a
     **ghost occupant** (no token) — a patient can only be admitted by visibly walking in.
   - **EXIT** = that occupant leaves (4 consecutive outside frames) or its track vanishes at
     the door. Only *that* occupant's token completes, so in a swap (one person leaving while
     another is inside) the right token advances and the other person is untouched.
   - **Pass-by** = anyone who crosses the door area for less than 2 s never becomes an
     occupant — no token is touched.
   - **Never backwards**: tokens only move `waiting/called → in → done`; the camera cannot
     return a token to an earlier state.
   - **Seams** — the ping-pong demo clip loops frame-continuously (no tracker reset, no event
     hold). Ordinary jump-cut sources freeze events for 1.5 s and re-bind occupants to the
     people visible after the cut.
   - **Loop continuity**: if a consultation is in progress when the clip loops, it stays
     occupied and is **re-bound** to the same person's new track ID — short demo clips
     still produce realistic long consultations.
5. **Zones** — two rectangles drawn by the operator on the monitor screen (persisted
   **per clip**): *WAITING area* and *DOCTOR room/door*. An auto-detect helper samples the
   clip and proposes zones, but only ever suggests a door near the frame edge — it never
   invents a mid-room "door".
6. **Crowd analytics** — `people in frame`, `in waiting area` (excludes anyone in the
   doctor zone, so the number **drops** when a patient is admitted), `in doctor room`,
   and a session peak.
7. **Presentation** — frames are annotated server-side (track IDs, zone rectangles,
   `CAM-01` label, live timestamp, REC dot) and streamed as a snapshot feed, so every
   screen shows the same CCTV-style view.

## 6. The queue logic (token state machine)

```
                 nurse "Call"                  camera entry (2 s dwell, occupant confirmed)
   WAITING ─────────────────▶ CALLED ─────────▶ IN CONSULTATION ──────▶ DONE
      ▲   auto-advance: the camera admits        │   that occupant walks out      │
      │   the next token; ghost occupants        │   (or its track vanishes)      │
      │   never take a token                     │                                │
      └─── skip/no-show ──▶ requeue ◀────────────┴──── nurse "Undo last consultation"
   also: SKIPPED / NO-SHOW  ──re-queue──▶ WAITING
```

**Linking rules when the camera sees an entry (in priority order):**
1. If the nurse already **called** a token → that token is admitted (nurse's choice wins).
2. Else, if **auto-advance** is on (default) → the next waiting token (priority first,
   then token number) is called and admitted automatically, and the event log says so.
3. Else (auto-advance off) → the visit is recorded as *unlinked* and the nurse assigns it
   manually. This is the "nurse confirms every admission" mode.

**Safety rules that protect the queue from camera noise:**
- A camera entry consumes a token only after the 2 s dwell confirmation — crossing the door
  area never admits anybody.
- Exits are matched **per occupant**: in a swap scene the woman leaving completes only her
  token (or nothing, if she was a ghost) — the man's consultation is untouched.
- Tokens never move backwards: there is no "undo admission because it was short" rule; once a
  patient is in, only a real exit finishes them.
- Consultations shorter than **4 real seconds** still complete the token but do not train the
  ETA model (noise guard).
- Every event is human-readable in the log with a reason code and its source:
  `ENTRY_CONFIRMED`, `PASS_BY`, `EXIT_LINKED`, `EXIT_GHOST`, `SEAM_HOLD`, `REBOUND`
  (`camera`, `nurse`, `manual`, `simulation`).

**Nurse controls:** register patient (name, new/follow-up, priority ⚑) · call next ·
call a specific token · skip · no-show · re-queue · **Undo last consultation** ·
mode toggle (camera advances / nurse confirms) · voice announcement toggle ·
CSV export · reset.

**Patient experience:** the TV shows IN CONSULTATION and NEXT UP (turns into flashing
"PLEASE PROCEED" when called); a patient types their token number to see *N patients
ahead*, an ETA range, and the expected clock time. Voice announcements
("Token number 5, please proceed…") are spoken by the browser on the nurse desk and TV.

## 7. The adaptive wait-time model

- Starts from an **8-minute prior** (typical OPD consultation).
- As the camera measures real consultations, it blends toward the **rolling median of the
  last 12** measurements, reaching full trust after 4 samples.
  `estimate = prior·(1−w) + median·w`, with `w = min(1, n/4)`.
- **TIME_SCALE** multiplies measured durations before they enter the model (60 for video
  clips, 1 for a live webcam), so a 10-second demo consultation becomes a realistic
  10-minute one without changing the algorithm.
- ETA for a waiting token = **remaining time of the current consultation** (average minus
  elapsed, floored at 1 min) + **(people ahead) × average**, shown as a range
  −25 % / +30 % and converted to an expected clock time.
- The same model powers the queue table estimates on the nurse desk and the big numbers
  on the patient display — one source of truth on the server.

## 8. Engineering quality

- **47 unit tests** (pytest): token lifecycle, priority ordering, occupant admissions, swaps,
  ghost exits, never-backwards invariant, TIME_SCALE learning gates, undo, model
  convergence, serialisation/migration, CSV export; tracker identity/expiry; entry dwell,
  pass-by, flicker, vanish exits, seam hold/re-bind, continuous loop identity, crowd counts.
- **Replay proof**: `scripts/replay_check.py` runs the real pipeline headless over the
  ping-pong demo clip for 3 loops and asserts that six tokens complete in order, never move
  backwards, and all train the ETA model.
- **Crash safety**: state is persisted with atomic file writes every 2 s; a corrupt file
  self-heals to a fresh state instead of crashing.
- **Containment guarantee**: the venv, pip cache, YOLO weights, uploads and state all
  live inside the project folder — nothing is installed globally, no internet needed
  after first setup.
- **Privacy**: frames exist only in memory; only anonymous counters, tokens and durations
  are stored locally.

## 9. Honest limitations (worth discussing)

1. **Identity** — the camera sees anonymous people; the queue assumes the person who
   walks in is the next token. It cannot verify that the right patient went in
   (fixing this needs face/person re-identification — future scope).
2. **Single doctor room** — one room, one active consultation at a time.
3. **Visibility** — the patient must remain visible inside the doctor zone while
   consulting (glass door / open door cameras work best); if the room is fully out of
   view, the nurse uses the manual Enter/Free buttons.
4. **Demo clips** — real deployments use a continuous feed; for demos the clip is played
   ping-pong (frame-continuous loop, no tracker reset) and TIME_SCALE=60 converts the short
   real durations into realistic clinic minutes.
5. **CPU inference** — ~10–16 FPS is plenty for a room but is not real-time video
   analytics at 30 FPS.
6. **No authentication** — it is a localhost tool; network exposure would need auth.

## 10. Suggested next steps / questions for discussion

- Should the queue support **multiple doctor rooms** (one zone + one queue per room)?
- **Person re-identification** at the door (cheap option: colour-histogram matching;
  proper option: Re-ID embedding model) to verify the right patient entered?
- **SMS/QR token check-in** so patients register from their phone?
- Integration with the hospital's existing registration system (the CSV export is the
  current hand-off point)?
- Hardware plan for a real deployment (camera placement, ceiling angle, lighting,
  a fanless mini-PC next to the TV)?

---

*Quick start for the demo: double-click `run.bat` → open the three URLs → on the Nurse
desk click "Add 5 demo patients" → on the Monitor load a clip and draw/auto-detect the
zones → watch the queue move as patients walk in and out of the doctor room.*
