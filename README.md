# Eagle Eye — Face Recognition Turret

A lightweight, CPU-only face recognition system paired with a motorized camera mount
that physically tracks whether it's looking at someone it knows. The desktop side
handles all the vision work; a microcontroller handles the physical aiming. The two
halves never talk directly to hardware on each other's side — they coordinate purely
through MQTT messages, so either one can be restarted, replaced, or debugged
independently.

## Why it's built this way

Most face-recognition demos stop at "detect a face, print a name." This project goes
one step further: the system has an actual physical behavior in response to what it
sees. If nobody recognized is currently in frame, the turret doesn't just sit idle —
it actively sweeps through a series of preset angles looking for someone it knows.
The moment it recognizes a face, it stops sweeping and holds that position, effectively
"locking on." Lose the face again, and it resumes the search. It's a small example of
closing the loop between perception and action rather than treating them as separate
projects.

The vision pipeline itself is intentionally CPU-friendly — no GPU is assumed, so the
same code runs on a laptop or a lightweight lab machine. It stages the problem the way
most production face-ID systems do:

```
Camera frame
   → Haar cascade face detection (fast, coarse localization)
   → 5-point facial landmark extraction (eyes, nose, mouth corners)
   → Geometric alignment / warp to a fixed 112×112 canonical pose
   → ArcFace ONNX embedding (512-d identity vector)
   → Nearest-neighbor match against an enrolled face database
   → Known name, or "Stranger"
   → Decision relayed over MQTT to the turret controller
```

Aligning every face to the same pose before embedding is what makes the matching
threshold meaningful — without it, pose variation alone would dominate the distance
between embeddings more than actual identity does.

## How the two sides communicate

The Python side never drives GPIO pins directly, and the microcontroller never runs
any recognition logic. All coordination happens over a shared MQTT broker:

| Topic | Direction | Meaning |
|---|---|---|
| `falcon/eye/servo/cmd` | desktop → board | `ANGLE:<0-180>`, `STOP`, `HOME` |
| `falcon/eye/servo/status` | board → desktop | current angle / motion state |
| `falcon/eye/recognition` | desktop → broker | recognition events (who / stranger) |
| `falcon/eye/status/<client_id>` | board → broker | online/offline presence (MQTT last-will) |

Two firmware options are included for the turret controller, depending on the board
you have on hand:

- `face-recognition-5pt/src/main.py` — MicroPython, targets an ESP8266
- `sketch_sep11a/sketch_sep11a.ino` — Arduino C++, targets an ESP32

Both implement the same command surface, so the desktop code doesn't need to know or
care which board is actually running.

## Repository layout

```
├── face-recognition-5pt/
│   ├── data/
│   │   ├── enroll/<name>/       # aligned enrollment crops per identity (gitignored)
│   │   ├── debug_aligned/       # alignment debug snapshots (gitignored)
│   │   └── db/                  # face_db.npz / face_db.json (gitignored)
│   ├── models/
│   │   ├── embedder_arcface.onnx  # ~166 MB, downloaded separately, not committed
│   │   └── face_landmarker.task   # MediaPipe landmark model
│   ├── src/
│   │   ├── camera.py            # webcam smoke test
│   │   ├── detect.py            # Haar detection test
│   │   ├── landmarks.py         # 5-point landmark test
│   │   ├── align.py             # alignment / warp test
│   │   ├── embed.py             # ArcFace embedding test
│   │   ├── enroll.py            # multi-identity enrollment tool
│   │   ├── evaluate.py          # threshold tuning (FAR/FRR sweep)
│   │   ├── recognize.py         # full runtime loop: recognition + MQTT + turret logic
│   │   ├── recognize_live.py    # software-only multi-face recognition (no MQTT / servo)
│   │   ├── face_tracking.py     # target face locking (SEARCHING / LOCKED / LOST) + position signal
│   │   ├── face_signals.py      # EAR, blink, eyes-closed, smile detection (MediaPipe landmarks)
│   │   ├── haar_5pt.py          # detector + alignment math
│   │   └── main.py              # ESP8266 firmware (flash to the board, don't run on desktop)
│   ├── init_project.py
│   └── book/                    # detailed design write-up
└── sketch_sep11a/
    └── sketch_sep11a.ino        # ESP32 firmware alternative
```

## Before pushing anywhere public

- WiFi credentials in `src/main.py` and `sketch_sep11a.ino` are placeholders
  (`WIFI_SSID` / `WIFI_PASS`) — fill in real values only in a local, gitignored copy,
  never in a commit.
- `data/enroll/` and `data/debug_aligned/` contain real people's face crops. Keep them
  gitignored; don't publish biometric data even to demonstrate multi-identity
  enrollment. A short description or screenshot of the flow is a safer substitute.
- `models/embedder_arcface.onnx` is ~166 MB, over GitHub's 100 MB hard limit — keep it
  gitignored and download it fresh per the setup steps below.

## Setup

1. **Virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
   ```
2. **Dependencies**
   ```bash
   pip install opencv-python numpy onnxruntime scipy tqdm mediapipe paho-mqtt
   ```
3. **ArcFace ONNX model**
   ```bash
   curl -L -o buffalo_l.zip "https://sourceforge.net/projects/insightface.mirror/files/v0.7/buffalo_l.zip/download"
   unzip -o buffalo_l.zip
   cp w600k_r50.onnx models/embedder_arcface.onnx
   rm -f buffalo_l.zip w600k_r50.onnx 1k3d68.onnx 2d106det.onnx det_10g.onnx genderage.onnx
   ```
4. **MediaPipe landmark model** — place `face_landmarker.task` (from
   [MediaPipe's model zoo](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker))
   at `models/face_landmarker.task`.
5. **WiFi / MQTT config** — set real values in a local, gitignored copy of `src/main.py`
   or `sketch_sep11a.ino` before flashing the board.

## Running it

Validate each pipeline stage before running the full loop:

```bash
python -m src.camera        # webcam sanity check
python -m src.detect        # face detection sanity check
python -m src.landmarks     # 5-point landmark sanity check
python -m src.align         # alignment/warp sanity check
python -m src.embed         # embedding sanity check

python -m src.enroll        # enroll one or more identities
python -m src.evaluate      # tune the match threshold (needs 2+ identities)

python -m src.recognize     # run the full scanner (needs the turret board online)
```

### Software-only modes (no turret, no MQTT)

```bash
python -m src.recognize_live                    # multi-face recognition window
python -m src.face_tracking --target <name>     # lock onto one enrolled person
```

`recognize_live` labels every face it can see (up to 10): enrolled people in green,
everyone else "Unknown" in red. Small/far faces are found by a second pass in which Haar
proposes candidates and the MediaPipe landmark model must confirm each one, so objects
are not boxed. Faces too small to identify (eyes under 20 px apart) show as "Unknown".
Keys: `q` quit, `r` reload DB, `+`/`-` adjust the distance threshold, `d` toggle debug
overlay. Options: `--camera`, `--width`, `--height`, `--threshold`.

`face_tracking` finds the enrolled `--target` among all visible faces, locks on, and
re-verifies identity every 10 frames. The state machine is
`SEARCHING -> LOCKED -> LOST -> SEARCHING`; the lock survives short dropouts
(24 frames) and follows the same face frame-to-frame by IoU + displacement. While
locked it shows:

- the **nose tip's** offset from the frame center: direction (`UP-LEFT`, `CENTER`, ...),
  `dx` / `dy` / total distance in pixels, and a line from the frame center to the nose;
- facial-expression analysis: `SMILE`, `FROWN`, `SAD`, `GRIMACE` or `NEUTRAL` (frown, sad
  and grimace come from MediaPipe blendshapes), `EYES OPEN` / `EYES CLOSED`, a `BLINK!`
  flash with a running blink count, and the live score for each expression.

When the target isn't locked, a flashing red/orange **FACE MISSING** banner covers the
frame. Other faces are drawn as gray "other (ignored)" boxes and never affect the lock.

Options: `--target <name>` (required), `--camera <index>` (defaults to
`FALCON_CAM_INDEX`; if that index can't be opened it warns and tries 1, 0, 2),
`--width` / `--height` (requested capture size, default 1280x720; the camera index and
actual resolution are shown on screen), `--threshold <dist>` (default 0.73, from
`python -m src.evaluate` on the enrolled data; re-tune it for your own people).

Flash the turret firmware (`src/main.py` on an ESP8266, or `sketch_sep11a.ino` on an
ESP32) and make sure it's connected to the same MQTT broker before starting
`recognize.py` — otherwise there's nothing listening for angle commands.

## Further reading

See [`face-recognition-5pt/book/`](./face-recognition-5pt/book) for the full
walkthrough of the pipeline design and threshold-tuning methodology.

## References

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin
  Loss for Deep Face Recognition.* CVPR 2019.
- InsightFace Project — 2D & 3D Face Analysis.
- ONNX / ONNX Runtime documentation.
- Lugaresi, C., Tang, J., Nash, H., et al. (2019). *MediaPipe: A Framework for Building
  Perception Pipelines.*
