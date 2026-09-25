# Eagle Eye — Face Recognition, Face Locking and Expression Tracking

A CPU-only face recognition system built for the Intelligent Robotics course. A webcam
feeds a vision pipeline that finds faces, works out who each one is, locks onto one chosen
person, and reports where their nose is relative to the center of the frame and what
expression they are making. An optional motorized camera mount (ESP8266/ESP32 servo,
driven over MQTT) can physically sweep and hold on a recognized face.

The project has two ways to run, sharing the same vision pipeline:

| Mode | Command | What it is for |
|---|---|---|
| **Software-only** (no hardware) | `recognize_live`, `face_tracking` | Recognition on screen, target locking, nose tracking, expressions, missing-face warning |
| **Turret** (needs the servo board) | `recognize` | Sweeps a servo through preset angles until it sees a known face, then holds on it |

## What it does

- **Recognizes people** against an enrolled database using ArcFace embeddings.
- **Labels every visible face** (`recognize_live`): enrolled people in green with their
  name, everyone else as **Unknown** in red, including small, far-away faces.
- **Locks onto one person** (`face_tracking`): finds the chosen name among all visible
  faces, follows them, ignores everyone else, and keeps checking it is still the right
  person.
- **Measures the nose tip** relative to the center of the frame: direction
  (`UP-LEFT`, `CENTER`, ...) and the offset in pixels.
- **Reads expressions:** smile, frown, sad, grimace, blinking and eyes closed.
- **Warns when the face is gone:** a flashing red/orange **FACE MISSING** banner.

## How it works

```
Camera frame (default 1280x720)
   -> MediaPipe FaceLandmarker on the full frame: up to 10 faces, 5 key landmarks each
   -> Far-face pass: Haar cascade proposes small candidates, the landmark model
      must confirm each one (so objects are not boxed as faces)
   -> Similarity alignment / warp of each face to a canonical 112x112 pose
   -> ArcFace ONNX embedding (512-d identity vector, L2-normalized)
   -> Cosine distance to every enrolled identity
   -> Name if distance <= threshold, otherwise "Unknown"
```

Aligning every face to the same pose before embedding is what makes the matching
threshold meaningful; without it, head pose alone would move the embedding more than
identity does.

**Face locking** (`face_tracking.py`) is a small state machine:

```
SEARCHING  -> target recognized among the visible faces ->  LOCKED
LOCKED     -> target no longer found                     ->  LOST
LOST       -> found again (identity re-verified)         ->  LOCKED
LOST       -> not found for 24 frames                    ->  SEARCHING
```

While locked, the same face is followed frame to frame by box overlap plus movement, and
its identity is re-verified every 10 frames so the lock cannot silently jump to someone
else.

**Nose tracking.** The nose tip landmark is smoothed (exponential average) and compared
with the frame center. Output: direction, `dx` / `dy` in pixels (+ is right/down) and the
straight-line distance. A small dead zone in the middle counts as `CENTER`.

**Expressions.** Blink and eyes-closed come from the eye aspect ratio (EAR) of the face
mesh. Smile comes from mouth width and mouth-corner lift. Frown, sad and grimace come
from MediaPipe blendshape scores:

| Expression | Driven by |
|---|---|
| Frown | brows pulled down + mouth corners down |
| Sad | inner brows raised + mouth corners down |
| Grimace | mouth stretched sideways + eyes squinted + nose wrinkled |

Each score is smoothed and uses on/off hysteresis so labels do not flicker. The live
percentage for each is printed at the bottom of the window.

## Quick start

Run everything from the `face-recognition-5pt` folder.

```powershell
cd face-recognition-5pt

# Pick the camera (0 = built-in, 1 = usually the external USB camera)
$env:FALCON_CAM_INDEX = "1"

py -m src.camera                       # confirm the right camera opens
py -m src.enroll                       # enroll people (needed once)
py -m src.recognize_live               # label every face: name or Unknown
py -m src.face_tracking --target Happy # lock onto one enrolled person
```

Both demos also accept `--camera 1` directly. To change the default for every script,
edit `DEFAULT_CAM_INDEX` in `src/cam_config.py`.

### `recognize_live` — everyone in view

Boxes every face it can find (up to 10). Enrolled people are green with their name;
everyone else is red **Unknown**. Faces too small to identify reliably (eyes under 20 px
apart) are still boxed and shown as Unknown ("too far to identify"), because a blurry
crop can match an enrolled person by chance.

Keys: `q` quit, `r` reload database, `+` / `-` adjust the threshold, `d` debug overlay.
Options: `--camera`, `--width`, `--height`, `--threshold`.

### `face_tracking` — one locked person

Locks onto `--target`. Other people are drawn as gray "other (ignored)" boxes and never
affect the lock. While locked it shows the nose offset, the expression label, the
blink count and the per-expression scores. When the target is not locked it shows the
flashing **FACE MISSING** banner.

Options: `--target <name>` (required), `--camera`, `--width`, `--height`,
`--threshold`. Key: `q` quit.

### Turret mode

```powershell
py -m src.recognize     # needs the servo board online and reachable over MQTT
```

## Demo checklist

| Requirement | How to show it |
|---|---|
| External HD camera | Run with `--camera 1`; the window and terminal show the camera index and resolution in use |
| Recognizes you without confusing others | `recognize_live` with a second person in frame: you in green, them Unknown |
| Other faces in the background | `recognize_live` labels them Unknown; `face_tracking` ignores them |
| Nose tip distance and direction from center | `face_tracking`: move your head, watch `NOSE <direction> dx dy dist` and the line from the center |
| Smile, frown, sad, blink, grimace | `face_tracking`: make each expression, watch the label and the percentages |
| Red/orange warning when the face is missing | `face_tracking`: cover your face or step out of frame |

## Recognition threshold and accuracy

A match is accepted when cosine distance is at or below the threshold. The default is
**0.73**, chosen with `py -m src.evaluate` on the enrolled data:

| Threshold | False accepts (FAR) | Genuine faces rejected (FRR) |
|---|---|---|
| 0.65 | 0.04% | 21% |
| 0.73 | 0.8% | 20% |
| 0.81 (old default) | roughly 10-20% (estimated between the 0.76 and 0.87 sweep points) | ~16% |

Lower is stricter. Re-tune it after re-enrolling, and enroll with the same camera and
lighting you will demo with, since that improves accuracy more than the threshold does.
The figures come from the enrollment crops themselves, so real-world error is somewhat
higher.

## Known limits

- **Distance:** at 1280x720, faces are found down to a box about 60 px wide, roughly
  3 m from a typical webcam. Smaller faces are not detected.
- **Head angle:** faces turned well sideways or partly hidden are not detected, so they
  get no box until they face the camera again.
- **Small faces are labeled Unknown, not identified.** This is deliberate.
- **Expression thresholds** are starting values. Tune them in `expression_scores()` in
  `src/face_signals.py` by watching the on-screen percentages.
- **Speed:** roughly 15-20 fps in `recognize_live` on a laptop CPU.

## Setup

1. **Virtual environment**
   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
2. **Dependencies**
   ```powershell
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
   at `models/face_landmarker.task`. Blendshapes (frown/sad/grimace) need this model.
5. **Turret only:** set real WiFi / MQTT values in a local, gitignored copy of
   `src/main.py` or `sketch_sep11a.ino` before flashing the board.

Each pipeline stage can be checked on its own:

```powershell
py -m src.camera      # webcam
py -m src.detect      # face detection
py -m src.landmarks   # 5-point landmarks
py -m src.align       # alignment / warp
py -m src.embed       # embedding
py -m src.evaluate    # threshold sweep (needs 2+ enrolled identities)
```

## Turret: how the two sides communicate

The Python side never drives GPIO pins, and the microcontroller never runs recognition.
They coordinate only through an MQTT broker, so either side can be restarted or replaced
independently.

| Topic | Direction | Meaning |
|---|---|---|
| `falcon/eye/servo/cmd` | desktop -> board | `ANGLE:<0-180>`, `STOP`, `HOME` |
| `falcon/eye/servo/status` | board -> desktop | current angle / motion state |
| `falcon/eye/recognition` | desktop -> broker | recognition events (who / stranger) |
| `falcon/eye/status/<client_id>` | board -> broker | online/offline presence (MQTT last-will) |

If nobody recognized is in frame, the turret sweeps through preset angles. When it
recognizes a face it stops and holds that position, and resumes sweeping if the face is
lost. Two firmware options implement the same command set:

- `src/main.py` — MicroPython, ESP8266
- `sketch_sep11a/sketch_sep11a.ino` — Arduino C++, ESP32

## Repository layout

```
├── README.md
├── face-recognition-5pt/
│   ├── data/
│   │   ├── enroll/<name>/         # enrollment crops per identity (real faces, do not publish)
│   │   ├── debug_aligned/         # alignment debug snapshots
│   │   └── db/                    # face_db.npz / face_db.json (enrolled embeddings)
│   ├── models/
│   │   ├── embedder_arcface.onnx  # ~166 MB, downloaded separately
│   │   └── face_landmarker.task   # MediaPipe landmark + blendshape model
│   ├── src/
│   │   ├── recognize_live.py      # multi-face recognition, far-face pass, Unknown labels
│   │   ├── face_tracking.py       # face locking, nose tracking, missing-face banner
│   │   ├── face_signals.py        # blink, eyes closed, smile, frown, sad, grimace
│   │   ├── recognize.py           # turret loop: recognition + MQTT + servo logic
│   │   ├── enroll.py              # enrollment tool
│   │   ├── evaluate.py            # threshold tuning (FAR/FRR sweep)
│   │   ├── camera.py              # camera opening helper + webcam test
│   │   ├── cam_config.py          # default camera index / FALCON_CAM_INDEX
│   │   ├── detect.py, landmarks.py, align.py, embed.py, haar_5pt.py
│   │   │                          # pipeline stages and their sanity checks
│   │   └── main.py                # ESP8266 firmware (flash to the board, not the PC)
│   ├── init_project.py
│   ├── run.bat, run.ps1           # launchers for the turret scanner
└── sketch_sep11a/sketch_sep11a.ino  # ESP32 firmware alternative
```

## Privacy and secrets

- `data/enroll/`, `data/debug_aligned/` and `data/db/` hold real people's face crops and
  embeddings. Do not publish them.
- WiFi credentials in `src/main.py` and `sketch_sep11a.ino` are placeholders; put real
  values only in a local copy.
- `models/embedder_arcface.onnx` is about 166 MB, over GitHub's 100 MB file limit, so
  keep it out of any repository and download it per the setup steps.

## References

- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin
  Loss for Deep Face Recognition.* CVPR 2019.
- InsightFace Project — 2D & 3D Face Analysis.
- ONNX / ONNX Runtime documentation.
- Lugaresi, C., Tang, J., Nash, H., et al. (2019). *MediaPipe: A Framework for Building
  Perception Pipelines.*
