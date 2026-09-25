# src/face_signals.py
"""
Facial signals extractor (EAR, blinks, eyes closed, smile, frown, sad, grimace).

Smile and blink/eye state come from face-mesh geometry. Frown, sad and grimace
come from MediaPipe's blendshape scores (needs the Tasks API landmarker).

Usage:
    extractor = FaceSignalExtractor()
    signals = extractor.analyze(frame, bbox=(x1, y1, x2, y2))
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except Exception as exc:
    mp = None
    _MP_IMPORT_ERROR = exc
else:
    _MP_IMPORT_ERROR = None

# MediaPipe Face Mesh landmark indices
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
LIP_TOP, LIP_BOTTOM = 13, 14
FACE_LEFT, FACE_RIGHT = 234, 454


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def eye_aspect_ratio(points: np.ndarray, idx: Tuple[int, ...]) -> float:
    """Calculate Eye Aspect Ratio (EAR) for a single eye given landmark points."""
    p1, p2, p3, p4, p5, p6 = (points[i] for i in idx)
    width = max(distance(p1, p4), 1e-6)
    return float((distance(p2, p6) + distance(p3, p5)) / (2.0 * width))


def _norm(value: float, lo: float, hi: float) -> float:
    """Map value from [lo, hi] to [0, 1], clipped."""
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def _avg(scores: Dict[str, float], *names: str) -> float:
    return sum(scores.get(n, 0.0) for n in names) / len(names)


def expression_scores(blend: Dict[str, float]) -> Dict[str, float]:
    """
    Turn MediaPipe blendshape scores into 0..1 frown / sad / grimace scores.

    frown   - brows pulled down + mouth corners down
    sad     - inner brows raised + mouth corners down
    grimace - mouth stretched sideways + eyes squinted + nose wrinkled
    """
    brow_down = _avg(blend, "browDownLeft", "browDownRight")
    brow_inner_up = blend.get("browInnerUp", 0.0)
    mouth_frown = _avg(blend, "mouthFrownLeft", "mouthFrownRight")
    stretch = _avg(blend, "mouthStretchLeft", "mouthStretchRight")
    squint = _avg(blend, "eyeSquintLeft", "eyeSquintRight")
    sneer = _avg(blend, "noseSneerLeft", "noseSneerRight")

    return {
        "frown": 0.6 * _norm(brow_down, 0.15, 0.6) + 0.4 * _norm(mouth_frown, 0.1, 0.5),
        "sad": 0.5 * _norm(brow_inner_up, 0.1, 0.5) + 0.5 * _norm(mouth_frown, 0.1, 0.5),
        "grimace": (
            0.4 * _norm(stretch, 0.15, 0.6)
            + 0.3 * _norm(squint, 0.2, 0.7)
            + 0.3 * _norm(sneer, 0.15, 0.6)
        ),
    }


@dataclass
class FaceSignals:
    """Data object containing extracted facial signals."""
    ear: float
    blink: bool
    eyes_closed: bool
    smile_score: float
    smiling: bool
    frown_score: float = 0.0
    frowning: bool = False
    sad_score: float = 0.0
    sad: bool = False
    grimace_score: float = 0.0
    grimacing: bool = False

    @property
    def expression(self) -> str:
        """Single label for the strongest active expression."""
        if self.smiling:
            return "SMILE"
        active = [
            (self.grimace_score, "GRIMACE") if self.grimacing else None,
            (self.sad_score, "SAD") if self.sad else None,
            (self.frown_score, "FROWN") if self.frowning else None,
        ]
        active = [a for a in active if a is not None]
        return max(active)[1] if active else "NEUTRAL"


class FaceSignalExtractor:
    """Extractor for real-time facial signals such as EAR, blink detection, and smile score."""

    def __init__(
        self,
        ear_threshold: float = 0.22,
        blink_min_frames: int = 1,
        blink_max_frames: int = 8,
        closed_frames: int = 9,
        smile_on: float = 0.45,
        smile_off: float = 0.35,
        expr_on: float = 0.50,
        expr_off: float = 0.35,
        expr_alpha: float = 0.5,
        model_asset_path: str = "models/face_landmarker.task",
    ) -> None:
        self.ear_threshold = float(ear_threshold)
        self.blink_min_frames = int(blink_min_frames)
        self.blink_max_frames = int(blink_max_frames)
        self.closed_frames = int(closed_frames)
        self.smile_on = float(smile_on)
        self.smile_off = float(smile_off)
        self.expr_on = float(expr_on)
        self.expr_off = float(expr_off)
        self.expr_alpha = float(expr_alpha)

        self.low_ear_frames = 0
        self.smiling = False
        self._expr_ema: Dict[str, float] = {}
        self._expr_active: Dict[str, bool] = {}

        if mp is None:
            raise RuntimeError(
                f"MediaPipe import failed: {_MP_IMPORT_ERROR}\n"
                "Install MediaPipe with: pip install -U mediapipe"
            )

        self._landmarker = None
        self._mesh = None

        # Primary: MediaPipe Tasks API
        try:
            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker
            FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
            RunningMode = mp.tasks.vision.RunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_asset_path),
                running_mode=RunningMode.IMAGE,
                num_faces=5,
                min_face_detection_confidence=0.3,
                min_face_presence_confidence=0.3,
                output_face_blendshapes=True,
            )
            self._landmarker = FaceLandmarker.create_from_options(options)
        except Exception:
            # Fallback: legacy mp.solutions.face_mesh
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                self._mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3,
                )

    def reset(self) -> None:
        """Reset temporal state counters."""
        self.low_ear_frames = 0
        self.smiling = False
        self._expr_ema.clear()
        self._expr_active.clear()

    def _update_expressions(
        self, blend: Optional[Dict[str, float]]
    ) -> Dict[str, Tuple[float, bool]]:
        """Smooth blendshape-derived scores and apply on/off hysteresis."""
        raw = expression_scores(blend) if blend else {"frown": 0.0, "sad": 0.0, "grimace": 0.0}
        out: Dict[str, Tuple[float, bool]] = {}
        for name, value in raw.items():
            prev = self._expr_ema.get(name, value)
            score = self.expr_alpha * value + (1.0 - self.expr_alpha) * prev
            self._expr_ema[name] = score
            active = self._expr_active.get(name, False)
            active = score >= (self.expr_off if active else self.expr_on)
            self._expr_active[name] = active
            out[name] = (float(score), active)
        return out

    def close(self) -> None:
        """Release underlying MediaPipe landmarker resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._mesh is not None:
            self._mesh.close()
            self._mesh = None

    def analyze_points(
        self,
        points: np.ndarray,
        blendshapes: Optional[Dict[str, float]] = None,
    ) -> FaceSignals:
        """Extract signals from face mesh landmark points (N, 2) and optional blendshapes."""
        left_ear = eye_aspect_ratio(points, LEFT_EYE)
        right_ear = eye_aspect_ratio(points, RIGHT_EYE)
        ear = 0.5 * (left_ear + right_ear)

        blink = False
        if ear < self.ear_threshold:
            self.low_ear_frames += 1
        else:
            if self.blink_min_frames <= self.low_ear_frames <= self.blink_max_frames:
                blink = True
            self.low_ear_frames = 0

        eyes_closed = self.low_ear_frames >= self.closed_frames

        # Pose-invariant and mouth-shape-invariant smile confidence score:
        # Combines normalized mouth width ratio AND upward mouth corner lift ratio
        left_eye_pt = points[33]
        right_eye_pt = points[263]
        eye_distance = max(distance(left_eye_pt, right_eye_pt), 1e-6)
        mouth_width = distance(points[MOUTH_LEFT], points[MOUTH_RIGHT])

        # Width expansion ratio
        width_ratio = mouth_width / eye_distance
        width_score = (width_ratio - 0.48) / (0.62 - 0.48)

        # Corner elevation ratio (upward corner lift relative to upper lip)
        corner_avg_y = float((points[MOUTH_LEFT][1] + points[MOUTH_RIGHT][1]) / 2.0)
        lip_top_y = float(points[LIP_TOP][1])
        lift_ratio = (lip_top_y - corner_avg_y) / eye_distance
        lift_score = (lift_ratio - 0.01) / (0.10 - 0.01)

        # Combined smile confidence score
        combined = 0.6 * width_score + 0.4 * lift_score
        smile_score = float(np.clip(combined, 0.0, 1.0))

        if self.smiling:
            self.smiling = smile_score >= self.smile_off
        else:
            self.smiling = smile_score >= self.smile_on

        expr = self._update_expressions(blendshapes)

        return FaceSignals(
            ear=float(ear),
            blink=blink,
            eyes_closed=eyes_closed,
            smile_score=smile_score,
            smiling=self.smiling,
            frown_score=expr["frown"][0],
            frowning=expr["frown"][1] and not self.smiling,
            sad_score=expr["sad"][0],
            sad=expr["sad"][1] and not self.smiling,
            grimace_score=expr["grimace"][0],
            grimacing=expr["grimace"][1],
        )

    def analyze(
        self,
        frame: np.ndarray,
        bbox: Optional[Sequence[int]] = None,
    ) -> Optional[FaceSignals]:
        """Extract facial signals using full-frame MediaPipe landmark detection."""
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        target_center = None
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox[:4])
            target_center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

        matched_points = None
        matched_blend: Optional[Dict[str, float]] = None

        if self._landmarker is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            res = self._landmarker.detect(mp_image)
            blends = res.face_blendshapes or []
            best_dist = float("inf")
            for i, face_lms in enumerate(res.face_landmarks or []):
                pts = np.array([[lm.x * w, lm.y * h] for lm in face_lms], dtype=np.float32)
                dist = 0.0
                if target_center is not None:
                    dist = float(np.linalg.norm(np.mean(pts, axis=0) - target_center))
                if dist < best_dist:
                    best_dist = dist
                    matched_points = pts
                    matched_blend = (
                        {c.category_name: float(c.score) for c in blends[i]}
                        if i < len(blends)
                        else None
                    )

        elif self._mesh is not None:
            res = self._mesh.process(frame_rgb)
            if res.multi_face_landmarks:
                best_dist = float("inf")
                for face_lms in res.multi_face_landmarks:
                    pts = np.array([[p.x * w, p.y * h] for p in face_lms.landmark], dtype=np.float32)
                    if target_center is None:
                        matched_points = pts
                        break
                    center = np.mean(pts, axis=0)
                    dist = float(np.linalg.norm(center - target_center))
                    if dist < best_dist:
                        best_dist = dist
                        matched_points = pts

        if matched_points is None or len(matched_points) == 0:
            return None

        return self.analyze_points(matched_points, matched_blend)