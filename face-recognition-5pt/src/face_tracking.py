# src/face_tracking.py
"""
Target-locked face tracking pipeline with real-time signal analysis (EAR, blinks, expressions).

Usage:
    python -m src.face_tracking --target <name>
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from .align import align_face_5pt
from .embed import ArcFaceEmbedderONNX
from .face_signals import FaceSignalExtractor
from .cam_config import get_cam_index
from .camera import open_camera
from .recognize_live import (
    DEFAULT_THRESHOLD,
    FaceDBMatcher,
    HaarFaceLandmarker5pt,
    load_db_npz,
)


class LockState(Enum):
    SEARCHING = auto()
    LOCKED = auto()
    LOST = auto()


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Calculate Intersection over Union (IoU) between two XYXY boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def center(box: Tuple[int, int, int, int]) -> np.ndarray:
    """Return center (x, y) coordinates of an XYXY box."""
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)


@dataclass
class TrackingSignal:
    """Nose-tip position relative to the frame center."""

    error_x: float  # normalized, -1 (left edge) .. +1 (right edge)
    error_y: float  # normalized, -1 (top edge) .. +1 (bottom edge)
    horizontal: str
    vertical: str
    offset_x_px: float  # nose tip minus frame center, pixels (+ = right)
    offset_y_px: float  # nose tip minus frame center, pixels (+ = down)
    distance_px: float
    nose_xy: Tuple[float, float]

    @property
    def direction(self) -> str:
        parts = [p for p in (self.vertical, self.horizontal) if p != "CENTER"]
        return "-".join(parts) if parts else "CENTER"


class LockedFaceTracker:
    def __init__(
        self,
        target_name: str,
        detector,
        embedder,
        matcher,
        verify_every: int = 10,
        lost_timeout: int = 24,
        ema_alpha: float = 0.30,
        dead_zone: float = 0.07,
    ):
        self.target_name = target_name
        self.detector = detector
        self.embedder = embedder
        self.matcher = matcher
        self.verify_every = verify_every
        self.lost_timeout = lost_timeout
        self.ema_alpha = ema_alpha
        self.dead_zone = dead_zone

        self.state = LockState.SEARCHING
        self.last_box: Optional[Tuple[int, int, int, int]] = None
        self.smooth_center: Optional[np.ndarray] = None  # smoothed nose tip
        self.lost_frames = 0
        self.frame_index = 0
        self.other_faces: list = []  # detected faces that are not the locked target

    @staticmethod
    def box(face) -> Tuple[int, int, int, int]:
        return (face.x1, face.y1, face.x2, face.y2)

    def identity(self, frame, face):
        aligned, _ = align_face_5pt(frame, face.kps, out_size=(112, 112))
        return self.matcher.match(self.embedder.embed(aligned))

    def target_is_verified(self, frame, face) -> bool:
        match = self.identity(frame, face)
        return bool(match.accepted and match.name == self.target_name)

    def acquire(self, frame, faces):
        best = None
        best_similarity = -1.0
        for face in faces:
            match = self.identity(frame, face)
            if (
                match.accepted
                and match.name == self.target_name
                and match.similarity > best_similarity
            ):
                best, best_similarity = face, match.similarity
        return best

    def associate(self, faces):
        if self.last_box is None or not faces:
            return None
        last_center = center(self.last_box)
        last_diag = max(
            np.linalg.norm(
                np.array(
                    [
                        self.last_box[2] - self.last_box[0],
                        self.last_box[3] - self.last_box[1],
                    ],
                    dtype=np.float32,
                )
            ),
            1.0,
        )
        ranked = []
        for face in faces:
            box = self.box(face)
            overlap = iou(self.last_box, box)
            displacement = np.linalg.norm(center(box) - last_center) / last_diag
            score = overlap - 0.35 * displacement
            ranked.append((score, face))

        if not ranked:
            return None

        score, candidate = max(ranked, key=lambda item: item[0])
        return candidate if score > -0.30 else None

    def update(self, frame):
        self.frame_index += 1
        faces = self.detector.detect(frame, max_faces=8)

        if self.state == LockState.SEARCHING:
            candidate = self.acquire(frame, faces)
        else:
            candidate = self.associate(faces)

        if (
            candidate is not None
            and (
                self.state == LockState.LOST
                or self.frame_index % self.verify_every == 0
            )
            and not self.target_is_verified(frame, candidate)
        ):
            candidate = None

        self.other_faces = [f for f in faces if f is not candidate]

        if candidate is None:
            self.lost_frames += 1
            if self.last_box is not None:
                self.state = LockState.LOST
            if self.lost_frames > self.lost_timeout:
                self.state = LockState.SEARCHING
                self.last_box = None
                self.smooth_center = None
            return None, None

        self.state = LockState.LOCKED
        self.lost_frames = 0
        self.last_box = self.box(candidate)
        raw_center = candidate.kps[2].astype(np.float32)  # nose tip

        if self.smooth_center is None:
            self.smooth_center = raw_center
        else:
            a = self.ema_alpha
            self.smooth_center = a * raw_center + (1.0 - a) * self.smooth_center

        return candidate, self.position_signal(frame.shape)

    def position_signal(self, shape) -> TrackingSignal:
        height, width = shape[:2]
        dx = float(self.smooth_center[0] - width / 2.0)
        dy = float(self.smooth_center[1] - height / 2.0)
        ex = dx / (width / 2.0)
        ey = dy / (height / 2.0)
        horizontal = "CENTER"
        vertical = "CENTER"

        if ex < -self.dead_zone:
            horizontal = "LEFT"
        elif ex > self.dead_zone:
            horizontal = "RIGHT"

        if ey < -self.dead_zone:
            vertical = "UP"
        elif ey > self.dead_zone:
            vertical = "DOWN"

        return TrackingSignal(
            error_x=ex,
            error_y=ey,
            horizontal=horizontal,
            vertical=vertical,
            offset_x_px=dx,
            offset_y_px=dy,
            distance_px=float(np.hypot(dx, dy)),
            nose_xy=(float(self.smooth_center[0]), float(self.smooth_center[1])),
        )


def draw_label(frame, text, xy, color, scale=0.62):
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA
    )
    cv2.putText(
        frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )


def draw_missing_banner(view, target: str, others: int, lost: bool) -> None:
    """Flashing red/orange 'face missing' warning across the frame."""
    h, w = view.shape[:2]
    flash = int(time.time() * 2) % 2 == 0
    color = (0, 0, 255) if flash else (0, 140, 255)  # red <-> orange (BGR)

    cv2.rectangle(view, (0, 0), (w - 1, h - 1), color, 8)

    band_h = max(70, h // 8)
    y0 = h // 2 - band_h // 2
    overlay = view.copy()
    cv2.rectangle(overlay, (0, y0), (w, y0 + band_h), color, -1)
    cv2.addWeighted(overlay, 0.75, view, 0.25, 0, view)

    title = "FACE MISSING" if lost else "SEARCHING FOR FACE"
    if others:
        sub = f"{target} not found - {others} other face(s) ignored"
    else:
        sub = f"{target} not in frame"
    (tw, _), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
    cv2.putText(view, title, ((w - tw) // 2, y0 + int(band_h * 0.55)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4, cv2.LINE_AA)
    (sw, _), _ = cv2.getTextSize(sub, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(view, sub, ((w - sw) // 2, y0 + int(band_h * 0.88)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="enrolled identity to lock")
    parser.add_argument("--camera", type=int, default=get_cam_index())
    parser.add_argument("--width", type=int, default=1280, help="requested capture width")
    parser.add_argument("--height", type=int, default=720, help="requested capture height")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    detector = HaarFaceLandmarker5pt(min_size=(70, 70), debug=False)
    embedder = ArcFaceEmbedderONNX(
        model_path="models/embedder_arcface.onnx",
        input_size=(112, 112),
        debug=False,
    )
    matcher = FaceDBMatcher(
        load_db_npz(Path("data/db/face_db.npz")),
        dist_thresh=args.threshold,
    )

    tracker = LockedFaceTracker(args.target, detector, embedder, matcher)
    signals = FaceSignalExtractor()

    cap, cam_used, (cam_w, cam_h) = open_camera(args.camera, args.width, args.height)
    cam_text = f"cam {cam_used} {cam_w}x{cam_h}"

    blink_total = 0
    blink_flash = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            locked_face, position = tracker.update(frame)
            view = frame.copy()
            h, w = view.shape[:2]

            # Other faces are deliberately ignored - show that instead of hiding them.
            for other in tracker.other_faces:
                cv2.rectangle(view, (other.x1, other.y1), (other.x2, other.y2), (150, 150, 150), 1)
                draw_label(view, "other (ignored)", (other.x1, max(14, other.y1 - 6)), (170, 170, 170), 0.5)

            state_text = f"{tracker.state.name}: {args.target}"
            state_color = (0, 180, 0) if locked_face is not None else (0, 140, 255)
            draw_label(view, state_text, (12, 28), state_color, 0.72)
            draw_label(view, cam_text, (w - 230, 28), (200, 200, 200), 0.55)

            if locked_face is not None:
                box = tracker.box(locked_face)
                x1, y1, x2, y2 = box
                cv2.rectangle(view, (x1, y1), (x2, y2), (255, 170, 0), 3)

                face_state = signals.analyze(frame, box)
                if face_state is not None:
                    if face_state.blink:
                        blink_total += 1
                        blink_flash = 6
                    blink_flash = max(0, blink_flash - 1)

                    eye_text = "EYES CLOSED" if face_state.eyes_closed else "EYES OPEN"
                    if blink_flash:
                        eye_text = "BLINK!"

                    expr_color = (0, 255, 255) if face_state.expression != "NEUTRAL" else (200, 200, 200)
                    draw_label(view, face_state.expression, (x1, max(55, y1 - 50)), expr_color)
                    draw_label(
                        view,
                        f"{eye_text} blinks={blink_total}",
                        (x1, max(78, y1 - 25)),
                        (255, 255, 0),
                    )
                    draw_label(
                        view,
                        f"EAR={face_state.ear:.3f} smile={face_state.smile_score * 100:.0f}% "
                        f"frown={face_state.frown_score * 100:.0f}% "
                        f"sad={face_state.sad_score * 100:.0f}% "
                        f"grimace={face_state.grimace_score * 100:.0f}%",
                        (12, h - 18),
                        (255, 255, 255),
                        0.52,
                    )

                if position is not None:
                    nx, ny = int(position.nose_xy[0]), int(position.nose_xy[1])
                    cv2.line(view, (w // 2, h // 2), (nx, ny), (255, 170, 0), 2, cv2.LINE_AA)
                    cv2.circle(view, (nx, ny), 6, (0, 255, 255), -1, cv2.LINE_AA)
                    draw_label(
                        view,
                        f"NOSE {position.direction}  "
                        f"dx={position.offset_x_px:+.0f}px dy={position.offset_y_px:+.0f}px "
                        f"dist={position.distance_px:.0f}px",
                        (12, 56),
                        (255, 170, 0),
                        0.60,
                    )
            else:
                signals.reset()
                blink_flash = 0
                draw_missing_banner(
                    view,
                    args.target,
                    len(tracker.other_faces),
                    lost=tracker.state == LockState.LOST,
                )

            # Frame-center crosshair.
            cv2.drawMarker(view, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 24, 1)
            dz = tracker.dead_zone
            cv2.rectangle(
                view,
                (int(w * (0.5 - dz / 2)), int(h * (0.5 - dz / 2))),
                (int(w * (0.5 + dz / 2)), int(h * (0.5 + dz / 2))),
                (120, 120, 120),
                1,
            )

            cv2.imshow("Locked Face Tracking", view)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        cap.release()
        signals.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()