# src/camera.py
from typing import Optional, Sequence, Tuple

import cv2

from .cam_config import get_cam_index


def open_camera(
    preferred: int,
    width: Optional[int] = 1280,
    height: Optional[int] = 720,
    fallbacks: Sequence[int] = (1, 0, 2),
) -> Tuple[cv2.VideoCapture, int, Tuple[int, int]]:
    """
    Open the preferred camera index, falling back to others (loudly).

    Requests width x height with MJPG (needed for HD over USB 2). Returns
    (capture, index_used, (actual_width, actual_height)) so callers can show
    which camera and resolution are really in use.
    """
    tried = []
    for idx in [preferred, *[i for i in fallbacks if i != preferred]]:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            tried.append(idx)
            continue

        if idx != preferred:
            print(f"[camera] WARNING: index {preferred} unavailable, using index {idx}")

        if width and height:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        actual = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        print(f"[camera] using index {idx} at {actual[0]}x{actual[1]}")
        return cap, idx, actual

    raise RuntimeError(f"Camera not available (tried {tried})")


def main():
    cap = cv2.VideoCapture(get_cam_index())
    if not cap.isOpened():
        raise RuntimeError("Camera not opened. Try changing index (0/1/2).")

    print("Camera test. Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        cv2.imshow("Camera Test", frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
