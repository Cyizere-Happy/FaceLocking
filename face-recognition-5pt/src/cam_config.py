# src/cam_config.py
"""
Camera index is not portable across machines - it depends on USB port and
driver enumeration order, so a value hardcoded for one laptop can silently
open the wrong device (or nothing) on another. Set FALCON_CAM_INDEX in the
environment to override the default without touching any script:

    PowerShell:  $env:FALCON_CAM_INDEX = "1"
"""

import os

DEFAULT_CAM_INDEX = 0


def get_cam_index() -> int:
    return int(os.environ.get("FALCON_CAM_INDEX", DEFAULT_CAM_INDEX))
