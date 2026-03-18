# ╔══════════════════════════════════════════════════════════════════╗
# ║  math_utils.py — Geometry, view detection, landmark projection  ║
# ╚══════════════════════════════════════════════════════════════════╝

import numpy as np
from config import VISIBILITY_THRESH
from landmarks import LM
from smoother import smooth_landmark


def angle_2d(a, b, c) -> float:
    """
    Angle at point B using 2D pixel coordinates (x, y).
    Best used for front-view knee angles where Z depth is unreliable.
    """
    a, b, c = np.array(a[:2], float), np.array(b[:2], float), np.array(c[:2], float)
    ba, bc  = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def angle_3d(a, b, c) -> float:
    """
    Angle at point B using normalised 3D coordinates (x, y, z) from MediaPipe.
    Most reliable for side-view angles where depth is meaningful.
    """
    av  = np.array([a.x, a.y, a.z])
    bv  = np.array([b.x, b.y, b.z])
    cv_ = np.array([c.x, c.y, c.z])
    ba, bc = av - bv, cv_ - bv
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def torso_lean(sh_px, hip_px) -> float:
    """Returns the lean angle (degrees) of the torso from vertical."""
    dx = sh_px[0] - hip_px[0]
    dy = max(hip_px[1] - sh_px[1], 1)
    return float(np.degrees(np.arctan2(abs(dx), dy)))


def detect_view(lms) -> str:
    """
    Returns 'SIDE' if the person is turned sideways, 'FRONT' otherwise.
    Uses the Z-depth difference between shoulders as the signal —
    large Z delta means one shoulder is significantly closer to camera.
    """
    z_diff = abs(lms[LM["left_shoulder"]].z - lms[LM["right_shoulder"]].z)
    return "SIDE" if z_diff > 0.12 else "FRONT"


def build_pts(lms, w: int, h: int) -> dict:
    """
    Build a pixel-coordinate dict for all landmarks that pass VISIBILITY_THRESH.
    Coordinates are EMA-smoothed then projected onto the full-resolution frame.
    Landmarks with low confidence are excluded entirely to prevent phantom joints.
    """
    pts = {}
    for name, idx in LM.items():
        lm = lms[idx]
        if lm.visibility < VISIBILITY_THRESH:
            continue
        sx, sy, _ = smooth_landmark(name, lm)
        pts[name] = (int(sx * w), int(sy * h))
    return pts
