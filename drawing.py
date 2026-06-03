# ╔══════════════════════════════════════════════════════════════════╗
# ║  drawing.py — Skeleton overlay and HUD panel rendering          ║
# ╚══════════════════════════════════════════════════════════════════╝

import cv2
from landmarks import SKELETON_CONNECTIONS
from config import DEPTH_INSUFFICIENT_ANGLE

# Colour palette (BGR)
C = {
    "green":  (0, 220, 100),
    "red":    (30, 60, 240),
    "yellow": (0, 200, 255),
    "white":  (240, 240, 240),
    "accent": (0, 165, 255),
    "dim":    (100, 100, 130),
    "panel":  (12, 12, 22),
    "shadow": (15, 15, 15),
}


def draw_skeleton(frame, pts: dict, color: tuple):
    for a, b in SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            pa = (int(pts[a][0]), int(pts[a][1]))
            pb = (int(pts[b][0]), int(pts[b][1]))
            cv2.line(frame, pa, pb, C["shadow"], 5, cv2.LINE_AA)
            cv2.line(frame, pa, pb, color, 3, cv2.LINE_AA)
    for pt in pts.values():
        p = (int(pt[0]), int(pt[1]))
        cv2.circle(frame, p, 7, C["shadow"], -1)
        cv2.circle(frame, p, 6, color, -1)
        cv2.circle(frame, p, 8, C["white"], 1)




