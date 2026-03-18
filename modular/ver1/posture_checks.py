# ╔══════════════════════════════════════════════════════════════════╗
# ║  posture_checks.py — All squat error detection functions        ║
# ║  Each function returns (alert_string_or_None, metric_value)     ║
# ╚══════════════════════════════════════════════════════════════════╝

from config import (
    BACK_LEAN_MAX, KNEE_CAVE_RATIO, KNEE_TOE_X_RATIO,
    HEEL_LIFT_NORM_THRESH, BUTT_WINK_HIP_ANGLE_MIN,
    BUTT_WINK_KNEE_ANGLE_MAX, NECK_ANGLE_MIN,
)
from landmarks import LM
from math_utils import angle_2d, angle_3d, torso_lean


# ── Front-view checks ─────────────────────────────────────────────────────────

def check_posture_front(lms, pts: dict, w: int):
    """
    Checks valid in front-facing view.
    Uses 2D pixel angles (Z is unreliable front-on).
    Returns: (alerts, lean_angle, knee_angle_left, knee_angle_right)
    """
    alerts = []
    need = {
        "left_hip", "left_knee", "left_ankle",
        "right_hip", "right_knee", "right_ankle",
        "left_shoulder", "right_shoulder",
    }
    if not need.issubset(pts):
        return alerts, 0.0, 0.0, 0.0

    kl = angle_2d(pts["left_hip"],  pts["left_knee"],  pts["left_ankle"])
    kr = angle_2d(pts["right_hip"], pts["right_knee"], pts["right_ankle"])

    # Knee valgus (cave): knee width narrows relative to ankle width
    knee_w  = abs(pts["left_knee"][0]  - pts["right_knee"][0])
    ankle_w = abs(pts["left_ankle"][0] - pts["right_ankle"][0])
    if ankle_w > 10 and knee_w < ankle_w * KNEE_CAVE_RATIO:
        alerts.append("Knees caving in — push knees out")

    # Excessive torso lean
    mid_sh  = ((pts["left_shoulder"][0] + pts["right_shoulder"][0]) // 2,
               (pts["left_shoulder"][1] + pts["right_shoulder"][1]) // 2)
    mid_hip = ((pts["left_hip"][0] + pts["right_hip"][0]) // 2,
               (pts["left_hip"][1] + pts["right_hip"][1]) // 2)
    lean = torso_lean(mid_sh, mid_hip)
    if lean > BACK_LEAN_MAX:
        alerts.append("Leaning too far forward — chest up")

    return alerts, lean, kl, kr


# ── Side-view checks ──────────────────────────────────────────────────────────

def check_posture_side(lms, pts: dict, w: int):
    """
    Checks valid in side-facing view.
    Uses 3D angles for knee (depth is reliable sideways).
    Returns: (alerts, lean_angle, knee_angle_left, knee_angle_right)
    """
    alerts   = []
    use_left = lms[LM["left_shoulder"]].z < lms[LM["right_shoulder"]].z
    s        = "left" if use_left else "right"

    need = {f"{s}_hip", f"{s}_knee", f"{s}_ankle", f"{s}_shoulder"}
    if not need.issubset(pts):
        return alerts, 0.0, 0.0, 0.0

    hip_lm  = lms[LM[f"{s}_hip"]]
    knee_lm = lms[LM[f"{s}_knee"]]
    ank_lm  = lms[LM[f"{s}_ankle"]]
    k       = angle_3d(hip_lm, knee_lm, ank_lm)
    kl = kr = k

    # Knee past toes: in side view, X axis is the depth axis
    knee_toe_x = (pts[f"{s}_knee"][0] - pts[f"{s}_ankle"][0]) / w
    if abs(knee_toe_x) > KNEE_TOE_X_RATIO:
        alerts.append("Knee past toes — sit back more")

    # Excessive torso lean
    lean = torso_lean(pts[f"{s}_shoulder"], pts[f"{s}_hip"])
    if lean > BACK_LEAN_MAX:
        alerts.append("Leaning too far forward — chest up")

    return alerts, lean, kl, kr


# ── Shared checks (both views) ────────────────────────────────────────────────

def check_heel_lift(lms, heel_baseline: dict | None):
    """
    Detects heels rising off the ground.
    Compares current heel Y against a baseline captured at rep start.
    MediaPipe Y: 0=top, 1=bottom → rising heel = decreasing Y = positive delta.
    Returns: (alert_string_or_None, max_rise_value)
    """
    if heel_baseline is None:
        return None, 0.0
    left_rise  = heel_baseline["left"]  - lms[LM["left_heel"]].y
    right_rise = heel_baseline["right"] - lms[LM["right_heel"]].y
    max_rise   = max(left_rise, right_rise)
    if max_rise > HEEL_LIFT_NORM_THRESH:
        return "Heels lifting — press heels down", max_rise
    return None, max_rise


def check_neck(lms, view: str):
    """
    Detects neck out of neutral (looking up or down excessively).
    Measures the ear→shoulder→hip angle.
    Neutral: ear stacked above shoulder → angle ~160–180°.
    Craning either direction closes this angle significantly.
    Returns: (alert_string_or_None, neck_angle)
    """
    try:
        if view == "SIDE":
            s  = "left" if lms[LM["left_shoulder"]].z < lms[LM["right_shoulder"]].z else "right"
            na = angle_3d(lms[LM[f"{s}_ear"]], lms[LM[f"{s}_shoulder"]], lms[LM[f"{s}_hip"]])
        else:
            al = angle_3d(lms[LM["left_ear"]],  lms[LM["left_shoulder"]],  lms[LM["left_hip"]])
            ar = angle_3d(lms[LM["right_ear"]], lms[LM["right_shoulder"]], lms[LM["right_hip"]])
            na = (al + ar) / 2
    except Exception:
        return None, 180.0

    if na < NECK_ANGLE_MIN:
        return "Neck not neutral — look straight ahead", na
    return None, na


def check_butt_wink(lms, avg_knee_angle: float):
    """
    Detects posterior pelvic tilt (butt wink) at the bottom of the squat.
    Only meaningful in SIDE view and only deep in the squat.
    Proxy: shoulder→hip→knee angle. Collapses when lumbar rounds and pelvis tucks.
    Returns: (alert_string_or_None, hip_angle)
    """
    if avg_knee_angle > BUTT_WINK_KNEE_ANGLE_MAX:
        return None, 180.0
    s         = "left" if lms[LM["left_shoulder"]].z < lms[LM["right_shoulder"]].z else "right"
    hip_angle = angle_3d(lms[LM[f"{s}_shoulder"]], lms[LM[f"{s}_hip"]], lms[LM[f"{s}_knee"]])
    if hip_angle < BUTT_WINK_HIP_ANGLE_MIN:
        return "Lower back rounding — brace your core", hip_angle
    return None, hip_angle
