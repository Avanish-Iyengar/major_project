# ╔══════════════════════════════════════════════════════════════════╗
# ║  exercises/squat.py — Barbell / Bodyweight Squat                ║
# ║                                                                  ║
# ║  Run from the project root:                                     ║
# ║    python exercises/squat.py                                    ║
# ║  Or via the menu:                                               ║
# ║    python main.py                                               ║
# ╚══════════════════════════════════════════════════════════════════╝

from exercise_base import ExerciseDefinition, JointCheck, RepTrigger
from exercise_runner import run_exercise
from landmarks import LM

from math_utils import angle_3d, torso_lean, detect_view





# ── 6. Squat thresholds (all angles in degrees unless noted) ──────────────────
SQUAT_DOWN_ANGLE         = 105   # Knee angle that starts a rep (→ DOWN stage)
SQUAT_UP_ANGLE           = 155   # Knee angle that completes a rep (→ UP stage)
DEPTH_INSUFFICIENT_ANGLE = 120   # Rep flagged if knee never went below this
BACK_LEAN_MAX            = 55    # Max torso lean from vertical
KNEE_CAVE_RATIO          = 0.72  # Front view: knee_width / ankle_width floor
KNEE_TOE_X_RATIO         = 0.06  # Side view: knee-past-toes as fraction of frame width
HEEL_LIFT_NORM_THRESH    = 0.03  # Normalised Y heel rise before alert fires (~2 cm)
BUTT_WINK_HIP_ANGLE_MIN  = 58    # shoulder-hip-knee angle below this = butt wink
BUTT_WINK_KNEE_ANGLE_MAX = 115   # Only check butt wink when knee is this bent
NECK_ANGLE_MIN           = 140   # ear-shoulder-hip angle below this = neck not neutral



# ─────────────────────────────────────────────────────────────────────────────
# All squat-specific posture checks
# Signature: fn(lms, pts, w, h, stage) -> str | None
# Return an alert string if the check fires, otherwise None.
# ─────────────────────────────────────────────────────────────────────────────

# Heel lift
# Baseline Y captured the moment stage transitions UP→DOWN.
# MediaPipe Y: 0=top, 1=bottom → rising heel = positive delta.
_heel_baseline: dict | None = None
_prev_stage: str = "UP"

def check_heel_lift(lms, pts, w, h, stage):
    global _heel_baseline, _prev_stage
    if stage == "DOWN" and _prev_stage == "UP":
        _heel_baseline = {
            "left":  lms[LM["left_heel"]].y,
            "right": lms[LM["right_heel"]].y,
        }
    _prev_stage = stage
    if stage == "UP":
        _heel_baseline = None
        return None
    if _heel_baseline is None:
        return None
    left_rise  = _heel_baseline["left"]  - lms[LM["left_heel"]].y
    right_rise = _heel_baseline["right"] - lms[LM["right_heel"]].y
    if max(left_rise, right_rise) > HEEL_LIFT_NORM_THRESH:
        return "Heels lifting — press heels down"
    return None


# Knee cave — front view only
# Fires when knee width < KNEE_CAVE_RATIO of ankle width.
def check_knee_cave(lms, pts, w, h, stage):
    if detect_view(lms) != "FRONT":
        return None
    need = {"left_knee", "right_knee", "left_ankle", "right_ankle"}
    if not need.issubset(pts):
        return None
    knee_w  = abs(pts["left_knee"][0]  - pts["right_knee"][0])
    ankle_w = abs(pts["left_ankle"][0] - pts["right_ankle"][0])
    if ankle_w > 10 and knee_w < ankle_w * KNEE_CAVE_RATIO:
        return "Knees caving in — push knees out"
    return None


# Torso lean — front view
# Midpoint of shoulders vs midpoint of hips; degrees from vertical.
def check_torso_lean_front(lms, pts, w, h, stage):
    if detect_view(lms) != "FRONT":
        return None
    need = {"left_shoulder", "right_shoulder", "left_hip", "right_hip"}
    if not need.issubset(pts):
        return None
    mid_sh  = (
        (pts["left_shoulder"][0] + pts["right_shoulder"][0]) // 2,
        (pts["left_shoulder"][1] + pts["right_shoulder"][1]) // 2,
    )
    mid_hip = (
        (pts["left_hip"][0] + pts["right_hip"][0]) // 2,
        (pts["left_hip"][1] + pts["right_hip"][1]) // 2,
    )
    if torso_lean(mid_sh, mid_hip) > BACK_LEAN_MAX:
        return "Leaning too far forward — chest up"
    return None


# Torso lean — side view
# Camera-facing shoulder and hip pixel positions.
def check_torso_lean_side(lms, pts, w, h, stage):
    if detect_view(lms) != "SIDE":
        return None
    s = "left" if lms[LM["left_shoulder"]].z < lms[LM["right_shoulder"]].z else "right"
    need = {f"{s}_shoulder", f"{s}_hip"}
    if not need.issubset(pts):
        return None
    if torso_lean(pts[f"{s}_shoulder"], pts[f"{s}_hip"]) > BACK_LEAN_MAX:
        return "Leaning too far forward — chest up"
    return None


# Knee past toes — side view
# X axis = depth axis in side view.
# Significant X delta of knee vs ankle = knee shooting forward.
def check_knee_past_toes(lms, pts, w, h, stage):
    if detect_view(lms) != "SIDE":
        return None
    s = "left" if lms[LM["left_shoulder"]].z < lms[LM["right_shoulder"]].z else "right"
    need = {f"{s}_knee", f"{s}_ankle"}
    if not need.issubset(pts):
        return None
    knee_toe_x = (pts[f"{s}_knee"][0] - pts[f"{s}_ankle"][0]) / w
    if abs(knee_toe_x) > KNEE_TOE_X_RATIO:
        return "Knee past toes — sit back more"
    return None


# Butt wink — side view, deep squat only
# shoulder→hip→knee angle collapses when pelvis posteriorly tilts.
# Guard: only fires when knee is already deep (< BUTT_WINK_KNEE_ANGLE_MAX).
def check_butt_wink(lms, pts, w, h, stage):
    if detect_view(lms) != "SIDE":
        return None
    s = "left" if lms[LM["left_shoulder"]].z < lms[LM["right_shoulder"]].z else "right"
    try:
        knee_angle = angle_3d(
            lms[LM[f"{s}_hip"]], lms[LM[f"{s}_knee"]], lms[LM[f"{s}_ankle"]]
        )
        if knee_angle > BUTT_WINK_KNEE_ANGLE_MAX:
            return None
        hip_angle = angle_3d(
            lms[LM[f"{s}_shoulder"]], lms[LM[f"{s}_hip"]], lms[LM[f"{s}_knee"]]
        )
    except Exception:
        return None
    if hip_angle < BUTT_WINK_HIP_ANGLE_MIN:
        return "Lower back rounding — brace your core"
    return None


# Neck alignment — both views
# ear→shoulder→hip angle; neutral ~160–180°. Craning closes it.
def check_neck(lms, pts, w, h, stage):
    try:
        view = detect_view(lms)
        if view == "SIDE":
            s  = "left" if lms[LM["left_shoulder"]].z < lms[LM["right_shoulder"]].z else "right"
            na = angle_3d(lms[LM[f"{s}_ear"]], lms[LM[f"{s}_shoulder"]], lms[LM[f"{s}_hip"]])
        else:
            al = angle_3d(lms[LM["left_ear"]],  lms[LM["left_shoulder"]],  lms[LM["left_hip"]])
            ar = angle_3d(lms[LM["right_ear"]], lms[LM["right_shoulder"]], lms[LM["right_hip"]])
            na = (al + ar) / 2
    except Exception:
        return None
    if na < NECK_ANGLE_MIN:
        return "Neck not neutral — look straight ahead"
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Exercise definition
# ─────────────────────────────────────────────────────────────────────────────

def get_definition() -> ExerciseDefinition:
    return ExerciseDefinition(
        name        = "Squat",
        description = "Bodyweight or loaded squat: bilateral lower body compound movement.",
        valid_views = ["FRONT", "SIDE"],

        joint_checks = [
            # Left knee — 2D pixel angle, FRONT view, rep driver for FRONT
            JointCheck(
                display_name     = "Knee (L)",
                landmark_a       = "left_hip",
                landmark_b       = "left_knee",
                landmark_c       = "left_ankle",
                min_angle        = 0.0,
                optimal_angle    = 90.0,
                max_angle        = 180.0,
                check_in_views   = ["FRONT"],
                use_3d           = False,
                is_rep_driver    = True,
                driver_for_views = ["FRONT"],
            ),
            # Right knee — 2D, FRONT view, display only
            JointCheck(
                display_name     = "Knee (R)",
                landmark_a       = "right_hip",
                landmark_b       = "right_knee",
                landmark_c       = "right_ankle",
                min_angle        = 0.0,
                optimal_angle    = 90.0,
                max_angle        = 180.0,
                check_in_views   = ["FRONT"],
                use_3d           = False,
                is_rep_driver    = False,
            ),
            # Camera-facing knee — 3D, SIDE view, rep driver for SIDE
            JointCheck(
                display_name     = "Knee (side)",
                landmark_a       = "left_hip",
                landmark_b       = "left_knee",
                landmark_c       = "left_ankle",
                min_angle        = 0.0,
                optimal_angle    = 90.0,
                max_angle        = 180.0,
                check_in_views   = ["SIDE"],
                use_3d           = True,
                is_rep_driver    = True,
                driver_for_views = ["SIDE"],
            ),
        ],

        rep_trigger = RepTrigger(
            joint_display_name  = "Knee (L)",
            enter_angle         = 105.0,   # SQUAT_DOWN_ANGLE
            exit_angle          = 155.0,   # SQUAT_UP_ANGLE
            depth_target        = 120.0,   # DEPTH_INSUFFICIENT_ANGLE
            depth_alert         = "Not deep enough",
            direction           = "decrease",
        ),

        extra_checks = [
            check_knee_cave,
            check_torso_lean_front,
            check_torso_lean_side,
            check_knee_past_toes,
            check_heel_lift,
            check_butt_wink,
            check_neck,
        ],

        gemini_errors = [
            '"Knees caving in" — valgus collapse, knee width drops below 72% of ankle width (front view)',
            '"Leaning too far forward" — torso tilts more than 55 degrees from vertical',
            '"Knee past toes" — knee X drifts more than 6% of frame width ahead of ankle (side view)',
            '"Not deep enough" — knee angle never dropped below 120 degrees during the rep',
            '"Heels lifting" — heel Y rises more than 0.03 normalised units above rep-start baseline',
            '"Lower back rounding" — shoulder-hip-knee angle drops below 58 degrees at the bottom (side view)',
            '"Neck not neutral" — ear-shoulder-hip angle drops below 140 degrees',
        ],
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_exercise(get_definition())
