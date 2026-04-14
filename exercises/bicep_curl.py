# ╔══════════════════════════════════════════════════════════════════╗
# ║  exercises/bicep_curl.py — Dumbbell Bicep Curl                  ║
# ║  Run:  python exercises/bicep_curl.py                           ║
# ╚══════════════════════════════════════════════════════════════════╝


from exercise_base import ExerciseDefinition, JointCheck, RepTrigger
from exercise_runner import run_exercise
from landmarks import LM


def check_elbow_flare(lms, pts, w, h, stage):
    """
    Elbows should stay close to the torso during a curl.
    If elbow drifts significantly forward of shoulder (front view X), flag it.
    """
    need = {"left_elbow", "left_shoulder", "right_elbow", "right_shoulder"}
    if not need.issubset(pts):
        return None
    l_drift = abs(pts["left_elbow"][0]  - pts["left_shoulder"][0])  / w
    r_drift = abs(pts["right_elbow"][0] - pts["right_shoulder"][0]) / w
    if max(l_drift, r_drift) > 0.10:
        return "Elbows flaring — keep elbows at your sides"
    return None


def check_wrist_alignment(lms, pts, w, h, stage):
    """
    Wrist should stay roughly in line with forearm (no excessive bend).
    Checks that wrist X is within a reasonable range of elbow X (front view).
    """
    need = {"left_wrist", "left_elbow"}
    if not need.issubset(pts):
        return None
    wrist_drift = abs(pts["left_wrist"][0] - pts["left_elbow"][0]) / w
    if wrist_drift > 0.08:
        return "Wrist bending — keep wrists straight"
    return None


def get_definition() -> ExerciseDefinition:
    return ExerciseDefinition(
        name        = "Bicep Curl",
        description = "Dumbbell or barbell bicep curl: elbow flexion exercise for biceps.",
        valid_views = ["FRONT", "SIDE"],

        joint_checks = [
            # Left elbow — rep driver
            JointCheck(
                display_name   = "Elbow (L)",
                landmark_a     = "left_shoulder",
                landmark_b     = "left_elbow",
                landmark_c     = "left_wrist",
                min_angle      = 30.0,
                optimal_angle  = 45.0,
                max_angle      = 170.0,
                alert_too_low  = "Elbow over-flexed — control the range",
                alert_too_high = "Arm not fully extended at bottom",
                check_in_views = ["FRONT", "SIDE"],
                use_3d         = False,
                is_rep_driver  = True,
            ),
            # Right elbow
            JointCheck(
                display_name   = "Elbow (R)",
                landmark_a     = "right_shoulder",
                landmark_b     = "right_elbow",
                landmark_c     = "right_wrist",
                min_angle      = 30.0,
                optimal_angle  = 45.0,
                max_angle      = 170.0,
                alert_too_low  = "Elbow over-flexed — control the range",
                alert_too_high = "Arm not fully extended at bottom",
                check_in_views = ["FRONT", "SIDE"],
                use_3d         = False,
                is_rep_driver  = False,
            ),
            # Shoulder stability: shoulder should not rise during the curl
            JointCheck(
                display_name   = "Shoulder (L)",
                landmark_a     = "left_elbow",
                landmark_b     = "left_shoulder",
                landmark_c     = "left_hip",
                min_angle      = 0.0,
                optimal_angle  = 10.0,
                max_angle      = 35.0,
                alert_too_high = "Shoulder rising — keep shoulder packed down",
                check_in_views = ["SIDE"],
                use_3d         = True,
                is_rep_driver  = False,
            ),
        ],

        rep_trigger = RepTrigger(
            joint_display_name = "Elbow (L)",
            enter_angle        = 130.0,   # arm starts straight, begins flexing
            exit_angle         = 150.0,   # arm returns to extended
            depth_target       = 60.0,    # must curl to at least 60°
            depth_alert        = "Curl not high enough — full range of motion",
            direction          = "decrease",
        ),

        extra_checks = [
            check_elbow_flare,
            check_wrist_alignment,
        ],

        gemini_errors = [
            '"Elbows flaring" — elbows drift away from torso, reducing bicep isolation',
            '"Wrist bending" — wrists break under load, risking wrist strain',
            '"Shoulder rising" — shoulder shrugs during the curl, using momentum',
            '"Arm not fully extended at bottom" — incomplete range of motion',
            '"Curl not high enough" — not reaching full elbow flexion at top',
        ],
    )


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    run_exercise(get_definition())
