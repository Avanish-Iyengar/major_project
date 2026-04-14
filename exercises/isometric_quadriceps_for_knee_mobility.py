# exercises/isometric_quadriceps_for_knee_mobility.py — Isometric quadriceps for knee mobility
# Run:  python exercises/isometric_quadriceps_for_knee_mobility.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exercise_base import ExerciseDefinition, JointCheck, IsometricTrigger
from exercise_runner import run_exercise

HOLD_MIN    = 130.0
HOLD_MAX    = 170.0
HOLD_TARGET = 10.0

def get_definition() -> ExerciseDefinition:
    return ExerciseDefinition(
        name        = 'Isometric quadriceps for knee mobility',
        description = 'Perform an isometric contraction of the quadriceps to improve knee mobility.',
        valid_views = ['SIDE'],

        joint_checks = [
            JointCheck(
                display_name    = 'Knee (L)',
                landmark_a      = 'left_hip',
                landmark_b      = 'left_knee',
                landmark_c      = 'left_ankle',
                min_angle       = HOLD_MIN,
                optimal_angle   = (HOLD_MIN + HOLD_MAX) / 2,
                max_angle       = HOLD_MAX,
                alert_too_low   = 'Too flexed',
                alert_too_high  = 'Not bent enough',
                check_in_views  = ['SIDE'],
                use_3d          = False,
                is_rep_driver   = False,
            ),
        ],

        rep_trigger = None,

        isometric_trigger = IsometricTrigger(
            joint_display_name = 'Knee (L)',
            hold_min_angle     = HOLD_MIN,
            hold_max_angle     = HOLD_MAX,
            hold_duration_secs = HOLD_TARGET,
            alert_not_in_pos   = 'Hold between 130 and 170 degrees',
            alert_break        = "Position lost — return to hold position",
        ),

        extra_checks  = [],   # add custom checks here if needed

        gemini_errors = [

        ],
    )


if __name__ == "__main__":
    run_exercise(get_definition())
