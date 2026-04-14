# ╔══════════════════════════════════════════════════════════════════╗
# ║  exercise_base.py — Core data structures for the exercise system ║
# ║                                                                  ║
# ║  Every exercise file creates an ExerciseDefinition object and   ║
# ║  returns it from a get_definition() function.                   ║
# ║  The engine (exercise_runner.py) accepts any ExerciseDefinition ║
# ║  and runs the full analysis loop generically.                   ║
# ╚══════════════════════════════════════════════════════════════════╝

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class JointCheck:
    """
    Defines an angle-based check for a single joint.

    The angle is measured at landmark_b (the vertex), formed by the
    triangle: landmark_a → landmark_b → landmark_c.

    Thresholds:
      min_angle    — below this fires alert_too_low  (joint too compressed/flexed)
      optimal_angle— displayed as the target marker on the depth bar
      max_angle    — above this fires alert_too_high (joint too extended/open)

    Set min_angle = 0 or max_angle = 180 to disable that side of the check.

    Fields:
      display_name   — shown on HUD next to the live angle value
      landmark_a/b/c — landmark name strings (keys in landmarks.LM)
      min_angle      — minimum acceptable angle (degrees)
      optimal_angle  — target angle shown as a bar marker (degrees)
      max_angle      — maximum acceptable angle (degrees)
      alert_too_low  — message when angle < min_angle
      alert_too_high — message when angle > max_angle
      check_in_views — which views this check applies to: ["FRONT"], ["SIDE"], or ["FRONT","SIDE"]
      use_3d         — True = angle_3d (reliable in SIDE view), False = angle_2d (front view)
      is_rep_driver  — True = this joint's angle drives the rep state machine
    """
    display_name:   str
    landmark_a:     str
    landmark_b:     str          # vertex
    landmark_c:     str
    min_angle:      float = 0.0
    optimal_angle:  float = 90.0
    max_angle:      float = 180.0
    alert_too_low:  str   = ""
    alert_too_high: str   = ""
    check_in_views: list  = field(default_factory=lambda: ["FRONT", "SIDE"])
    use_3d:         bool  = False
    is_rep_driver:  bool  = False
    driver_for_views: list = field(default_factory=lambda: ["FRONT", "SIDE"])
    # If is_rep_driver=True, this joint only drives the rep machine when the
    # current view is in driver_for_views. Allows different drivers per view.


@dataclass
class RepTrigger:
    """
    Defines what drives the rep state machine.

    The engine watches the angle of the driver joint:
      - When angle drops below  enter_angle → stage = "DOWN" (rep begins)
      - When angle rises above  exit_angle  → stage = "UP"   (rep completes)
      - depth_target: the minimum angle the joint must reach while in DOWN
        for the rep to count (set to 0 to skip depth check)

    direction:
      "decrease" — rep starts when angle goes DOWN (e.g. squats, lunges, curls)
      "increase" — rep starts when angle goes UP   (e.g. leg raises, presses)
    """
    joint_display_name: str     # must match a JointCheck.display_name marked is_rep_driver=True
    enter_angle:        float   # crossing this starts a rep
    exit_angle:         float   # crossing this completes a rep
    depth_target:       float = 0.0   # minimum angle required during DOWN (0 = no check)
    depth_alert:        str   = "Not deep enough"
    direction:          str   = "decrease"   # "decrease" | "increase"


@dataclass
class IsometricTrigger:
    """
    Defines the hold logic for isometric / static exercises.

    Instead of counting reps, the engine:
      - Detects when the joint angle enters the hold zone
        (between hold_min_angle and hold_max_angle)
      - Starts a hold timer
      - Monitors the angle stays within the zone
      - Fires alert_break if they drop out of position
      - Counts total accumulated seconds held (displayed instead of reps)
      - Optionally targets a hold_duration_seconds goal

    Fields:
      joint_display_name  — must match a JointCheck display_name
      hold_min_angle      — angle must be >= this to count as "in position"
      hold_max_angle      — angle must be <= this to count as "in position"
      hold_duration_secs  — target hold duration in seconds (shown on progress bar)
      alert_not_in_pos    — shown when angle is outside the hold zone
      alert_break         — shown when they exit the hold zone after holding
    """
    joint_display_name: str
    hold_min_angle:     float         # lower bound of the hold zone (degrees)
    hold_max_angle:     float         # upper bound of the hold zone (degrees)
    hold_duration_secs: float = 30.0  # target hold time in seconds
    alert_not_in_pos:   str   = "Get into position — bend your knee to the target angle"
    alert_break:        str   = "Hold position — do not straighten your leg"


@dataclass
class ExerciseDefinition:
    """
    Complete specification of an exercise.
    Pass this to exercise_runner.run_exercise() to start analysis.

    Fields:
      name            — displayed in HUD title and Gemini report
      description     — sentence describing the exercise for Gemini context
      joint_checks    — list of JointCheck objects (angle-based rules)
      rep_trigger     — RepTrigger defining the rep state machine
      valid_views     — views this exercise supports: ["FRONT"], ["SIDE"], ["FRONT","SIDE"]
      extra_checks    — optional list of callables for non-angle checks
                        signature: fn(lms, pts, w, h, stage) -> str | None
                        return an alert string or None
      gemini_errors   — list of error descriptions sent to Gemini for context
                        (auto-built from joint_checks if left empty)
    """
    name:               str
    description:        str
    joint_checks:       list[JointCheck]
    rep_trigger:        RepTrigger | None   # None for isometric exercises
    valid_views:        list[str]           = field(default_factory=lambda: ["FRONT", "SIDE"])
    extra_checks:       list[Callable]      = field(default_factory=list)
    gemini_errors:      list[str]           = field(default_factory=list)
    isometric_trigger:  object              = None   # IsometricTrigger | None

    def __post_init__(self):
        # Auto-populate gemini_errors from joint_checks if not provided
        if not self.gemini_errors:
            for jc in self.joint_checks:
                if jc.alert_too_low:
                    self.gemini_errors.append(
                        f'"{jc.alert_too_low}" — {jc.display_name} angle below {jc.min_angle}°'
                    )
                if jc.alert_too_high:
                    self.gemini_errors.append(
                        f'"{jc.alert_too_high}" — {jc.display_name} angle above {jc.max_angle}°'
                    )
            # Only add depth alert for dynamic exercises (rep_trigger is not None)
            if (self.rep_trigger and
                    self.rep_trigger.depth_alert and
                    self.rep_trigger.depth_target > 0):
                self.gemini_errors.append(
                    f'"{self.rep_trigger.depth_alert}" — '
                    f'joint never reached {self.rep_trigger.depth_target}° during the rep'
                )
