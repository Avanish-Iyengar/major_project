# ╔══════════════════════════════════════════════════════════════════╗
# ║  exercise_runner.py — Generic analysis engine                   ║
# ║  Runs any ExerciseDefinition: camera → MediaPipe → checks →    ║
# ║  rep counting (dynamic) OR hold timing (isometric) →           ║
# ║  voice alerts → HUD overlay → Gemini post-session report        ║
# ╚══════════════════════════════════════════════════════════════════╝

import cv2
import time
import sys
import os
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("❌ Run: pip install mediapipe opencv-python numpy requests")
    sys.exit(1)

from config import (
    MODEL_PATH, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT, DETECTION_SCALE,
    DEPTH_INSUFFICIENT_ANGLE,
)
from landmarks import LM
from smoother import reset_smoother
from math_utils import build_pts, detect_view, angle_2d, angle_3d
from voice import init_tts, speak
from drawing import draw_skeleton, C
from gemini import build_gemini_prompt_generic, show_report_window
from exercise_base import ExerciseDefinition


# ── Angle helpers ──────────────────────────────────────────────────────────────

def _compute_joint_angle(jc, lms, pts):
    if jc.use_3d:
        try:
            return angle_3d(lms[LM[jc.landmark_a]], lms[LM[jc.landmark_b]], lms[LM[jc.landmark_c]])
        except Exception:
            return None
    else:
        if jc.landmark_a not in pts or jc.landmark_b not in pts or jc.landmark_c not in pts:
            return None
        return angle_2d(pts[jc.landmark_a], pts[jc.landmark_b], pts[jc.landmark_c])


def _run_joint_checks(definition, lms, pts, w, view):
    """Run all JointChecks for current view. Returns (alerts, joint_angles, driver_angle)."""
    alerts, joint_angles, driver_angle = [], {}, 0.0
    for jc in definition.joint_checks:
        if view not in jc.check_in_views:
            continue
        angle = _compute_joint_angle(jc, lms, pts)
        if angle is None:
            continue
        joint_angles[jc.display_name] = angle
        if jc.is_rep_driver and view in jc.driver_for_views:
            driver_angle = angle
        if jc.alert_too_low and angle < jc.min_angle:
            alerts.append(jc.alert_too_low)
        elif jc.alert_too_high and angle > jc.max_angle:
            alerts.append(jc.alert_too_high)
    return alerts, joint_angles, driver_angle


# ── HUD drawing ────────────────────────────────────────────────────────────────

def _draw_hud(frame, definition, state: dict, fps: float):
    """Draw the semi-transparent left HUD panel onto frame in-place."""
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (300, h), C["panel"], -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    is_iso  = definition.isometric_trigger is not None
    title   = definition.name.upper()[:20]

    # ── Title ──────────────────────────────────────────────────────────
    cv2.putText(frame, title, (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, C["accent"], 1, cv2.LINE_AA)
    view_str = state.get("view", "FRONT")
    cv2.putText(frame, f"[{view_str} VIEW]", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    cv2.line(frame, (12, 58), (288, 58), C["accent"], 1)

    # ── Counters ───────────────────────────────────────────────────────
    if is_iso:
        # Left: hold timer. Right: sets completed.
        hold_secs  = state.get("hold_secs", 0.0)
        sets_done  = state.get("rep_count", 0)
        in_pos     = state.get("in_position", False)
        timer_col  = C["green"] if in_pos else C["yellow"]
        cv2.putText(frame, "HOLD TIME", (12, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, timer_col, 1)
        cv2.putText(frame, f"{hold_secs:.1f}s", (12, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 2.4, timer_col, 2, cv2.LINE_AA)
        cv2.putText(frame, "SETS", (170, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["green"], 1)
        cv2.putText(frame, str(sets_done), (170, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, C["green"], 2, cv2.LINE_AA)
    else:
        # Left: total reps. Right: clean reps.
        cv2.putText(frame, "TOTAL", (12, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["white"], 1)
        cv2.putText(frame, str(state.get("rep_count", 0)), (12, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, C["white"], 2, cv2.LINE_AA)
        cv2.putText(frame, "CLEAN", (160, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["green"], 1)
        cv2.putText(frame, str(state.get("clean_reps", 0)), (160, 135),
                    cv2.FONT_HERSHEY_DUPLEX, 3.0, C["green"], 2, cv2.LINE_AA)

    # ── Stage / position badge ─────────────────────────────────────────
    if is_iso:
        in_pos  = state.get("in_position", False)
        b_col   = C["green"] if in_pos else C["yellow"]
        b_text  = "  IN POSITION  " if in_pos else "  GET IN POSITION  "
    else:
        stage  = state.get("stage", "UP")
        b_col  = C["green"] if stage == "UP" else C["yellow"]
        b_text = f"  STAGE: {stage}  "
    cv2.rectangle(frame, (12, 145), (288, 165), b_col, -1)
    cv2.putText(frame, b_text, (14, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["panel"], 1)

    # ── Last rep / set result (dynamic only) ───────────────────────────
    if not is_iso:
        rep_was_clean   = state.get("rep_was_clean")
        rep_fail_reason = state.get("rep_fail_reason", "")
        if rep_was_clean is True:
            cv2.putText(frame, "CLEAN REP", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["green"], 1, cv2.LINE_AA)
        elif rep_was_clean is False:
            cv2.putText(frame, f"BAD: {rep_fail_reason[:22]}", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["red"], 1, cv2.LINE_AA)

    # ── Live joint metrics ─────────────────────────────────────────────
    cv2.line(frame, (12, 192), (288, 192), (40, 40, 60), 1)
    joint_angles = state.get("joint_angles", {})
    y = 210
    for name, angle in list(joint_angles.items())[:6]:
        cv2.putText(frame, f"{name}: {int(angle)}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 210), 1)
        y += 19
    cv2.putText(frame, f"FPS: {fps:.0f}", (12, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 210), 1)

    # ── Progress bar ───────────────────────────────────────────────────
    bar_y = y + 22
    cv2.putText(frame, "DEPTH" if not is_iso else "HOLD PROGRESS",
                (12, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    bar_w, bar_h = 200, 10

    if is_iso:
        iso          = definition.isometric_trigger
        hold_secs    = state.get("hold_secs", 0.0)
        ratio        = min(1.0, hold_secs / max(iso.hold_duration_secs, 1))
        target_ratio = 1.0   # goal is full bar
    else:
        rt           = definition.rep_trigger
        driver_angle = state.get("driver_angle", 155.0)
        angle_high   = max(rt.enter_angle, rt.exit_angle) + 20
        angle_low    = max(0.0, min(rt.enter_angle, rt.exit_angle) - 20)
        span         = max(1.0, angle_high - angle_low)
        if rt.direction == "decrease":
            ratio = max(0.0, min(1.0, (angle_high - driver_angle) / span))
        else:
            ratio = max(0.0, min(1.0, (driver_angle - angle_low) / span))
        target_ratio = (angle_high - DEPTH_INSUFFICIENT_ANGLE) / span

    cv2.rectangle(frame, (12, bar_y), (12 + bar_w, bar_y + bar_h), (40, 40, 60), -1)
    bar_color = C["green"] if ratio >= target_ratio * 0.85 else C["yellow"]
    cv2.rectangle(frame, (12, bar_y), (12 + int(bar_w * ratio), bar_y + bar_h), bar_color, -1)
    tx = 12 + int(bar_w * min(1.0, target_ratio))
    cv2.line(frame, (tx, bar_y - 2), (tx, bar_y + bar_h + 2), C["white"], 1)

    # ── Alerts ─────────────────────────────────────────────────────────
    alert_y = bar_y + bar_h + 20
    cv2.line(frame, (12, alert_y - 6), (288, alert_y - 6), (40, 40, 60), 1)
    alerts = state.get("alerts", [])
    if alerts:
        cv2.putText(frame, "! ALERTS", (12, alert_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["red"], 1)
        ay = alert_y + 32
        for alert in alerts[:4]:
            words, line = alert.split(), ""
            for word in words:
                if len(line + word) < 28:
                    line += ("" if not line else " ") + word
                else:
                    cv2.putText(frame, line.strip(), (14, ay),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["red"], 1)
                    ay += 16; line = word
            cv2.putText(frame, line.strip(), (14, ay),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, C["red"], 1)
            ay += 20
    else:
        cv2.putText(frame, "POSTURE OK", (12, alert_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["green"], 1)

    hint = "Q = quit   R = reset" if not is_iso else "Q = quit   R = reset hold"
    cv2.putText(frame, hint, (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C["dim"], 1)


# ── Main engine ────────────────────────────────────────────────────────────────

def run_exercise(definition: ExerciseDefinition):
    """
    Entry point called by every exercise file.
    Handles both dynamic (rep-based) and isometric (hold-based) exercises.
    """
    # ── Model check ───────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"""
❌ Model file not found: {MODEL_PATH}
Download LITE (recommended):
  curl -o pose_landmarker_lite.task \\
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
Then update MODEL_PATH in config.py.
""")
        sys.exit(1)

    init_tts()
    is_iso = definition.isometric_trigger is not None
    rt     = definition.rep_trigger          # None for isometric
    iso    = definition.isometric_trigger    # None for dynamic

    # ── MediaPipe ─────────────────────────────────────────────────────────
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_segmentation_masks=False,
    )
    landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    # ── Camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print(f"❌ Could not open camera at index {CAMERA_INDEX}.")
        print("   Try changing CAMERA_INDEX in config.py (0, 1, 2...)")
        landmarker.close()
        sys.exit(1)

    for _ in range(5):   # warmup frames
        cap.read()

    # ── Session state — dynamic ───────────────────────────────────────────
    rep_count          = 0
    clean_reps         = 0
    stage              = "UP"
    rep_had_error      = False
    rep_fail_reason    = ""
    rep_was_clean      = None
    best_driver_angle  = 180.0 if (rt and rt.direction == "decrease") else 0.0
    rep_errors         = []
    session_log        = []

    # ── Session state — isometric ─────────────────────────────────────────
    hold_start:  float | None = None   # time.time() when hold began
    hold_secs:   float        = 0.0    # seconds held in current hold
    in_position: bool         = False  # currently inside hold zone
    prev_in_pos: bool         = False  # previous frame's in_position (for edge detect)

    # ── Shared state ──────────────────────────────────────────────────────
    alerts       = []
    joint_angles = {}
    driver_angle = 0.0
    fps          = 0.0
    prev_time    = time.time()
    view         = definition.valid_views[0]

    type_label = "isometric" if is_iso else "dynamic"
    print(f"\n🏋  {definition.name} ({type_label}) — starting.")
    print(f"   {definition.description}")
    print(f"   Q = quit   R = reset\n")

    WIN_TITLE = f"XAI - {definition.name}"
    consecutive_failures = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print("❌ Camera stopped sending frames.")
                break
            continue
        consecutive_failures = 0

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # MediaPipe on downscaled frame
        det_w  = int(w * DETECTION_SCALE)
        det_h  = int(h * DETECTION_SCALE)
        small  = cv2.resize(frame, (det_w, det_h))
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        if result.pose_landmarks:
            lms  = result.pose_landmarks[0]
            pts  = build_pts(lms, w, h)
            view = detect_view(lms)

            if view in definition.valid_views:
                alerts, joint_angles, driver_angle = _run_joint_checks(
                    definition, lms, pts, w, view
                )
                for fn in definition.extra_checks:
                    a = fn(lms, pts, w, h, stage)
                    if a:
                        alerts.append(a)
            else:
                alerts = []

            # ── ISOMETRIC state machine ────────────────────────────────────
            if is_iso:
                # Find the monitored joint angle for the hold zone check
                iso_angle = 0.0
                for jc in definition.joint_checks:
                    if jc.display_name == iso.joint_display_name:
                        iso_angle = joint_angles.get(jc.display_name, 0.0)
                        break

                in_position = (
                    iso_angle >= iso.hold_min_angle and
                    iso_angle <= iso.hold_max_angle and
                    iso_angle > 0
                )

                if in_position:
                    if not prev_in_pos:
                        # Just entered the hold zone — start/restart timer
                        hold_start = time.time()
                        hold_secs  = 0.0
                    else:
                        hold_secs = time.time() - hold_start

                    if alerts:
                        speak(alerts[0])

                    # Goal reached — count as a completed set
                    if hold_secs >= iso.hold_duration_secs:
                        rep_count   += 1
                        hold_secs    = 0.0
                        hold_start   = time.time()
                        print(f"  ✅ Set {rep_count} complete! "
                              f"({iso.hold_duration_secs:.0f}s hold achieved)")
                        session_log.append({
                            "rep":        rep_count,
                            "errors":     [],
                            "best_angle": int(iso_angle),
                            "clean":      True,
                        })
                else:
                    # Outside hold zone
                    if prev_in_pos and hold_secs > 0.5:
                        # Was holding, now broke out — alert
                        speak(iso.alert_not_in_pos)
                        print(f"  ✗ Hold broken after {hold_secs:.1f}s")
                    hold_secs = 0.0
                    hold_start = None
                    if not alerts and iso_angle > 0:
                        alerts = [iso.alert_not_in_pos]

                prev_in_pos = in_position

            # ── DYNAMIC state machine ──────────────────────────────────────
            else:
                # UP → DOWN entry
                if stage == "UP":
                    entered = (
                        (rt.direction == "decrease" and driver_angle < rt.enter_angle) or
                        (rt.direction == "increase" and driver_angle > rt.enter_angle)
                    )
                    if entered and driver_angle > 0:
                        stage             = "DOWN"
                        rep_had_error     = False
                        rep_fail_reason   = ""
                        rep_errors        = []
                        best_driver_angle = driver_angle

                # While DOWN
                if stage == "DOWN":
                    if rt.direction == "decrease":
                        best_driver_angle = min(best_driver_angle, driver_angle)
                    else:
                        best_driver_angle = max(best_driver_angle, driver_angle)

                    if alerts:
                        speak(alerts[0])
                        for a in alerts:
                            if a not in rep_errors:
                                rep_errors.append(a)
                        if not rep_had_error:
                            rep_had_error   = True
                            rep_fail_reason = alerts[0]

                # DOWN → UP exit
                if stage == "DOWN":
                    exited = (
                        (rt.direction == "decrease" and driver_angle > rt.exit_angle) or
                        (rt.direction == "increase" and driver_angle < rt.exit_angle)
                    )
                    if exited:
                        stage      = "UP"
                        rep_count += 1

                        if rt.depth_target > 0:
                            depth_ok = (
                                (rt.direction == "decrease" and best_driver_angle <= rt.depth_target) or
                                (rt.direction == "increase" and best_driver_angle >= rt.depth_target)
                            )
                            if not depth_ok and not rep_had_error:
                                rep_had_error   = True
                                rep_fail_reason = rt.depth_alert
                                rep_errors.append(rt.depth_alert)
                                speak(rt.depth_alert)

                        rep_was_clean = not rep_had_error
                        if rep_was_clean:
                            clean_reps += 1
                            print(f"  ✅ Rep {rep_count} — clean! "
                                  f"(best: {int(best_driver_angle)}°, clean: {clean_reps})")
                        else:
                            print(f"  ✗ Rep {rep_count} — {rep_fail_reason} "
                                  f"(best: {int(best_driver_angle)}°)")

                        session_log.append({
                            "rep":        rep_count,
                            "errors":     list(rep_errors),
                            "best_angle": int(best_driver_angle),
                            "clean":      rep_was_clean,
                        })

            # Skeleton overlay
            skel_color = C["red"] if alerts else C["green"]
            draw_skeleton(frame, pts, skel_color)
            for jc in definition.joint_checks:
                if jc.display_name in joint_angles and jc.landmark_b in pts:
                    pt  = pts[jc.landmark_b]
                    val = joint_angles[jc.display_name]
                    cv2.putText(frame, f"{int(val)}", (pt[0] + 10, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, skel_color, 1, cv2.LINE_AA)
        else:
            alerts = []

        # FPS
        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # Build state dict for HUD
        state = {
            "rep_count":       rep_count,
            "clean_reps":      clean_reps,
            "stage":           stage,
            "alerts":          list(alerts),
            "joint_angles":    dict(joint_angles),
            "driver_angle":    driver_angle,
            "rep_was_clean":   rep_was_clean,
            "rep_fail_reason": rep_fail_reason,
            "view":            view,
            "hold_secs":       hold_secs,
            "in_position":     in_position if is_iso else False,
        }

        _draw_hud(frame, definition, state, fps)

        cv2.imshow(WIN_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset dynamic state
            rep_count = clean_reps = 0
            stage = "UP"; rep_had_error = False
            rep_was_clean = None; rep_fail_reason = ""
            best_driver_angle = 180.0 if (rt and rt.direction == "decrease") else 0.0
            rep_errors = []; session_log.clear()
            # Reset isometric state
            hold_start = None; hold_secs = 0.0
            in_position = False; prev_in_pos = False
            reset_smoother()
            print("  Reset.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\n✅ Done.  Reps/Sets: {rep_count}")

    # ── Gemini post-session report ─────────────────────────────────────────
    if session_log:
        print("\n🤖 Opening Gemini coaching report...")
        prompt = build_gemini_prompt_generic(definition, session_log, rep_count, clean_reps)
        show_report_window(prompt)
    else:
        print("   (No reps/sets recorded — skipping Gemini report.)")
