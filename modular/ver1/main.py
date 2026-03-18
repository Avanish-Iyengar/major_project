"""
╔══════════════════════════════════════════════════════════════════╗
║  XAI Physiotherapy — Squat Analyzer                             ║
║  Run:  python main.py                                           ║
║  Edit: config.py  to change keys, thresholds, and camera       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import cv2
import time
import sys
import os

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("❌ Run: pip install mediapipe opencv-python numpy pygame requests")
    sys.exit(1)

from config import (
    MODEL_PATH, CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT,
    DETECTION_SCALE, SQUAT_DOWN_ANGLE, SQUAT_UP_ANGLE, DEPTH_INSUFFICIENT_ANGLE,
)
from landmarks import LM
from smoother import reset_smoother
from math_utils import build_pts, detect_view
from posture_checks import (
    check_posture_front, check_posture_side,
    check_heel_lift, check_neck, check_butt_wink,
)
from voice import init_tts, speak
from drawing import draw_skeleton, draw_hud, C
from gemini import build_gemini_prompt, show_report_window


def main():
    # ── Model check ───────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"""
❌ Model file not found: {MODEL_PATH}

Download LITE (recommended):
  curl -o pose_landmarker_lite.task \\
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"

Download FULL (more accurate, slower):
  curl -o pose_landmarker.task \\
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

Then update MODEL_PATH in config.py.
""")
        sys.exit(1)

    init_tts()

    # ── MediaPipe setup ───────────────────────────────────────────────────
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

    # ── Camera setup ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    # ── Session state ─────────────────────────────────────────────────────
    rep_count           = 0
    clean_reps          = 0
    stage               = "UP"
    rep_had_error       = False
    rep_fail_reason     = ""
    rep_was_clean       = None
    min_angle_this_rep  = 180.0
    alerts              = []
    lean = kl = kr      = 0.0
    neck_angle          = 180.0
    heel_rise           = 0.0
    fps                 = 0.0
    prev_time           = time.time()
    view                = "FRONT"
    heel_baseline: dict | None  = None
    session_log: list[dict]     = []
    rep_errors_this_rep: list[str] = []

    print("\n🏋  Squat Analyzer running.")
    print("   Errors checked: knee cave · torso lean · knee past toes ·")
    print("                   heel lift · butt wink · neck · depth")
    print("   Q = quit   R = reset\n")

    # ── Main loop ─────────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Downscale → detect → project onto full-res frame
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

            # View-dependent posture checks
            if view == "SIDE":
                alerts, lean, kl, kr = check_posture_side(lms, pts, w)
            else:
                alerts, lean, kl, kr = check_posture_front(lms, pts, w)

            avg = (kl + kr) / 2

            # Shared checks
            neck_alert, neck_angle = check_neck(lms, view)
            if neck_alert:
                alerts.append(neck_alert)

            if view == "SIDE":
                bw_alert, _ = check_butt_wink(lms, avg)
                if bw_alert:
                    alerts.append(bw_alert)

            # ── Rep state machine ──────────────────────────────────────────
            if avg < SQUAT_DOWN_ANGLE and stage == "UP":
                stage               = "DOWN"
                rep_had_error       = False
                rep_fail_reason     = ""
                min_angle_this_rep  = avg
                rep_errors_this_rep = []
                heel_baseline       = {
                    "left":  lms[LM["left_heel"]].y,
                    "right": lms[LM["right_heel"]].y,
                }

            heel_alert, heel_rise = (
                check_heel_lift(lms, heel_baseline) if stage == "DOWN" else (None, 0.0)
            )
            if heel_alert:
                alerts.append(heel_alert)

            if stage == "DOWN":
                min_angle_this_rep = min(min_angle_this_rep, avg)
                if alerts:
                    speak(alerts[0])
                    for a in alerts:
                        if a not in rep_errors_this_rep:
                            rep_errors_this_rep.append(a)
                    if not rep_had_error:
                        rep_had_error   = True
                        rep_fail_reason = alerts[0]

            if avg > SQUAT_UP_ANGLE and stage == "DOWN":
                stage         = "UP"
                rep_count    += 1
                heel_baseline = None

                depth_ok = min_angle_this_rep <= DEPTH_INSUFFICIENT_ANGLE
                if not depth_ok and not rep_had_error:
                    rep_had_error   = True
                    rep_fail_reason = "Not deep enough"
                    rep_errors_this_rep.append("Not deep enough")
                    speak("Not deep enough")

                if rep_had_error:
                    rep_was_clean = False
                    print(f"  ✗ Rep {rep_count} — {rep_fail_reason} "
                          f"(depth: {int(min_angle_this_rep)}°)")
                else:
                    clean_reps   += 1
                    rep_was_clean = True
                    print(f"  ✅ Rep {rep_count} — clean! "
                          f"(depth: {int(min_angle_this_rep)}°, clean: {clean_reps})")

                session_log.append({
                    "rep":         rep_count,
                    "errors":      list(rep_errors_this_rep),
                    "depth_angle": int(min_angle_this_rep),
                    "clean":       not rep_had_error,
                })

            # Draw
            skel_color = C["red"] if alerts else C["green"]
            draw_skeleton(frame, pts, skel_color)
            for name, angle_val in [("left_knee", kl), ("right_knee", kr)]:
                if name in pts:
                    pt = pts[name]
                    cv2.putText(frame, f"{int(angle_val)}", (pt[0] + 10, pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, skel_color, 1, cv2.LINE_AA)
        else:
            alerts = []

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        draw_hud(frame, rep_count, clean_reps, stage, alerts, fps, lean, kl, kr,
                 view, rep_was_clean, rep_fail_reason, min_angle_this_rep,
                 neck_angle, heel_rise)
        cv2.imshow("XAI Squat Analyzer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rep_count = clean_reps = 0
            stage = "UP"; rep_had_error = False
            rep_was_clean = None; rep_fail_reason = ""
            min_angle_this_rep = 180.0
            heel_baseline = None
            session_log.clear()
            rep_errors_this_rep = []
            reset_smoother()
            print("  Reset.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\n✅ Done.  Total: {rep_count}  |  Clean: {clean_reps}")

    # ── Post-session Gemini report ─────────────────────────────────────────
    if session_log:
        print("\n🤖 Opening Gemini coaching report...")
        prompt = build_gemini_prompt(session_log, rep_count, clean_reps)
        show_report_window(prompt)
    else:
        print("   (No reps recorded — skipping AI report.)")


if __name__ == "__main__":
    main()
