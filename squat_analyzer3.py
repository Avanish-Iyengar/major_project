"""
╔══════════════════════════════════════════════════════╗
║   XAI Physiotherapy — Squat Analyzer & Rep Counter   ║
║   MediaPipe 0.10+ (Tasks API)                        ║
╚══════════════════════════════════════════════════════╝

Setup:
    pip install mediapipe opencv-python numpy

    Download model file once (in same folder as script):
    curl -o pose_landmarker.task \
      "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"

Run:
    python squat_analyzer.py
"""

import cv2
import numpy as np
import time, sys, os

#===========================================================================================================
import threading
import pygame
pygame.mixer.init()

AUDIO_MAP = {
    "Knees caving in — push knees out":   "audio/caving.mp3",
    "Leaning too far forward — chest up": "audio/chest_up.mp3",
    "Leaning too far forward - chest up": "audio/chest_up.mp3",  # side view variant
    "Knee past toes - sit back more":     "audio/toes.mp3",
    "Not deep enough":                    "audio/not_deep.mp3",
}
_last_spoken: dict[str, float] = {}
TTS_COOLDOWN = 4.0  # seconds before same alert can repeat

def speak(text):
    now = time.time()
    if now - _last_spoken.get(text, 0) < TTS_COOLDOWN:
        return
    _last_spoken[text] = now
    path = AUDIO_MAP.get(text)
    if path:
        def _play():
            sound = pygame.mixer.Sound(path)
            sound.play()
        threading.Thread(target=_play, daemon=True).start()

#=================================================================================================================================================

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:
    print("❌ Run: pip install mediapipe opencv-python numpy")
    sys.exit(1)

MODEL_PATH = "pose_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print(f"""
❌ Model file not found: {MODEL_PATH}
Download:
  curl -o pose_landmarker.task \\
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
""")
    sys.exit(1)

'''
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 160)   # speaking speed (words/min)
    _tts_engine.setProperty("volume", 1.0)
    TTS_AVAILABLE = True
except Exception:
    TTS_AVAILABLE = False
    print("⚠  pyttsx3 not found — install with: pip install pyttsx3")
    print("   Running without voice alerts.")

# Queue for non-blocking TTS (speak in a background thread)
_tts_queue  = queue.Queue()
_last_spoken: dict[str, float] = {}   # alert text → last time it was spoken
TTS_COOLDOWN = 4.0   # seconds before the same alert can be spoken again

'''

# ─── Landmark indices ─────────────────────────────────────────────────────────

LM = {
    "left_shoulder":  11, "right_shoulder": 12,
    "left_hip":       23, "right_hip":      24,
    "left_knee":      25, "right_knee":     26,
    "left_ankle":     27, "right_ankle":    28,
    "left_foot":      31, "right_foot":     32,
}

SKELETON_CONNECTIONS = [
    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),     ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),           ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),         ("right_knee", "right_ankle"),
    ("left_ankle", "left_foot"),         ("right_ankle", "right_foot"),
]

# ─── Thresholds ───────────────────────────────────────────────────────────────

SQUAT_DOWN_ANGLE  = 105   # knee angle below this  → stage "DOWN"
SQUAT_UP_ANGLE    = 155   # knee angle above this  → stage "UP" (count rep)

# Depth: warn only if we're mid-squat (stage==DOWN) and angle hasn't dropped far enough
DEPTH_INSUFFICIENT_ANGLE = 120  # if still above this while in DOWN stage → not deep enough

BACK_LEAN_MAX     = 55    # torso lean from vertical (degrees)
KNEE_CAVE_RATIO   = 0.72  # front view: knee_width / ankle_width

# Side view: knee-past-toes uses pixel X (when sideways, X = depth axis)
# Positive means knee is in front of ankle (toward camera)
KNEE_TOE_X_RATIO  = 0.06  # knee.x ahead of ankle.x as fraction of frame width

# ─── Math helpers ─────────────────────────────────────────────────────────────

def angle_2d(a, b, c):
    """Angle at B using only (x, y) pixel coords."""
    a, b, c = np.array(a[:2], float), np.array(b[:2], float), np.array(c[:2], float)
    ba, bc  = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def angle_3d(a, b, c):
    """Angle at B using (x, y, z) normalised coords from MediaPipe."""
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

def px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)

def detect_view(lms):
    """
    If shoulders have a large Z difference → person is turned sideways.
    MediaPipe Z is relative to hip depth; left vs right shoulder differ a lot when sideways.
    """
    z_diff = abs(lms[LM["left_shoulder"]].z - lms[LM["right_shoulder"]].z)
    return "SIDE" if z_diff > 0.12 else "FRONT"

# ─── Posture checks ───────────────────────────────────────────────────────────

def torso_lean(sh_px, hip_px):
    """Returns lean angle (degrees) of torso from vertical."""
    dx = sh_px[0] - hip_px[0]
    dy = max(hip_px[1] - sh_px[1], 1)
    return float(np.degrees(np.arctan2(abs(dx), dy)))


def check_posture_front(lms, pts, w):
    alerts = []

    # Front view: 2D pixel angles work correctly here.
    # When facing the camera, Y-axis captures the knee bend perfectly.
    # 3D angles are unreliable front-on because hip/knee/ankle share similar Z depth.
    kl = angle_2d(pts["left_hip"],  pts["left_knee"],  pts["left_ankle"])
    kr = angle_2d(pts["right_hip"], pts["right_knee"], pts["right_ankle"])
    avg = (kl + kr) / 2

    # 1. Knee cave (compare pixel X widths — valid for front view)
    knee_w  = abs(pts["left_knee"][0]  - pts["right_knee"][0])
    ankle_w = abs(pts["left_ankle"][0] - pts["right_ankle"][0])
    if ankle_w > 10 and knee_w < ankle_w * KNEE_CAVE_RATIO:
        alerts.append("Knees caving in — push knees out")

    # 2. Torso lean
    mid_sh  = ((pts["left_shoulder"][0] + pts["right_shoulder"][0]) // 2,
                (pts["left_shoulder"][1] + pts["right_shoulder"][1]) // 2)
    mid_hip = ((pts["left_hip"][0] + pts["right_hip"][0]) // 2,
                (pts["left_hip"][1] + pts["right_hip"][1]) // 2)
    lean = torso_lean(mid_sh, mid_hip)
    if lean > BACK_LEAN_MAX:
        alerts.append("Leaning too far forward — chest up")

    return alerts, lean, kl, kr


def check_posture_side(lms, pts, w):
    alerts = []

    # Pick the side closer to camera (more negative Z = closer)
    use_left = lms[LM["left_shoulder"]].z < lms[LM["right_shoulder"]].z

    if use_left:
        hip_lm, knee_lm, ankle_lm = lms[LM["left_hip"]], lms[LM["left_knee"]], lms[LM["left_ankle"]]
        hip_px, knee_px, ankle_px = pts["left_hip"], pts["left_knee"], pts["left_ankle"]
        sh_px = pts["left_shoulder"]
    else:
        hip_lm, knee_lm, ankle_lm = lms[LM["right_hip"]], lms[LM["right_knee"]], lms[LM["right_ankle"]]
        hip_px, knee_px, ankle_px = pts["right_hip"], pts["right_knee"], pts["right_ankle"]
        sh_px = pts["right_shoulder"]

    # ── Knee angle: use 3D coordinates (most reliable regardless of view) ──
    k = angle_3d(hip_lm, knee_lm, ankle_lm)
    kl = kr = k  # only one visible side for display purposes

    # 1. Knee past toes — side view: use PIXEL X
    #    When standing sideways, X axis = depth.
    #    If knee.x is significantly ahead of ankle.x → knee shooting forward.
    #    Which direction "forward" is depends on which way they face.
    #    We compare knee_px.x - ankle_px.x; significant positive OR negative = problem.
    knee_toe_x = (knee_px[0] - ankle_px[0]) / w
    if abs(knee_toe_x) > KNEE_TOE_X_RATIO:
        alerts.append("Knee past toes - sit back more")

    # 2. Torso lean
    lean = torso_lean(sh_px, hip_px)
    if lean > BACK_LEAN_MAX:
        alerts.append("Leaning too far forward - chest up")

    return alerts, lean, kl, kr

# ─── Colors & Drawing ─────────────────────────────────────────────────────────

C = {
    "green":  (0, 220, 100),  "red":    (30, 60, 240),
    "yellow": (0, 200, 255),  "white":  (240, 240, 240),
    "accent": (0, 165, 255),  "dim":    (100, 100, 130),
    "panel":  (12, 12, 22),
}

def draw_skeleton(frame, pts, color):
    for a, b in SKELETON_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for pt in pts.values():
        cv2.circle(frame, pt, 6, color, -1)
        cv2.circle(frame, pt, 8, (255, 255, 255), 1)

def draw_hud(frame, reps, clean_reps, stage, alerts, fps, lean, kl, kr, view, rep_was_clean, rep_fail_reason, min_angle_this_rep=180.0):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (300, h), C["panel"], -1)
    cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)

    # Title
    cv2.putText(frame, "SQUAT ANALYZER", (12, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, C["accent"], 1, cv2.LINE_AA)
    cv2.putText(frame, f"[{view} VIEW]", (12, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    cv2.line(frame, (12, 58), (288, 58), C["accent"], 1)

    # Rep counters
    cv2.putText(frame, "TOTAL", (12, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["white"], 1)
    cv2.putText(frame, str(reps), (12, 135), cv2.FONT_HERSHEY_DUPLEX, 3.0, C["white"], 2, cv2.LINE_AA)
    cv2.putText(frame, "CLEAN", (160, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C["green"], 1)
    cv2.putText(frame, str(clean_reps), (160, 135), cv2.FONT_HERSHEY_DUPLEX, 3.0, C["green"], 2, cv2.LINE_AA)

    # Stage badge
    s_col = C["green"] if stage == "UP" else C["yellow"]
    cv2.rectangle(frame, (12, 145), (145, 165), s_col, -1)
    cv2.putText(frame, f" STAGE: {stage}", (14, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C["panel"], 1)

    # Last rep result
    if rep_was_clean is not None:
        if rep_was_clean:
            cv2.putText(frame, "CLEAN REP", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, C["green"], 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, f"BAD REP: {rep_fail_reason}", (12, 182),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["red"], 1, cv2.LINE_AA)

    # Metrics
    cv2.line(frame, (12, 192), (288, 192), (40, 40, 60), 1)
    metrics = [f"Knee (L): {int(kl)}", f"Knee (R): {int(kr)}",
               f"Torso lean: {int(lean)}", f"FPS: {fps:.0f}"]
    for i, m in enumerate(metrics):
        cv2.putText(frame, m, (12, 212 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 210), 1)

    # Depth progress bar (only meaningful when squatting)
    # Maps angle: 155° (standing) → 80° (deep squat) onto a bar
    bar_y = 304
    cv2.putText(frame, "DEPTH", (12, bar_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["dim"], 1)
    bar_w   = 200
    bar_h   = 10
    angle_high, angle_low = 155.0, 80.0
    current_angle = (kl + kr) / 2
    depth_ratio   = max(0.0, min(1.0, (angle_high - current_angle) / (angle_high - angle_low)))
    cv2.rectangle(frame, (12, bar_y), (12 + bar_w, bar_y + bar_h), (40, 40, 60), -1)
    bar_color = C["green"] if depth_ratio >= ((angle_high - DEPTH_INSUFFICIENT_ANGLE) / (angle_high - angle_low)) else C["yellow"]
    cv2.rectangle(frame, (12, bar_y), (12 + int(bar_w * depth_ratio), bar_y + bar_h), bar_color, -1)
    # Target line
    target_x = 12 + int(bar_w * ((angle_high - DEPTH_INSUFFICIENT_ANGLE) / (angle_high - angle_low)))
    cv2.line(frame, (target_x, bar_y - 2), (target_x, bar_y + bar_h + 2), C["white"], 1)

    # Alerts
    cv2.line(frame, (12, 328), (288, 328), (40, 40, 60), 1)
    if alerts:
        cv2.putText(frame, "! ALERTS", (12, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C["red"], 1)
        y = 370
        for alert in alerts[:3]:
            words, line = alert.split(), ""
            for word in words:
                if len(line + word) < 28:
                    line += word + " "
                else:
                    cv2.putText(frame, line.strip(), (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["red"], 1)
                    y += 16; line = word + " "
            cv2.putText(frame, line.strip(), (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C["red"], 1)
            y += 22
    else:
        cv2.putText(frame, "POSTURE OK", (12, 348), cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["green"], 1)

    cv2.putText(frame, "Q = quit   R = reset", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, C["dim"], 1)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
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

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    rep_count        = 0
    clean_reps       = 0
    stage            = "UP"
    rep_had_error    = False
    rep_fail_reason  = ""      # reason the last rep failed
    rep_was_clean    = None
    min_angle_this_rep = 180.0  # track deepest point reached this rep
    alerts           = []
    lean = kl = kr   = 0.0
    fps              = 0.0
    prev_time        = time.time()
    view             = "FRONT"

    print("\n🏋  Squat Analyzer running.")
    print("   Auto-detects FRONT or SIDE view.")
    print("   Only reps with correct posture throughout count as CLEAN.")
    print("   Q = quit   R = reset\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int(time.time() * 1000))

        if result.pose_landmarks:
            lms  = result.pose_landmarks[0]
            pts  = {name: px(lms[idx], w, h) for name, idx in LM.items()}
            view = detect_view(lms)

            if view == "SIDE":
                alerts, lean, kl, kr = check_posture_side(lms, pts, w)
            else:
                alerts, lean, kl, kr = check_posture_front(lms, pts, w)

            avg = (kl + kr) / 2

            # ── Rep state machine ─────────────────────────────────────────
            # UP → DOWN transition
            if avg < SQUAT_DOWN_ANGLE and stage == "UP":
                stage              = "DOWN"
                rep_had_error      = False
                rep_fail_reason    = ""
                min_angle_this_rep = avg   # start tracking depth

            if stage == "DOWN":
                min_angle_this_rep = min(min_angle_this_rep, avg)
                if alerts:
                    speak(alerts[0])          # ← voice fires here
                    if not rep_had_error:
                        rep_had_error   = True
                        rep_fail_reason = alerts[0]
            # DOWN → UP transition: evaluate the completed rep
            if avg > SQUAT_UP_ANGLE and stage == "DOWN":
                stage      = "UP"
                rep_count += 1

                # Check depth: did they reach the required angle?
                depth_ok = min_angle_this_rep <= DEPTH_INSUFFICIENT_ANGLE

                if not depth_ok and not rep_had_error:
                    rep_had_error   = True
                    rep_fail_reason = "Not deep enough"

                if rep_had_error:
                    rep_was_clean = False
                    print(f"  ✗ Rep {rep_count} — {rep_fail_reason} "
                          f"(depth reached: {int(min_angle_this_rep)}°)")
                else:
                    clean_reps   += 1
                    rep_was_clean = True
                    print(f"  ✅ Rep {rep_count} — clean! "
                          f"(depth: {int(min_angle_this_rep)}°, clean total: {clean_reps})")

            skel_color = C["red"] if alerts else C["green"]
            draw_skeleton(frame, pts, skel_color)

            for name, angle in [("left_knee", kl), ("right_knee", kr)]:
                pt = pts[name]
                cv2.putText(frame, f"{int(angle)}", (pt[0] + 10, pt[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, skel_color, 1, cv2.LINE_AA)
        else:
            alerts = []

        now       = time.time()
        fps       = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        draw_hud(frame, rep_count, clean_reps, stage, alerts, fps, lean, kl, kr, view, rep_was_clean, rep_fail_reason, min_angle_this_rep)
        cv2.imshow("XAI Squat Analyzer", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            rep_count = clean_reps = 0
            stage = "UP"; rep_had_error = False
            rep_was_clean = None; rep_fail_reason = ""
            min_angle_this_rep = 180.0
            print("  Reset.")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\n✅ Done.  Total: {rep_count}  |  Clean: {clean_reps}")

if __name__ == "__main__":
    main()
