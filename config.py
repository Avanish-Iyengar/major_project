# ╔══════════════════════════════════════════════════════════════════╗
# ║  config.py — All user-configurable variables                    ║
# ║  Edit this file to tune the analyzer without touching any logic ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 1. MediaPipe model ────────────────────────────────────────────────────────
# Lite  = faster, slightly less accurate. Recommended for real-time.
# Full  = slower, more accurate.
MODEL_PATH = "pose_landmarker.task"

# ── 2. API keys ───────────────────────────────────────────────────────────────
XI_KEY         = "YOUR_ELEVENLABS_API_KEY"  # elevenlabs.io → Profile → API Key
VOICE_ID       = "YOUR_VOICE_ID"            # elevenlabs.io → Voices → copy ID
GEMINI_API_KEY = "AIzaSyCg4fN9kFSYCHa_XPHZF3lNrqpIXCHYC8g"      # aistudio.google.com → Get API Key
OLLAMA_MODEL = "gemma3:4b" 
OLLAMA_URL = "http://localhost:11434"

# ── 3. Camera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0       # 0 = default webcam. Try 1 or 2 if wrong camera opens.
CAMERA_WIDTH    = 1280    # Reduce to 854 if FPS stays below ~20
CAMERA_HEIGHT   = 720     # Reduce to 480 if FPS stays below ~20
DETECTION_SCALE = 0.9     # MediaPipe runs at this fraction of full resolution.
                          # 0.5 = 2–3× faster. Raise toward 1.0 if joints drift.

# ── 4. Smoothing ──────────────────────────────────────────────────────────────
EMA_ALPHA         = 1.0   # Landmark smoothing. Lower = smoother, more lag.
                          # Range: 0.2 (buttery) → 0.8 (nearly raw)
VISIBILITY_THRESH = 0.4   # Landmarks below this confidence are ignored.
                          # Lower to 0.4 if joints disappear; raise to 0.75 for fewer phantoms.

# ── 5. Voice ──────────────────────────────────────────────────────────────────
TTS_COOLDOWN = 4.0        # Seconds before the same alert can repeat.

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

