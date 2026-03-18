# ╔══════════════════════════════════════════════════════════════════╗
# ║  landmarks.py — MediaPipe landmark indices and skeleton map     ║
# ╚══════════════════════════════════════════════════════════════════╝

# MediaPipe Pose landmark indices
LM = {
    # Head / neck
    "nose":            0,
    "left_ear":        7,   "right_ear":        8,

    # Upper body
    "left_shoulder":  11,   "right_shoulder":  12,
    "left_elbow":     13,   "right_elbow":     14,
    "left_wrist":     15,   "right_wrist":     16,

    # Lower body
    "left_hip":       23,   "right_hip":       24,
    "left_knee":      25,   "right_knee":      26,
    "left_ankle":     27,   "right_ankle":     28,
    "left_heel":      29,   "right_heel":      30,
    "left_foot":      31,   "right_foot":      32,
}

# Pairs of landmark names to connect when drawing the skeleton
SKELETON_CONNECTIONS = [
    # Head
    ("left_ear",       "left_shoulder"),
    ("right_ear",      "right_shoulder"),

    # Upper body
    ("left_shoulder",  "right_shoulder"),
    ("left_shoulder",  "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow",     "left_wrist"),
    ("right_elbow",    "right_wrist"),

    # Torso
    ("left_shoulder",  "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip",       "right_hip"),

    # Lower body
    ("left_hip",    "left_knee"),
    ("right_hip",   "right_knee"),
    ("left_knee",   "left_ankle"),
    ("right_knee",  "right_ankle"),
    ("left_ankle",  "left_heel"),
    ("right_ankle", "right_heel"),
    ("left_heel",   "left_foot"),
    ("right_heel",  "right_foot"),
]
