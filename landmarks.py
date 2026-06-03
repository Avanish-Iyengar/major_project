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

    # Hand landmarks (MediaPipe Hand landmarks)
    "left_wrist_hand":  0,   "right_wrist_hand":  0,
    "left_thumb_cmc":   1,   "right_thumb_cmc":   1,
    "left_thumb_mcp":   2,   "right_thumb_mcp":   2,
    "left_thumb_ip":    3,   "right_thumb_ip":    3,
    "left_thumb_tip":   4,   "right_thumb_tip":   4,
    "left_index_mcp":   5,   "right_index_mcp":   5,
    "left_index_pip":   6,   "right_index_pip":   6,
    "left_index_dip":   7,   "right_index_dip":   7,
    "left_index_tip":   8,   "right_index_tip":   8,
    "left_middle_mcp":  9,   "right_middle_mcp":  9,
    "left_middle_pip": 10,   "right_middle_pip": 10,
    "left_middle_dip": 11,   "right_middle_dip": 11,
    "left_middle_tip": 12,   "right_middle_tip": 12,
    "left_ring_mcp":   13,   "right_ring_mcp":   13,
    "left_ring_pip":   14,   "right_ring_pip":   14,
    "left_ring_dip":   15,   "right_ring_dip":   15,
    "left_ring_tip":   16,   "right_ring_tip":   16,
    "left_pinky_mcp":  17,   "right_pinky_mcp":  17,
    "left_pinky_pip":  18,   "right_pinky_pip":  18,
    "left_pinky_dip":  19,   "right_pinky_dip":  19,
    "left_pinky_tip":  20,   "right_pinky_tip":  20,

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

    # Hand connections - DISABLED (require separate MediaPipe Hands model)
    # ("left_wrist",     "left_thumb_cmc"),
    # ("left_thumb_cmc", "left_thumb_mcp"),
    # ("left_thumb_mcp", "left_thumb_ip"),
    # ("left_thumb_ip",  "left_thumb_tip"),
    # ("left_wrist",     "left_index_mcp"),
    # ("left_index_mcp", "left_index_pip"),
    # ("left_index_pip", "left_index_dip"),
    # ("left_index_dip", "left_index_tip"),
    # ("left_wrist",     "left_middle_mcp"),
    # ("left_middle_mcp", "left_middle_pip"),
    # ("left_middle_pip", "left_middle_dip"),
    # ("left_middle_dip", "left_middle_tip"),
    # ("left_wrist",     "left_ring_mcp"),
    # ("left_ring_mcp",  "left_ring_pip"),
    # ("left_ring_pip",  "left_ring_dip"),
    # ("left_ring_dip",  "left_ring_tip"),
    # ("left_wrist",     "left_pinky_mcp"),
    # ("left_pinky_mcp", "left_pinky_pip"),
    # ("left_pinky_pip", "left_pinky_dip"),
    # ("left_pinky_dip", "left_pinky_tip"),
    # ("right_wrist",    "right_thumb_cmc"),
    # ("right_thumb_cmc", "right_thumb_mcp"),
    # ("right_thumb_mcp", "right_thumb_ip"),
    # ("right_thumb_ip",  "right_thumb_tip"),
    # ("right_wrist",    "right_index_mcp"),
    # ("right_index_mcp", "right_index_pip"),
    # ("right_index_pip", "right_index_dip"),
    # ("right_index_dip", "right_index_tip"),
    # ("right_wrist",    "right_middle_mcp"),
    # ("right_middle_mcp", "right_middle_pip"),
    # ("right_middle_pip", "right_middle_dip"),
    # ("right_middle_dip", "right_middle_tip"),
    # ("right_wrist",    "right_ring_mcp"),
    # ("right_ring_mcp", "right_ring_pip"),
    # ("right_ring_pip", "right_ring_dip"),
    # ("right_ring_dip", "right_ring_tip"),
    # ("right_wrist",    "right_pinky_mcp"),
    # ("right_pinky_mcp", "right_pinky_pip"),
    # ("right_pinky_pip", "right_pinky_dip"),
    # ("right_pinky_dip", "right_pinky_tip"),

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
