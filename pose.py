import time
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from config import MODEL_PATH, DETECTION_SCALE

def setup_pose():
    """Create MediaPipe PoseLandmarker with configured model path."""
    opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.6,
        min_pose_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_segmentation_masks=False,
    )
    return mp_vision.PoseLandmarker.create_from_options(opts)

def detect_pose(landmarker, frame):
    """Run pose detection on a frame and return MediaPipe result."""
    h, w = frame.shape[:2]
    det_w = int(w * DETECTION_SCALE)
    det_h = int(h * DETECTION_SCALE)
    small = cv2.resize(frame, (det_w, det_h))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_img, int(time.time() * 1000))
