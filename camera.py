# camera.py
import cv2, threading
from config import CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT

class ThreadedCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self._lock = threading.Lock()
        self._running = True
        threading.Thread(target=self._grab, daemon=True).start()

    def _grab(self):
        while self._running:
            ret, frame = self.cap.read()
            with self._lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self._lock:
            return self.ret, self.frame.copy() if self.ret else (False, None)

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self._running = False
        self.cap.release()

def init_camera():
    return ThreadedCamera()
