"""
╔══════════════════════════════════════════════════════════════════╗
║               EchoVision  –  src/camera.py                       ║
║  Threaded video capture so frame grabbing never blocks the       ║
║  detection/inference loop (reduces perceived latency).           ║
╚══════════════════════════════════════════════════════════════════╝
"""

import threading
import time

import cv2

from .utils import log


class VideoStream:
    """
    Wraps cv2.VideoCapture with a background thread that always keeps the
    *latest* frame ready, instead of letting frames queue up and go stale
    (important for a real-time navigation aid — you want NOW, not 300ms ago).
    """

    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        self._lock = threading.Lock()
        self._frame = None
        self._ret = False
        self._stopped = False

        ret, frame = self.cap.read()
        self._ret, self._frame = ret, frame

        self._thread = threading.Thread(target=self._update, daemon=True)
        self._thread.start()
        log.info(f"VideoStream started on source={source}")

    def _update(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            with self._lock:
                self._ret, self._frame = ret, frame
            if not ret:
                time.sleep(0.05)

    def read(self):
        with self._lock:
            return self._ret, (self._frame.copy() if self._frame is not None else None)

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS) or 25.0

    def stop(self):
        self._stopped = True
        self._thread.join(timeout=1.0)
        self.cap.release()
        log.info("VideoStream stopped")
