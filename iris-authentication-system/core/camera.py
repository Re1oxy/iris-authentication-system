import cv2


class Camera:
    """Camera abstraction using OpenCV."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.capture = None

    def open(self):
        self.capture = cv2.VideoCapture(self.device_id)
        if not self.capture.isOpened():
            raise RuntimeError("Cannot open camera")

    def read(self):
        if self.capture is None:
            raise RuntimeError("Camera not opened")
        ret, frame = self.capture.read()
        if not ret:
            raise RuntimeError("Cannot read frame")
        return frame

    def release(self):
        if self.capture:
            self.capture.release()
