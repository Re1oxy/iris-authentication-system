class Camera:
    """
    Camera abstraction class.
    Responsible for capturing frames from a video device.
    """

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self.capture = None

    def open(self):
        """Open camera device."""
        pass

    def read(self):
        """Read single frame from camera."""
        pass

    def release(self):
        """Release camera resource."""
        pass
