import cv2
import numpy as np

try:
    # New mediapipe API (0.10+)
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    _NEW_API = True
except ImportError:
    _NEW_API = False

try:
    # Legacy API fallback
    import mediapipe as mp
    _face_mesh_legacy = mp.solutions.face_mesh
    _NEW_API = False
except AttributeError:
    pass


class IrisDetector:
    """Detect iris region using MediaPipe Face Mesh."""

    def __init__(self):
        self._use_new_api = False
        self._detector = None
        self.face_mesh = None

        try:
            # Try legacy API first (most common)
            import mediapipe as mp
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                refine_landmarks=True,
                max_num_faces=1
            )
            self._use_new_api = False
        except AttributeError:
            # Try new Tasks API
            try:
                import mediapipe as mp
                from mediapipe.tasks import python
                from mediapipe.tasks.python import vision

                # Download model if needed
                import urllib.request, os
                model_path = "face_landmarker.task"
                if not os.path.exists(model_path):
                    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
                    print(f"Downloading MediaPipe model...")
                    urllib.request.urlretrieve(url, model_path)

                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.FaceLandmarkerOptions(
                    base_options=base_options,
                    output_face_blendshapes=False,
                    output_facial_transformation_matrixes=False,
                    num_faces=1
                )
                self._detector = vision.FaceLandmarker.create_from_options(options)
                self._use_new_api = True
            except Exception as e:
                print(f"MediaPipe init failed: {e}")
                self._use_fallback = True

    def extract_iris(self, frame):
        """Extract iris crop from frame. Returns cropped image or None."""
        if frame is None:
            return None

        try:
            if self._use_new_api:
                return self._extract_new_api(frame)
            elif self.face_mesh is not None:
                return self._extract_legacy(frame)
            else:
                return self._extract_haar(frame)
        except Exception:
            return self._extract_haar(frame)

    # MediaPipe eyelid landmark indices for left eye
    # Top eyelid: 159, Bottom eyelid: 145, Left corner: 33, Right corner: 133
    EAR_TOP    = 159
    EAR_BOTTOM = 145
    EAR_LEFT   = 33
    EAR_RIGHT  = 133
    EAR_THRESHOLD = 0.15  # below this = eye closed

    def _eye_aspect_ratio(self, landmarks, h, w):
        """Compute Eye Aspect Ratio. Returns 0.0 if eye closed."""
        def pt(idx):
            lm = landmarks[idx]
            return np.array([lm.x * w, lm.y * h])
        top    = pt(self.EAR_TOP)
        bottom = pt(self.EAR_BOTTOM)
        left   = pt(self.EAR_LEFT)
        right  = pt(self.EAR_RIGHT)
        vertical   = np.linalg.norm(top - bottom)
        horizontal = np.linalg.norm(left - right)
        if horizontal < 1e-6:
            return 0.0
        return vertical / horizontal

    def _extract_legacy(self, frame):
        """Legacy mediapipe solutions API."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        h, w, _ = frame.shape
        landmarks = result.multi_face_landmarks[0].landmark

        # Reject closed eye
        ear = self._eye_aspect_ratio(landmarks, h, w)
        if ear < self.EAR_THRESHOLD:
            return None

        iris_points = []
        for lm in landmarks[468:472]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            iris_points.append((x, y))

        return self._crop_iris(frame, iris_points, h, w)

    def _extract_new_api(self, frame):
        """New mediapipe Tasks API."""
        import mediapipe as mp
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if not result.face_landmarks:
            return None

        h, w, _ = frame.shape
        iris_points = []
        for lm in result.face_landmarks[0][468:472]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            iris_points.append((x, y))

        return self._crop_iris(frame, iris_points, h, w)

    def _extract_haar(self, frame):
        """
        Fallback: Haar cascade eye detector.
        Less precise but works without mediapipe.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        if len(eyes) == 0:
            return None

        x, y, ew, eh = eyes[0]
        return frame[y:y+eh, x:x+ew]

    def _crop_iris(self, frame, iris_points, h, w):
        if not iris_points:
            return None
        x_coords, y_coords = zip(*iris_points)
        x_min, x_max = max(0, min(x_coords) - 40), min(w, max(x_coords) + 40)
        y_min, y_max = max(0, min(y_coords) - 40), min(h, max(y_coords) + 40)
        crop = frame[y_min:y_max, x_min:x_max]
        return crop if crop.size > 0 else None