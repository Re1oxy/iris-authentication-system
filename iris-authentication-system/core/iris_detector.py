import cv2
import mediapipe as mp


class IrisDetector:
    """Detect iris region using MediaPipe Face Mesh."""

    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            refine_landmarks=True
        )

    def extract_iris(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            return None

        h, w, _ = frame.shape
        iris_points = []

        for lm in result.multi_face_landmarks[0].landmark[468:472]:
            x = int(lm.x * w)
            y = int(lm.y * h)
            iris_points.append((x, y))

        x_coords, y_coords = zip(*iris_points)
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return frame[y_min:y_max, x_min:x_max]
