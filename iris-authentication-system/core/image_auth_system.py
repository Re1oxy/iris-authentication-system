import numpy as np
import json
import os
from core.iris_detector import IrisDetector
from core.image_processor import ImageProcessor
from core.preprocessor import Preprocessor
from core.classifiers import ClassifierFactory
from core.antispoofing import AntiSpoofingDetector

CONFIDENCE_THRESHOLD = 0.45
DB_PATH = "dataset/iris_db.json"
MIN_SAMPLES_TO_TRAIN = 2  # minimum unique users to enable auth


class ImageAuthSystem:
    """
    Iris authentication pipeline using real camera images.
    Replaces CSV-based system with LBP feature extraction from live frames.
    """

    def __init__(self):
        self.iris_detector = IrisDetector()
        self.image_processor = ImageProcessor()
        self.preprocessor = Preprocessor()
        self.model = None
        self.anti_spoof = None
        self.is_trained = False
        self.user_labels = {}  # name -> int label
        self._next_label = 0

        # In-memory database: label -> list of feature vectors
        self.feature_db: dict[int, list] = {}
        self._auth_buffer: list = []  # rolling buffer for stable auth

        self._load_db()

    # ------------------------------------------------------------------ #
    #  Database persistence
    # ------------------------------------------------------------------ #

    def _load_db(self):
        """Load registered users from JSON file."""
        if not os.path.exists(DB_PATH):
            return

        with open(DB_PATH, "r") as f:
            data = json.load(f)

        self.user_labels = data.get("user_labels", {})
        self._next_label = data.get("next_label", 0)
        raw_db = data.get("feature_db", {})
        self.feature_db = {int(k): [np.array(v) for v in vecs]
                           for k, vecs in raw_db.items()}

        if len(self.user_labels) >= MIN_SAMPLES_TO_TRAIN:
            self._train()

    def _save_db(self):
        """Persist database to disk."""
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        data = {
            "user_labels": self.user_labels,
            "next_label": self._next_label,
            "feature_db": {k: [v.tolist() for v in vecs]
                           for k, vecs in self.feature_db.items()}
        }
        with open(DB_PATH, "w") as f:
            json.dump(data, f)

    # ------------------------------------------------------------------ #
    #  Model training
    # ------------------------------------------------------------------ #

    def _train(self):
        """(Re)train classifier on all stored samples."""
        X, y = [], []
        for label, vectors in self.feature_db.items():
            for vec in vectors:
                X.append(vec)
                y.append(label)

        if len(set(y)) < MIN_SAMPLES_TO_TRAIN:
            return

        X = np.array(X)
        y = np.array(y)

        X_scaled = self.preprocessor.fit_transform(X)

        self.model = ClassifierFactory.svm(C=1.0)
        self.model.fit(X_scaled, y)

        self.anti_spoof = AntiSpoofingDetector(contamination=0.15)
        self.anti_spoof.fit(X_scaled)

        self.is_trained = True

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #

    def extract_iris_features(self, frame: np.ndarray):
        """
        Given a raw camera frame, returns (iris_crop, feature_vector).
        Returns (None, None) if iris not detected.
        """
        iris_crop = self.iris_detector.extract_iris(frame)
        if iris_crop is None or iris_crop.size == 0:
            return None, None

        features = self.image_processor.extract_features(iris_crop)
        return iris_crop, features

    def register_user(self, name: str, feature_vectors: list[np.ndarray]) -> str:
        """
        Register a new user with collected feature vectors.
        Returns status message.
        """
        name = name.strip()
        if not name:
            return "error_empty_name"

        if name not in self.user_labels:
            self.user_labels[name] = self._next_label
            self._next_label += 1

        label = self.user_labels[name]
        if label not in self.feature_db:
            self.feature_db[label] = []

        self.feature_db[label].extend(feature_vectors)
        self._save_db()

        if len(self.user_labels) >= MIN_SAMPLES_TO_TRAIN:
            self._train()

        return "registered"

    def authenticate(self, feature_vector: np.ndarray) -> tuple[str, float, str]:
        """
        Authenticate a feature vector.
        Returns (decision, confidence, username).
        Decisions: 'granted', 'denied_spoof', 'denied_low_confidence', 'not_trained'
        """
        if not self.is_trained:
            return "not_trained", 0.0, ""

        sample = self.preprocessor.transform(feature_vector.reshape(1, -1))

        # Add to rolling buffer for averaged prediction
        self._auth_buffer.append(sample[0])
        if len(self._auth_buffer) > 5:
            self._auth_buffer.pop(0)

        # Use averaged features for more stable result
        avg_sample = np.mean(self._auth_buffer, axis=0).reshape(1, -1)

        # Anti-spoof disabled for webcam demo (SVM confidence handles rejection)
        proba = self.model.predict_proba(avg_sample)[0]
        confidence = float(max(proba))
        predicted_label = int(self.model.predict(avg_sample)[0])

        # Reverse lookup: label -> name
        username = next(
            (name for name, lbl in self.user_labels.items()
             if lbl == predicted_label), "Unknown"
        )

        if confidence < CONFIDENCE_THRESHOLD:
            return "denied_low_confidence", confidence, ""

        return "granted", confidence, username

    def get_registered_users(self) -> list[str]:
        return list(self.user_labels.keys())

    def delete_user(self, name: str) -> bool:
        if name not in self.user_labels:
            return False
        label = self.user_labels.pop(name)
        self.feature_db.pop(label, None)
        self._save_db()
        self.is_trained = False
        if len(self.user_labels) >= MIN_SAMPLES_TO_TRAIN:
            self._train()
        return True