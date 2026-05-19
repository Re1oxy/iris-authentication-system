import cv2
import numpy as np
from skimage.feature import local_binary_pattern


class ImageProcessor:
    """
    Extracts feature vectors from iris images.
    Uses LBP (Local Binary Patterns) — works well for texture analysis.
    Replaces CSV-based DataLoader for real image pipeline.
    """

    LBP_RADIUS = 3
    LBP_POINTS = 24
    FEATURE_SIZE = 64  # normalized output size

    def extract_features(self, iris_img: np.ndarray) -> np.ndarray | None:
        """
        Given a cropped iris image, returns a 1D feature vector.
        Returns None if image is invalid.
        """
        if iris_img is None or iris_img.size == 0:
            return None

        # Convert to grayscale
        if len(iris_img.shape) == 3:
            gray = cv2.cvtColor(iris_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = iris_img

        # Resize to fixed size for consistent features
        resized = cv2.resize(gray, (self.FEATURE_SIZE, self.FEATURE_SIZE))

        # LBP texture features
        lbp = local_binary_pattern(
            resized,
            self.LBP_POINTS,
            self.LBP_RADIUS,
            method="uniform"
        )

        # Histogram of LBP patterns
        n_bins = self.LBP_POINTS + 2
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=n_bins,
            range=(0, n_bins),
            density=True
        )

        return hist.astype(np.float32)

    def preprocess_for_display(self, iris_img: np.ndarray, size=(200, 200)) -> np.ndarray:
        """Resize iris image for display in GUI."""
        if iris_img is None or iris_img.size == 0:
            return np.zeros((size[1], size[0], 3), dtype=np.uint8)
        return cv2.resize(iris_img, size)