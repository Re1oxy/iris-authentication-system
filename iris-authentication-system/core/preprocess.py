import cv2
import numpy as np

class Preprocessor:
    """Image preprocessing and normalization"""

    def preprocess(self, iris_img, size=(64, 64)):
        gray = cv2.cvtColor(iris_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, size)
        normalized = resized / 255.0
        return normalized.astype(np.float32)