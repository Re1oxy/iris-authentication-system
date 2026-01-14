import numpy as np
from core.antispoofing import AntiSpoofingDetector


def test_genuine_sample():
    X = np.random.normal(size=(100, 5))
    detector = AntiSpoofingDetector()
    detector.fit(X)

    sample = np.random.normal(size=(1, 5))
    assert detector.is_genuine(sample)
