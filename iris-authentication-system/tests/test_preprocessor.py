import numpy as np
from core.preprocessor import Preprocessor


def test_normalization_mean():
    X = np.array([[1, 2], [3, 4]])
    pre = Preprocessor()
    Xn = pre.fit_transform(X)
    assert abs(Xn.mean()) < 1e-6
