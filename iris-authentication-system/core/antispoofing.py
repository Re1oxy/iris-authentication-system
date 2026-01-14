from sklearn.ensemble import IsolationForest


class AntiSpoofingDetector:
    """
    Detects anomalous feature vectors
    that may indicate spoofing attempts.
    """

    def __init__(self, contamination=0.05):
        self.model = IsolationForest(contamination=contamination)

    def fit(self, X):
        self.model.fit(X)

    def is_genuine(self, X):
        """
        Returns True if sample is genuine,
        False if suspected spoofing.
        """
        prediction = self.model.predict(X)
        return prediction[0] == 1
