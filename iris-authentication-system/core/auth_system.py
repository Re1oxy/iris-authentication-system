from core.data_loader import DataLoader
from core.preprocessor import Preprocessor
from core.classifiers import ClassifierFactory
from core.evaluator import Evaluator
from core.antispoofing import AntiSpoofingDetector

CONFIDENCE_THRESHOLD = 0.7

class AuthSystem:
    """Main iris authentication pipeline."""

    def __init__(self):
        loader = DataLoader(
            train_path="dataset/database.csv",
            test_path="dataset/test_dataset.csv",
            label_column=0
        )
        X_train, y_train = loader.load_train()

        pre = Preprocessor()
        self.X_train = pre.fit_transform(X_train)

        self.model = ClassifierFactory.svm(C=1.0)
        self.model.fit(self.X_train, y_train)

        self.anti_spoof = AntiSpoofingDetector()
        self.anti_spoof.fit(self.X_train)

        self.preprocessor = pre

    def authenticate(self, sample):
        sample = self.preprocessor.transform(sample.to_numpy().reshape(1, -1)) if hasattr(sample, "to_numpy") else self.preprocessor.transform(sample)

        if not self.anti_spoof.is_genuine(sample):
            return "denied_spoof", 0.0

        proba = self.model.predict_proba(sample)
        confidence = max(proba[0])

        if confidence < CONFIDENCE_THRESHOLD:
            return "denied_low_confidence", confidence
        else:
            return "granted", confidence


    def run(self):
        """Run authentication on a sample and print results."""
        loader = DataLoader(
            train_path="dataset/database.csv",
            test_path="dataset/test_dataset.csv",
            label_column=0
        )
        X_test, _ = loader.load_test()
        sample = X_test[0].reshape(1, -1)

        decision, confidence = self.authenticate(sample)

        print(f"Decision: {decision}, Confidence: {confidence:.4f}")


if __name__ == "__main__":
    AuthSystem().run()