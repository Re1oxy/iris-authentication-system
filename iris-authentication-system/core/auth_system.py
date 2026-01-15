from core.data_loader import DataLoader
from core.preprocessor import Preprocessor
from core.classifiers import ClassifierFactory
from core.evaluator import Evaluator
from core.antispoofing import AntiSpoofingDetector

CONFIDENCE_THRESHOLD = 0.7


class AuthSystem:
    """Main iris authentication pipeline."""

    def run(self):
        loader = DataLoader(
            train_path="dataset/database.csv",
            test_path="dataset/test_dataset.csv",
            label_column= 0
        )

        X_train, y_train = loader.load_train()
        X_test, y_test = loader.load_test()

        pre = Preprocessor()
        X_train = pre.fit_transform(X_train)
        X_test = pre.transform(X_test)

        model = ClassifierFactory.svm(C=1.0)
        model.fit(X_train, y_train)

        acc, matrix = Evaluator.evaluate(model, X_test, y_test)

        print(f"Accuracy: {acc:.4f}")
        print("Confusion matrix:")
        print(matrix)

        anti_spoof = AntiSpoofingDetector()
        anti_spoof.fit(X_train)

        sample = X_test[0].reshape(1, -1)

        if not anti_spoof.is_genuine(sample):
            print("Access denied: spoofing suspected")
            return

        proba = model.predict_proba(sample)
        confidence = max(proba[0])

        if confidence < CONFIDENCE_THRESHOLD:
            print("Access denied: low confidence")
        else:
            print("Access granted")





if __name__ == "__main__":
    AuthSystem().run()

