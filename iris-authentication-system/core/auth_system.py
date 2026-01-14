from core.data_loader import DataLoader
from core.preprocessor import Preprocessor
from core.classifiers import ClassifierFactory
from core.evaluator import Evaluator


class AuthSystem:
    """Main iris authentication pipeline."""

    def run(self):
        loader = DataLoader(
            train_path="dataset/dataset.csv",
            test_path="dataset/test_dataset.csv",
            label_column="label"
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


if __name__ == "__main__":
    AuthSystem().run()
