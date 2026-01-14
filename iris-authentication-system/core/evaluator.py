from sklearn.metrics import accuracy_score, confusion_matrix


class Evaluator:
    """Model evaluation utilities."""

    @staticmethod
    def evaluate(model, X_test, y_test):
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        matrix = confusion_matrix(y_test, predictions)
        return accuracy, matrix
