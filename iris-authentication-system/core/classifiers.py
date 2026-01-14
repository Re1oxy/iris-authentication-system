from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class ClassifierFactory:
    """Factory for different ML classifiers."""

    @staticmethod
    def knn(n_neighbors=3):
        return KNeighborsClassifier(n_neighbors=n_neighbors)

    @staticmethod
    def svm(C=1.0, kernel="rbf"):
        return SVC(C=C, kernel=kernel)

    @staticmethod
    def random_forest(n_estimators=100):
        return RandomForestClassifier(n_estimators=n_estimators)
