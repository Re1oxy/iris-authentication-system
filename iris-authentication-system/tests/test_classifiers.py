from core.classifiers import ClassifierFactory


def test_knn_creation():
    clf = ClassifierFactory.knn()
    assert clf is not None
